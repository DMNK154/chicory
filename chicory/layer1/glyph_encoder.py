"""ByT5-based glyph embedding encoder for the glyph Ramsey lattice.

Embeds tag name strings through a ByT5 encoder, producing rich
character-level representations (d_model-dimensional float32 vectors).
Uses PCA projection to 2D for angle computation on the glyph lattice.

Model loading is lazy and thread-safe, following the same pattern as
EmbeddingEngine.  When no model directory is configured (or loading
fails), callers fall back to the letter-count SHA256 path.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Optional

import numpy as np

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine

logger = logging.getLogger(__name__)


class GlyphEncoder:
    """ByT5 encoder wrapper for glyph lattice embeddings."""

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db
        self._model = None  # AutoModelForSeq2SeqLM (encoder used)
        self._tokenizer = None  # ByT5Tokenizer
        self._device: str = "cpu"
        self._pca_basis: Optional[np.ndarray] = None  # shape (2, d_model)
        self._lock = threading.Lock()

    # ── Model Loading ─────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Lazily load the ByT5 tokenizer and model.  Thread-safe."""
        if self._model is not None:
            return

        with self._lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return

            model_dir = self._config.glyph_model_dir
            if not model_dir:
                logger.warning("glyph_model_dir is empty — GlyphEncoder disabled")
                return

            try:
                import torch
                from transformers import (
                    AutoModelForSeq2SeqLM,
                    AutoTokenizer,
                )

                resolved = str(model_dir).replace("\\", "/")
                logger.info("Loading ByT5 glyph encoder from %s", resolved)

                self._tokenizer = AutoTokenizer.from_pretrained(
                    resolved, local_files_only=True, use_fast=False,
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    resolved, local_files_only=True,
                )

                # CPU preferred on Windows (CUDA paging errors)
                self._device = "cpu"
                self._model = model.to(self._device).eval()

                logger.info(
                    "ByT5 glyph encoder loaded (d_model=%d, device=%s)",
                    model.config.d_model, self._device,
                )
            except Exception:
                logger.exception("Failed to load ByT5 glyph encoder")
                self._model = None
                self._tokenizer = None

    @property
    def is_available(self) -> bool:
        """True iff the ByT5 model loaded successfully."""
        return self._model is not None and self._tokenizer is not None

    # ── Embedding ─────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string via the ByT5 encoder.

        Returns a normalized float32 vector of shape (d_model,).
        """
        self._load_model()
        if not self.is_available:
            raise RuntimeError("GlyphEncoder model not loaded")

        import torch

        inputs = self._tokenizer(
            text, return_tensors="pt", padding=False,
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)

        with torch.no_grad():
            encoder_output = self._model.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
            )

        hidden = encoder_output.last_hidden_state  # (1, seq_len, d_model)
        mask = attention_mask.unsqueeze(-1).float()  # (1, seq_len, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (1, d_model)

        vec = pooled.squeeze(0).cpu().numpy().astype(np.float32)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple strings in one forward pass.

        Returns a normalized float32 array of shape (N, d_model).
        """
        if not texts:
            return np.empty((0, self._config.glyph_embedding_dimension), dtype=np.float32)

        self._load_model()
        if not self.is_available:
            raise RuntimeError("GlyphEncoder model not loaded")

        import torch

        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True,
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)

        with torch.no_grad():
            encoder_output = self._model.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
            )

        hidden = encoder_output.last_hidden_state  # (N, seq_len, d_model)
        mask = attention_mask.unsqueeze(-1).float()  # (N, seq_len, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (N, d_model)

        vecs = pooled.cpu().numpy().astype(np.float32)

        # L2 normalize each row
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs /= norms

        return vecs

    # ── PCA Basis ─────────────────────────────────────────────────────

    def ensure_pca_basis(self, min_for_svd: int = 20) -> None:
        """Compute PCA basis from existing glyph embeddings in DB.

        With fewer than *min_for_svd* vectors, uses a random orthonormal
        scaffold so that newly placed tags still get angular diversity.
        """
        if self._pca_basis is not None:
            return

        dim = self._config.glyph_embedding_dimension

        # Try to load existing glyph vectors from DB
        rows = self._db.execute(
            "SELECT glyph_vector, glyph_dimension FROM glyph_positions"
        ).fetchall()

        vecs: list[np.ndarray] = []
        for r in rows:
            rdim = r["glyph_dimension"]
            if rdim == dim:
                vecs.append(np.frombuffer(r["glyph_vector"], dtype=np.float32).copy())

        if len(vecs) >= min_for_svd:
            matrix = np.stack(vecs)
            self._compute_pca_from_matrix(matrix)
        else:
            self._random_orthonormal_scaffold(dim)

    def _bootstrap_pca_from_matrix(self, vecs: np.ndarray) -> None:
        """Compute PCA from a fresh batch of vectors (used during seeding)."""
        if len(vecs) >= 2:
            self._compute_pca_from_matrix(vecs)
        else:
            self._random_orthonormal_scaffold(self._config.glyph_embedding_dimension)

    def _compute_pca_from_matrix(self, vecs: np.ndarray) -> None:
        """SVD-based PCA: top 2 components of the centered matrix."""
        centered = vecs - vecs.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            self._pca_basis = Vt[:2].astype(np.float32)  # (2, d_model)
        except np.linalg.LinAlgError:
            self._random_orthonormal_scaffold(vecs.shape[1])

    def _random_orthonormal_scaffold(self, dim: int) -> None:
        """Gram-Schmidt random orthonormal basis for early bootstrap."""
        rng = np.random.default_rng()
        raw = rng.standard_normal((2, dim)).astype(np.float32)
        u0 = raw[0] / np.linalg.norm(raw[0])
        u1 = raw[1] - np.dot(raw[1], u0) * u0
        u1 = u1 / np.linalg.norm(u1)
        self._pca_basis = np.vstack([u0, u1])

    def invalidate_pca_cache(self) -> None:
        """Clear the PCA basis so it's recomputed on next use."""
        self._pca_basis = None

    # ── Angle Computation ─────────────────────────────────────────────

    def compute_angle(self, vec: np.ndarray) -> Optional[float]:
        """Project a pre-computed embedding to an angle in [0, 2*pi).

        Requires PCA basis to be initialized (call ensure_pca_basis first).
        """
        if self._pca_basis is None:
            return None

        projected = self._pca_basis @ vec  # (2,)
        angle = math.atan2(float(projected[1]), float(projected[0]))
        if angle < 0:
            angle += 2 * math.pi
        return angle
