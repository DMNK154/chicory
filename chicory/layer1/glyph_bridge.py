"""GPT-GU glyph bridge for Chicory's glyph Ramsey lattice.

Translates Chicory tag names to GPT-GU glyphs via the lexicon or ByT5
generation, then embeds the *glyph symbols* (not the English words)
through ByT5's encoder.  This produces lattice embeddings in GPT-GU's
symbolic space, where structural glyph relationships surface as Ramsey
resonances.

Implements the same duck-type interface as GlyphEncoder so it can be
used as a drop-in replacement in SynchronicityEngine.

Requires GPT-GU to be installed at the path specified by
``config.gptgu_path``.  When GPT-GU is unavailable, ``is_available``
returns False and the orchestrator falls back to GlyphEncoder or
letter-count.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
from typing import Optional

import numpy as np

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.layer1.glyph_lexicon import (
    DICT2GLYPH,
    GLYPH2DICT,
    PAIR_TO_TEXT,
    concept_for_glyph,
    glyph_for_concept,
)

logger = logging.getLogger(__name__)


class GlyphBridge:
    """Bridge GPT-GU's glyph system into Chicory's glyph Ramsey lattice.

    The hardcoded lexicon (glyph_lexicon.py) is always available for
    symbol↔concept lookups.  GPT-GU is only needed for ByT5 model-based
    generation of novel glyphs not in the lexicon.
    """

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db
        self._available = False
        self._lock = threading.Lock()
        self._pca_basis: Optional[np.ndarray] = None  # (2, d_model)

        # Hardcoded lexicon — always available
        self._lex_dict2glyph: dict[str, str] = DICT2GLYPH
        self._pair_to_text: dict[tuple[str, str], tuple[str, str]] = PAIR_TO_TEXT

        # GPT-GU references (set by _load, needed only for ByT5 generation)
        self._tok = None
        self._mdl = None
        self._device: str = "cpu"
        self._dict2glyph_fn = None  # GPT-GU's model-based dict2glyph

        self._load()

    # ── Loading ────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load ByT5 model from GPT-GU for novel glyph generation.

        The lexicon is always available from glyph_lexicon.py.
        GPT-GU is only needed for model-based generation of glyphs
        not in the hardcoded lexicon.  If GPT-GU is unavailable,
        the bridge still works for all known glyphs.
        """
        gptgu_path = self._config.gptgu_path
        if not gptgu_path:
            # No GPT-GU path — lexicon-only mode (still functional)
            self._available = True
            logger.info(
                "GlyphBridge loaded in lexicon-only mode (%d entries, no ByT5 generation)",
                len(self._lex_dict2glyph),
            )
            return

        try:
            # Add GPT-GU to sys.path so we can import app.guardrails
            if gptgu_path not in sys.path:
                sys.path.insert(0, gptgu_path)

            from app.guardrails import (
                dict2glyph as _dict2glyph,
                ensure_loaded,
            )

            self._dict2glyph_fn = _dict2glyph

            tok, mdl, device = ensure_loaded()
            self._tok = tok
            self._mdl = mdl
            self._device = device
            self._available = True

            d_model = mdl.config.d_model
            logger.info(
                "GlyphBridge loaded (d_model=%d, device=%s, lexicon=%d entries)",
                d_model, device, len(self._lex_dict2glyph),
            )
        except ImportError:
            # GPT-GU not importable — still available in lexicon-only mode
            self._available = True
            logger.warning(
                "Could not import GPT-GU from %s — lexicon-only mode (%d entries)",
                gptgu_path, len(self._lex_dict2glyph),
            )
        except Exception:
            # Model load failed — still available in lexicon-only mode
            self._available = True
            logger.warning("ByT5 model load failed — lexicon-only mode", exc_info=True)

    @property
    def is_available(self) -> bool:
        """True iff the glyph lexicon is loaded (always true after __init__)."""
        return self._available

    @property
    def has_model(self) -> bool:
        """True iff the ByT5 model is loaded for novel glyph generation."""
        return self._tok is not None and self._mdl is not None

    # ── Translation ────────────────────────────────────────────────────

    def _translate_tag(self, tag_name: str) -> Optional[tuple[str, str, str]]:
        """Translate a tag name to a GPT-GU glyph.

        Returns (glyph_symbol, concept_name, source) or None on failure.
        Source is 'lexicon' or 'byt5_gen'.
        """
        # Check persistent cache first
        cached = self._db.execute(
            "SELECT glyph_symbol, glyph_concept, source FROM glyph_bridge_cache WHERE tag_name = ?",
            (tag_name,),
        ).fetchone()
        if cached:
            return (cached["glyph_symbol"], cached["glyph_concept"] or "", cached["source"])

        # Try lexicon lookup (multiple case strategies)
        lex = self._lex_dict2glyph
        for variant in (tag_name, tag_name.title(), tag_name.capitalize(), tag_name.lower()):
            if variant in lex:
                glyph = lex[variant]
                return (glyph, variant, "lexicon")

        # Fall back to ByT5 generation via dict2glyph (requires model)
        if self._dict2glyph_fn is not None:
            try:
                glyph = self._dict2glyph_fn(tag_name)
                if glyph:
                    return (glyph, tag_name, "byt5_gen")
            except Exception:
                logger.debug("dict2glyph failed for %r", tag_name, exc_info=True)

        return None

    def _cache_translation(
        self, tag_name: str, glyph_symbol: str, concept: str,
        embedding: np.ndarray, source: str,
    ) -> None:
        """Persist a translation + embedding in the bridge cache."""
        self._db.execute(
            """
            INSERT INTO glyph_bridge_cache
                (tag_name, glyph_symbol, glyph_concept, embedding, embedding_dim, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(tag_name) DO UPDATE SET
                glyph_symbol = excluded.glyph_symbol,
                glyph_concept = excluded.glyph_concept,
                embedding = excluded.embedding,
                embedding_dim = excluded.embedding_dim,
                source = excluded.source,
                cached_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
            """,
            (tag_name, glyph_symbol, concept,
             embedding.astype(np.float32).tobytes(),
             len(embedding), source),
        )

    # ── Embedding ──────────────────────────────────────────────────────

    def _encode_glyphs(self, glyph_symbols: list[str]) -> np.ndarray:
        """Encode glyph symbol strings through ByT5 encoder.

        Returns mean-pooled, L2-normalized vectors of shape (N, d_model).
        """
        import torch

        inputs = self._tok(
            glyph_symbols, return_tensors="pt", padding=True,
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)

        with torch.no_grad():
            encoder_output = self._mdl.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
            )

        hidden = encoder_output.last_hidden_state  # (N, seq_len, d_model)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        vecs = pooled.cpu().numpy().astype(np.float32)

        # L2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs /= norms

        return vecs

    def embed(self, text: str) -> np.ndarray:
        """Embed a single tag name via glyph translation + ByT5 encoder.

        Returns a normalized float32 vector of shape (d_model,).
        Requires ByT5 model — raises RuntimeError in lexicon-only mode.
        """
        if not self.has_model:
            raise RuntimeError("ByT5 model not loaded (lexicon-only mode)")

        # Check cache for pre-computed embedding
        cached = self._db.execute(
            "SELECT embedding, embedding_dim FROM glyph_bridge_cache WHERE tag_name = ?",
            (text,),
        ).fetchone()
        if cached:
            return np.frombuffer(cached["embedding"], dtype=np.float32).copy()

        translation = self._translate_tag(text)
        if translation is None:
            raise RuntimeError(f"Could not translate tag {text!r} to glyph")

        glyph_symbol, concept, source = translation
        vecs = self._encode_glyphs([glyph_symbol])
        vec = vecs[0]

        self._cache_translation(text, glyph_symbol, concept, vec, source)
        self._db.connection.commit()

        return vec

    def embed_batch(self, tag_names: list[str]) -> np.ndarray:
        """Embed multiple tag names via glyph translation + ByT5 encoder.

        Returns normalized float32 array of shape (N, d_model).
        Requires ByT5 model — raises RuntimeError in lexicon-only mode.
        """
        if not tag_names:
            return np.empty((0, self._config.glyph_embedding_dimension), dtype=np.float32)

        if not self.has_model:
            raise RuntimeError("ByT5 model not loaded (lexicon-only mode)")

        d_model = self._mdl.config.d_model
        result = np.zeros((len(tag_names), d_model), dtype=np.float32)

        # Phase 1: Check cache for already-computed embeddings
        uncached_indices: list[int] = []
        uncached_names: list[str] = []

        for i, name in enumerate(tag_names):
            cached = self._db.execute(
                "SELECT embedding FROM glyph_bridge_cache WHERE tag_name = ?",
                (name,),
            ).fetchone()
            if cached:
                vec = np.frombuffer(cached["embedding"], dtype=np.float32).copy()
                result[i, :len(vec)] = vec
            else:
                uncached_indices.append(i)
                uncached_names.append(name)

        if not uncached_names:
            return result

        # Phase 2: Translate uncached tags to glyphs
        translations: list[Optional[tuple[str, str, str]]] = []
        for name in uncached_names:
            translations.append(self._translate_tag(name))

        # Phase 3: Batch encode all glyph symbols
        to_encode_indices: list[int] = []  # indices into uncached_names
        glyph_symbols: list[str] = []

        for j, trans in enumerate(translations):
            if trans is not None:
                to_encode_indices.append(j)
                glyph_symbols.append(trans[0])  # glyph_symbol

        if glyph_symbols:
            vecs = self._encode_glyphs(glyph_symbols)

            for k, j in enumerate(to_encode_indices):
                i = uncached_indices[j]
                vec = vecs[k]
                result[i, :len(vec)] = vec

                # Cache the result
                glyph_symbol, concept, source = translations[j]
                self._cache_translation(uncached_names[j], glyph_symbol, concept, vec, source)

            self._db.connection.commit()

        return result

    # ── Glyph Mapping (for symbolic bonus) ─────────────────────────────

    def get_glyph_mapping(self, tag_name: str) -> Optional[str]:
        """Return the glyph symbol for a tag, or None if not cached."""
        cached = self._db.execute(
            "SELECT glyph_symbol FROM glyph_bridge_cache WHERE tag_name = ?",
            (tag_name,),
        ).fetchone()
        if cached:
            return cached["glyph_symbol"]
        return None

    def get_all_glyph_mappings(self) -> dict[str, str]:
        """Return all cached tag_name -> glyph_symbol mappings."""
        rows = self._db.execute(
            "SELECT tag_name, glyph_symbol FROM glyph_bridge_cache"
        ).fetchall()
        return {r["tag_name"]: r["glyph_symbol"] for r in rows}

    def concept_for_glyph(self, glyph: str) -> str | None:
        """Return the concept name for a glyph symbol (always available)."""
        return concept_for_glyph(glyph)

    def glyph_for_concept(self, concept: str) -> str | None:
        """Return the glyph symbol for a concept name (always available)."""
        return glyph_for_concept(concept)

    @property
    def lexicon(self) -> dict[str, str]:
        """The full concept→glyph lexicon (always available)."""
        return self._lex_dict2glyph

    @property
    def inverse_lexicon(self) -> dict[str, str]:
        """The full glyph→concept mapping (always available)."""
        return GLYPH2DICT

    def get_pair_relationships(self) -> dict[frozenset[str], str]:
        """Return GPT-GU's PAIR_TO_TEXT as {frozenset(g_a, g_b): 'concept_a vs concept_b'}."""
        result: dict[frozenset[str], str] = {}
        for (ga, gb), (ca, cb) in self._pair_to_text.items():
            key = frozenset((ga, gb))
            if key not in result:
                result[key] = f"{ca} vs {cb}"
        return result

    def get_directed_pairs(self) -> list[tuple[str, str, str, str]]:
        """Return PAIR_TO_TEXT as directed tuples: (glyph_a, glyph_b, concept_a, concept_b).

        Each entry represents a directional relationship A→B (semiotic ordering).
        Pairs in PAIR_TO_TEXT are stored both ways, so dedup to canonical direction.
        """
        seen: set[frozenset[str]] = set()
        result: list[tuple[str, str, str, str]] = []
        for (ga, gb), (ca, cb) in self._pair_to_text.items():
            key = frozenset((ga, gb))
            if key not in seen:
                seen.add(key)
                result.append((ga, gb, ca, cb))
        return result

    def get_cross_reference_data(self) -> Optional[dict]:
        """Load glyph cross-reference data: oppositions, transformations, co-occurrences.

        Primary source: hardcoded PAIR_TO_TEXT (always available).
        Extended source: GPT-GU's cross_reference module (if importable).

        Returns dict with keys 'oppositions', 'transformations', 'co_occurrences'.
        """
        if not self._available:
            return None

        from collections import Counter, defaultdict

        # Base data from hardcoded pair relationships (always available)
        oppositions: dict[str, set[str]] = defaultdict(set)
        for (ga, gb) in self._pair_to_text:
            oppositions[ga].add(gb)

        transformations: dict[str, set[str]] = defaultdict(set)
        co_occurrences: dict[str, Counter] = defaultdict(Counter)

        # Try to extend with GPT-GU's richer cross-reference data
        try:
            from app.cross_reference import get_cross_reference_engine
            xref = get_cross_reference_engine()

            # Merge GPT-GU oppositions
            for glyph_a, opp_set in xref.oppositions.items():
                oppositions[glyph_a].update(opp_set)

            # GPT-GU transformations
            for glyph_a, target_set in xref.transformations.items():
                transformations[glyph_a].update(target_set)

            # GPT-GU co-occurrences
            for glyph_a, counter in xref.co_occurrences.items():
                co_occurrences[glyph_a].update(counter)

            logger.debug("Extended cross-reference with GPT-GU data")
        except ImportError:
            logger.debug("GPT-GU cross_reference not available — using lexicon pairs only")
        except Exception:
            logger.debug("Failed to load GPT-GU cross-reference", exc_info=True)

        return {
            "oppositions": dict(oppositions),
            "transformations": dict(transformations),
            "co_occurrences": dict(co_occurrences),
        }

    # ── PCA Basis ──────────────────────────────────────────────────────

    def ensure_pca_basis(self, min_for_svd: int = 20) -> None:
        """Compute PCA basis from existing glyph embeddings in DB."""
        if self._pca_basis is not None:
            return

        dim = self._config.glyph_embedding_dimension

        rows = self._db.execute(
            "SELECT glyph_vector, glyph_dimension FROM glyph_positions"
        ).fetchall()

        vecs: list[np.ndarray] = []
        for r in rows:
            if r["glyph_dimension"] == dim:
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
            self._pca_basis = Vt[:2].astype(np.float32)
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

    def compute_angle(self, vec: np.ndarray) -> Optional[float]:
        """Project a pre-computed embedding to an angle in [0, 2*pi)."""
        if self._pca_basis is None:
            return None

        projected = self._pca_basis @ vec
        angle = math.atan2(float(projected[1]), float(projected[0]))
        if angle < 0:
            angle += 2 * math.pi
        return angle

    # ── Cache Management ───────────────────────────────────────────────

    def rebuild_cache(self) -> int:
        """Clear and repopulate cache for all active word tags."""
        if not self.has_model:
            return 0

        self._db.execute("DELETE FROM glyph_bridge_cache")
        self._db.connection.commit()

        rows = self._db.execute(
            "SELECT DISTINCT name FROM tags WHERE is_active = 1 AND LENGTH(name) > 1"
        ).fetchall()
        tag_names = [r["name"] for r in rows]

        if not tag_names:
            return 0

        self.embed_batch(tag_names)
        return len(tag_names)
