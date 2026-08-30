"""The on-disk FAISS index cache.

The cache exists to avoid a 39-57s rebuild on every server start. That is
only worth having if a loaded index is indistinguishable from a freshly
built one, and if it is refused the moment it stops matching the database.
A cache that silently serves stale contents is a much worse bug than the
slow start it replaces, so the staleness tests matter more than the
round-trip ones.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from chicory.layer1.vector_index import VectorIndex

pytest.importorskip("faiss", reason="faiss not installed")

DIM = 32


@pytest.fixture
def tmp_dir():
    """A scratch directory.

    Not pytest's `tmp_path`: this machine has a `pytest-of-wrwar` directory
    under %TEMP% with ACLs that deny access, so `tmp_path` raises
    PermissionError before any test body runs. mkdtemp sidesteps it by
    creating directly under %TEMP%.
    """
    d = Path(tempfile.mkdtemp(prefix="chicory-idxcache-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _corpus(n: int, seed: int = 0) -> tuple[list[str], np.ndarray]:
    rng = np.random.RandomState(seed)
    mat = rng.randn(n, DIM).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    ids = [f"mem-{i // 3:04d}" for i in range(n)]  # ~3 chunks per memory
    return ids, mat


def _build(n: int = 400) -> tuple[VectorIndex, np.ndarray]:
    ids, mat = _corpus(n)
    vi = VectorIndex(dimension=DIM, nprobe=8)
    vi.build_from_matrix(ids, mat)
    return vi, mat


def test_save_load_roundtrip_is_identical(tmp_dir: Path):
    vi, mat = _build()
    path = tmp_dir / "idx.faiss"
    vi.save(path, {"max_embedding_id": 7, "count": 400})

    loaded = VectorIndex(dimension=DIM, nprobe=8)
    meta = loaded.load(path)
    assert meta is not None
    assert meta["max_embedding_id"] == 7 and meta["count"] == 400

    for q in mat[:20]:
        assert vi.search(q, 10) == loaded.search(q, 10)


def test_load_returns_none_when_files_absent(tmp_dir: Path):
    assert VectorIndex(dimension=DIM).load(tmp_dir / "nope.faiss") is None


def test_load_rejects_dimension_mismatch(tmp_dir: Path):
    vi, _ = _build()
    path = tmp_dir / "idx.faiss"
    vi.save(path, {})
    assert VectorIndex(dimension=DIM + 1).load(path) is None


def test_load_rejects_truncated_id_map(tmp_dir: Path):
    """ntotal and the id map must agree, or positions decode to wrong ids."""
    vi, _ = _build()
    path = tmp_dir / "idx.faiss"
    vi.save(path, {})

    nonce = (tmp_dir / "idx.faiss.current").read_text(encoding="utf-8").strip()
    ids_path = tmp_dir / f"idx.faiss.{nonce}" / "ids.npz"
    with np.load(ids_path, allow_pickle=False) as z:
        uniq, inv = z["uniq"], z["inv"]
    with open(ids_path, "wb") as fh:
        np.savez(fh, uniq=uniq, inv=inv[:-5])

    assert VectorIndex(dimension=DIM).load(path) is None


def test_add_is_batched_and_keeps_id_map_aligned():
    vi, _ = _build()
    before = vi.ntotal
    rng = np.random.RandomState(99)
    vecs = [rng.randn(DIM).astype(np.float32) for _ in range(4)]
    vecs = [v / np.linalg.norm(v) for v in vecs]
    vi.add("newmem", vecs)

    assert vi.ntotal == before + 4
    assert len(vi._id_to_memory) == vi.ntotal, "id map drifted from ntotal"
    assert vi._id_to_memory[-1] == "newmem"
    hits = vi.search(vecs[0], 5)
    assert any(mid == "newmem" for mid, _ in hits)


def test_needs_rebuild_is_proportional_not_absolute():
    """A flat threshold retrains a 351k-vector index for a 0.14% change."""
    vi, _ = _build(n=400)
    vi._rebuild_threshold = 100
    vi._adds_since_build = 150          # over the flat threshold...
    # ...but only 150/400 = 37.5% -> above 25%, so a rebuild is warranted
    assert vi.needs_rebuild

    big = VectorIndex(dimension=DIM, nprobe=8, rebuild_threshold=100)
    ids, mat = _corpus(10_000, seed=3)
    big.build_from_matrix(ids, mat)
    big._adds_since_build = 150          # 1.5% of the corpus
    assert not big.needs_rebuild, "flat threshold would retrain for a 1.5% change"


# ── Regressions from the hardening pass ─────────────────────────────
#
# Each of these reproduces a defect that shipped in the first version of
# the cache. All four shared one shape: the cache stayed *accepted* while
# no longer describing the data, so the wrong answer survived restarts
# instead of being corrected by one.


def test_load_rejects_count_not_equal_ntotal(tmp_dir: Path):
    """meta claiming more rows than the index holds means the corpus moved
    mid-build, and the cache is stamped as covering a memory it never
    indexed."""
    vi, _ = _build(400)
    path = tmp_dir / "idx.faiss"
    vi.save(path, {"count": 401, "watermark": [401, 401, 1]})
    assert VectorIndex(dimension=DIM).load(path) is None


def test_interrupted_save_leaves_previous_set_intact(tmp_dir: Path):
    """A save that dies partway must not disturb what is already committed.

    The dangerous state is a NEW index beside an OLD id map: both have the
    same length, so a length check passes and every query returns real
    vectors under the wrong memory ids. Publishing through a single
    pointer swap makes that state unreachable -- an interrupted save
    leaves an orphan directory nothing points at.
    """
    ids_a, mat_a = _corpus(400, seed=1)
    vi_a = VectorIndex(dimension=DIM, nprobe=8)
    vi_a.build_from_matrix(ids_a, mat_a)
    path = tmp_dir / "idx.faiss"
    vi_a.save(path, {"count": 400, "generation": "A"})

    # A second save of a different corpus that dies after the index write.
    ids_b, mat_b = _corpus(400, seed=2)
    vi_b = VectorIndex(dimension=DIM, nprobe=8)
    vi_b.build_from_matrix(ids_b, mat_b)

    import numpy as _np
    real_savez = _np.savez
    _np.savez = lambda *a, **k: (_ for _ in ()).throw(OSError("crash"))
    try:
        with pytest.raises(Exception):
            vi_b.save(path, {"count": 400, "generation": "B"})
    finally:
        _np.savez = real_savez

    loaded = VectorIndex(dimension=DIM, nprobe=8)
    meta = loaded.load(path)
    assert meta is not None, "interrupted save destroyed a good cache"
    assert meta["generation"] == "A", "load served the half-written set"

    # And the id map must still be generation A's, not a mix.
    hits = loaded.search(mat_a[0], 3)
    assert hits and all(m.startswith("mem-") for m, _ in hits)
    assert vi_a.search(mat_a[0], 3) == hits


def test_successful_save_prunes_superseded_sets(tmp_dir: Path):
    """Each save writes a new directory; old ones must not accumulate."""
    path = tmp_dir / "idx.faiss"
    for gen in range(3):
        ids, mat = _corpus(400, seed=gen)
        vi = VectorIndex(dimension=DIM, nprobe=8)
        vi.build_from_matrix(ids, mat)
        vi.save(path, {"count": 400, "generation": gen})

    set_dirs = [d for d in tmp_dir.iterdir() if d.is_dir()]
    assert len(set_dirs) == 1, f"superseded sets left on disk: {set_dirs}"
    assert VectorIndex(dimension=DIM).load(path)["generation"] == 2


def test_orphan_set_directory_is_ignored(tmp_dir: Path):
    """A directory nothing points at must never be loaded."""
    vi, _ = _build(400)
    path = tmp_dir / "idx.faiss"
    vi.save(path, {"count": 400, "generation": "good"})
    (tmp_dir / "idx.faiss.deadbeefdeadbeef").mkdir()

    meta = VectorIndex(dimension=DIM).load(path)
    assert meta is not None and meta["generation"] == "good"


def test_discard_cache_removes_everything(tmp_dir: Path):
    vi, _ = _build(400)
    path = tmp_dir / "idx.faiss"
    vi.save(path, {"count": 400})
    vi.discard_cache(path)
    assert VectorIndex(dimension=DIM).load(path) is None
    assert not [d for d in tmp_dir.iterdir()], "files left after discard"
