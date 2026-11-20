from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

gsply = pytest.importorskip("gsply")

SCRIPT = Path(__file__).resolve().parents[1] / "compress_plys.py"
SPEC = importlib.util.spec_from_file_location("compress_plys", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
compress_plys = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compress_plys
SPEC.loader.exec_module(compress_plys)


def _write_frame(path: Path, n: int = 600, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    gsply.GSData.from_arrays(
        means=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        scales=rng.uniform(-6, -3, (n, 3)).astype(np.float32),
        quats=quats,
        opacities=rng.uniform(-3, 3, n).astype(np.float32),
        sh0=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        shN=rng.uniform(-0.2, 0.2, (n, 15, 3)).astype(np.float32),
        format="ply",
    ).save(path, compressed=False)


def test_folder_is_compressed_next_to_the_sources(tmp_path):
    for i in range(2):
        _write_frame(tmp_path / f"frame_{i:06d}.ply", seed=i)
    (tmp_path / "export_meta.json").write_text("{}")

    assert compress_plys.main(["--input-dir", str(tmp_path)]) == 0

    for i in range(2):
        src = tmp_path / f"frame_{i:06d}.ply"
        dst = tmp_path / f"frame_{i:06d}.compressed.ply"
        assert src.exists() and dst.exists()
        assert dst.stat().st_size < src.stat().st_size / 2
        a, b = gsply.plyread(src), gsply.plyread(dst)
        assert b.means.shape == a.means.shape
        np.testing.assert_allclose(b.means, a.means, atol=2e-3)  # 11/10/11-bit positions per chunk


def test_delete_source_and_rerun_skips_compressed_files(tmp_path):
    _write_frame(tmp_path / "frame_000000.ply")
    compress_plys.main(["--input-dir", str(tmp_path), "--delete-source"])
    assert [p.name for p in tmp_path.iterdir()] == ["frame_000000.compressed.ply"]
    with pytest.raises(SystemExit, match="no uncompressed"):
        compress_plys.main(["--input-dir", str(tmp_path)])
