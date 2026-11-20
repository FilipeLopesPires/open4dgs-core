from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

DELIVER = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, DELIVER / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prep = _load("prepare_export_checkpoint", "prepare_export_checkpoint.py")
packager = _load("make_trigs_timeline_for_prep_tests", "make-trigs-timeline.py")

N = 5
# sigmoid: 0.5, ~0.0067, ~0.0025, 0.5, 0.5 -> rows 2 (0.0025) is below 0.005
OPACITY_LOGITS = [0.0, -5.0, -6.0, 0.0, 0.0]


def _params() -> dict[str, torch.Tensor]:
    return {
        "means": torch.arange(N * 3, dtype=torch.float32).reshape(N, 3),
        "scales": torch.zeros(N, 3),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(N, 1),
        "opacities": torch.tensor(OPACITY_LOGITS),
        "sh0": torch.zeros(N, 1, 3),
        "shN": torch.zeros(N, 0, 3),
        "times": torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]]),
        "log_durations": torch.log(torch.full((N, 1), 0.1)),
        "anchors": torch.zeros(N, 3),
        "zeta_base_r": torch.zeros(N, 3),
        "zeta_base_t": torch.zeros(N, 3),
        "zeta_c0_r": torch.zeros(N, 3),
        "zeta_c0_t": torch.zeros(N, 3),
        "zeta_c1_r": torch.zeros(N, 3),
        "zeta_c1_t": torch.zeros(N, 3),
        "zeta_c2_r": torch.zeros(N, 3),
        "zeta_c2_t": torch.zeros(N, 3),
        "is_static": torch.zeros(N, 1),
        "dynamic_protect": torch.zeros(N, 1),
    }


def _run(tmp_path: Path, *, snapshot: bool = True, threshold: float = 0.02) -> Path:
    """A run folder as trigs-train-colmap leaves it: no config in trigs_ckpt.pt."""
    run = tmp_path / "runs" / "demo"
    run.mkdir(parents=True)
    params = _params()
    torch.save(
        {
            "params": params,
            "temporal_params": {"times": params["times"].clone(), "log_durations": params["log_durations"].clone()},
            "motion_params": {"anchors": params["anchors"].clone()},
            "metrics": {"final_count": N},
        },
        run / "trigs_ckpt.pt",
    )
    (run / "run_args.json").write_text(json.dumps({"start_frame": 1, "end_frame": 12}))
    if snapshot:
        snaps = run / "viewer_snapshots"
        snaps.mkdir()
        torch.save(
            {
                "format": "trigs_viewer_snapshot",
                "fps": 10.0,
                "n_frames": 10,
                "params": {"means": params["means"].clone()},
                "config": {"fps": 10.0, "temporal_active_threshold": threshold,
                           "temporal_active_cull_threshold": 0.0, "lr_means": 1e-4},
            },
            snaps / "snap_s0000100.pt",
        )
        (snaps / "latest.json").write_text(
            json.dumps({"file": "snap_s0000100.pt", "step": 100, "extra": {"n_rows": N}})
        )
    return run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_copy_carries_snapshot_config_and_drops_faint_rows(tmp_path):
    run = _run(tmp_path)
    src = run / "trigs_ckpt.pt"
    before = _sha(src)
    out = run / "export_ckpt.pt"

    assert prep.main(["--checkpoint", str(src), "--output", str(out), "--n-frames", "12", "--fps", "10"]) == 0

    assert _sha(src) == before  # the source checkpoint is never modified
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["config"]["temporal_active_threshold"] == 0.02
    # the run's own frame count wins over the runner's --n-frames
    assert ckpt["n_frames"] == 10 and ckpt["fps"] == 10.0
    assert ckpt["params"]["means"].shape[0] == 4
    assert ckpt["params"]["means"][:, 0].tolist() == [0.0, 3.0, 9.0, 12.0]
    assert ckpt["temporal_params"]["times"].reshape(-1).tolist() == pytest.approx([0.1, 0.3, 0.7, 0.9])
    assert ckpt["motion_params"]["anchors"].shape[0] == 4

    record = json.loads(Path(str(out) + ".json").read_text())
    assert record["rows_removed"] == 1
    assert record["config_source"] == "viewer_snapshot:snap_s0000100.pt"
    assert record["deform_config"] == {"fps": 10.0, "temporal_active_threshold": 0.02,
                                       "temporal_active_cull_threshold": 0.0}
    assert any("--n-frames 12 ignored" in w for w in record["warnings"])


def test_zero_min_opacity_keeps_every_row(tmp_path):
    run = _run(tmp_path)
    out = run / "export_ckpt.pt"
    prep.main(["--checkpoint", str(run / "trigs_ckpt.pt"), "--output", str(out), "--min-opacity", "0"])
    assert torch.load(out, map_location="cpu", weights_only=False)["params"]["means"].shape[0] == N


def test_without_snapshot_frames_come_from_run_args_and_config_stays_unknown(tmp_path):
    run = _run(tmp_path, snapshot=False)
    out = run / "export_ckpt.pt"
    record = prep.prepare(run / "trigs_ckpt.pt", out, min_opacity=0.005, fps=10.0)
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert "config" not in ckpt
    assert ckpt["n_frames"] == 12 and record["n_frames_source"] == "run_args.json"
    assert record["fps_source"] == "command_line"
    assert any("no training config found" in w for w in record["warnings"])


def test_refuses_to_overwrite_the_source(tmp_path):
    run = _run(tmp_path)
    src = run / "trigs_ckpt.pt"
    with pytest.raises(SystemExit, match="must differ"):
        prep.main(["--checkpoint", str(src), "--output", str(src), "--overwrite"])


def test_packager_uses_the_trained_threshold_from_the_export_copy(tmp_path):
    run = _run(tmp_path, threshold=0.02)
    out = run / "export_ckpt.pt"
    prep.main(["--checkpoint", str(run / "trigs_ckpt.pt"), "--output", str(out)])
    trigsbin = run / "demo.trigsbin"

    assert packager.main([
        "from-checkpoint", "--checkpoint", str(out), "--output", str(trigsbin),
        "--frame-rate", "10", "--max-chunk-duration-seconds", "10", "--tensor-dtype", "mixed",
    ]) == 0

    _, manifest = packager.read_trigsbin_manifest(trigsbin)
    assert manifest["semantics"]["temporalCullThreshold"] == 0.02
    assert manifest["packaging"]["autoSplit"]["opacityThresholdSource"] == "checkpoint_config"
    assert manifest["packaging"]["autoSplit"]["planSource"] == "balanced_max_duration"
    assert len(manifest["clips"]) == 1 and manifest["clips"][0]["dynamic"]["count"] == 4
    tensors = manifest["clips"][0]["dynamic"]["tensors"]
    assert tensors["times"]["dtype"] == "float32" and tensors["means"]["dtype"] == "float16"


def test_threshold_resolution_order_and_trainer_default():
    assert packager.resolve_temporal_threshold(0.2, {"trainerCullThreshold": 0.05}) == (0.2, "command_line")
    assert packager.resolve_temporal_threshold(
        None, {"trainerCullThreshold": 0.05, "configCullThreshold": 0.01}
    ) == (0.05, "trainer_metrics")
    assert packager.resolve_temporal_threshold(None, {"configCullThreshold": 0.03}) == (0.03, "checkpoint_config")
    assert packager.resolve_temporal_threshold(None, {}) == (0.01, "trainer_default")


def test_config_cull_threshold_mirrors_the_trainer():
    f = packager.config_temporal_cull_threshold
    assert f({"temporal_active_threshold": 0.01}) == 0.01
    assert f({"temporal_active_threshold": 0.05, "temporal_active_cull_threshold": 0.01}) == 0.01
    # an override above the base threshold is ignored, as in the trainer
    assert f({"temporal_active_threshold": 0.01, "temporal_active_cull_threshold": 0.05}) == 0.01
    assert f({"lr_means": 1e-4}) is None and f(None) is None


def test_balanced_plan_keeps_short_sequences_whole_and_splits_long_ones_evenly():
    windows, _, schedules, total, source = packager.build_single_checkpoint_split_plan(
        meta={"n_frames": 90}, frame_rate=30.0, total_frames=None,
        chunk_duration_s=None, chunk_overlap_s=0.0, max_chunk_duration_s=10.0,
    )
    assert source == "balanced_max_duration" and total == 90
    assert len(windows) == 1 and (schedules[0].core_start, schedules[0].core_end) == (0, 90)

    windows, _, schedules, total, _ = packager.build_single_checkpoint_split_plan(
        meta={"n_frames": 377}, frame_rate=30.0, total_frames=None,
        chunk_duration_s=None, chunk_overlap_s=0.0, max_chunk_duration_s=10.0,
    )
    assert len(windows) == 2
    assert windows[0][1] == pytest.approx(376 / 30 / 2)
    assert [(s.core_start, s.core_end) for s in schedules] == [(0, 188), (188, 377)]

    windows, *_ = packager.build_single_checkpoint_split_plan(
        meta={"n_frames": 901}, frame_rate=30.0, total_frames=None,
        chunk_duration_s=None, chunk_overlap_s=0.0, max_chunk_duration_s=10.0,
    )
    assert len(windows) == 3 and all(hi - lo == pytest.approx(10.0) for lo, hi in windows)


def test_balanced_plan_is_exclusive_with_fixed_chunks():
    with pytest.raises(ValueError, match="not both"):
        packager.build_single_checkpoint_split_plan(
            meta={"n_frames": 90}, frame_rate=30.0, total_frames=None,
            chunk_duration_s=2.5, chunk_overlap_s=0.0, max_chunk_duration_s=10.0,
        )
