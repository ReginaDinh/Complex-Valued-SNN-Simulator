"""End-to-end: the iso-phase control must give the known answer (phase-aware >> phase-blind)."""

import csv
from pathlib import Path

from experiments.run import main

CFG = Path(__file__).resolve().parents[1] / "configs" / "iso_phase.yaml"


def test_iso_phase_known_answer(tmp_path):
    out = main([str(CFG), "--seeds", "2", "--steps", "150", "--out", str(tmp_path)])
    with open(out / "summary.csv") as f:
        acc = {r["model"]: float(r["acc_mean"]) for r in csv.DictReader(f)}
    assert acc["cvsnn"] >= 0.9
    assert 0.35 <= acc["lif_magnitude"] <= 0.65
    for f in ("config.yaml", "meta.json", "runs.jsonl", "summary.md"):
        assert (out / f).exists()


def test_hardware_config_runs_and_reports_platforms(tmp_path):
    cfg = Path(__file__).resolve().parents[1] / "configs" / "iso_phase_hw.yaml"
    out = main([str(cfg), "--seeds", "1", "--steps", "40", "--out", str(tmp_path)])
    with open(out / "summary.csv") as f:
        rows = {r["model"]: r for r in csv.DictReader(f)}
    for p in ("digital_event", "digital_dense", "analog_crossbar"):
        assert float(rows["cvsnn"][f"energy_pj_{p}"]) > 0
    assert "acc_prog_noise=0.2" in rows["cvsnn"]
