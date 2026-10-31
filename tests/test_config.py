"""Thresholds live in YAML, never in code. These tests pin the loading rules."""

from pathlib import Path

import pytest

from irap_screening.config import load_config


def test_defaults_load_from_packaged_yaml():
    cfg = load_config()
    assert cfg.segment.length_m == 100
    assert cfg.curvature.threshold_m == 800
    assert cfg.grade.threshold_pct == 7.5
    assert cfg.lane.clrernet_conf == 0.41


def test_cli_override_wins_and_is_recorded_in_provenance():
    cfg = load_config(overrides=["curvature.threshold_m=350"])
    assert cfg.curvature.threshold_m == 350
    assert cfg.provenance["curvature.threshold_m"] == "cli"
    assert cfg.provenance["grade.threshold_pct"] == "default"


def test_profile_overrides_default_but_loses_to_cli(tmp_path: Path):
    profile = tmp_path / "rural.yaml"
    profile.write_text("curvature:\n  threshold_m: 500\ngrade:\n  threshold_pct: 6.0\n")
    cfg = load_config(profile=profile, overrides=["curvature.threshold_m=350"])
    assert cfg.curvature.threshold_m == 350
    assert cfg.grade.threshold_pct == 6.0
    assert cfg.provenance["curvature.threshold_m"] == "cli"
    assert cfg.provenance["grade.threshold_pct"] == "profile"


def test_invalid_threshold_fails_at_load_not_at_report_time():
    with pytest.raises(ValueError):
        load_config(overrides=["curvature.threshold_m=-5"])


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError):
        load_config(overrides=["curvature.nonexistent=1"])
