from __future__ import annotations

from pathlib import Path

from pa_fem.config import CaseConfig, load_config


def test_reference_config_has_confinement_and_resolution_contract():
    root = Path(__file__).parents[1]
    cfg = load_config(root / "configs" / "reference.json")
    assert cfg.optical.laser_pulse_width_s < 0.1 * min(cfg.constraint_times())
    assert cfg.mesh.element_size_m <= 50e-6
    assert cfg.time.dt_s <= 10e-9
    assert cfg.source_frequency_hz() > 3.0e6


def test_config_round_trip():
    cfg = CaseConfig()
    clone = CaseConfig.from_dict(cfg.to_dict())
    assert clone == cfg

