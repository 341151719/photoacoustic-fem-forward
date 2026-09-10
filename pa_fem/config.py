"""SI-unit case configuration and physical consistency checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GeometryConfig:
    domain_x_min_m: float = -6.0e-3
    domain_x_max_m: float = 6.0e-3
    domain_y_min_m: float = 0.0
    domain_y_max_m: float = 10.0e-3
    tissue_x_min_m: float = -5.0e-3
    tissue_x_max_m: float = 5.0e-3
    tissue_y_min_m: float = 2.0e-3
    tissue_y_max_m: float = 9.0e-3
    absorber_x_m: float = 0.0
    absorber_y_m: float = 5.5e-3
    absorber_radius_m: float = 0.30e-3
    sensor_x_min_m: float = -1.5e-3
    sensor_x_max_m: float = 1.5e-3
    sensor_y_m: float = 0.0


@dataclass(frozen=True)
class MeshConfig:
    # ``debug`` uses a coarser mesh; ``reference`` is the recommended baseline.
    element_size_m: float = 0.20e-3
    interface_size_m: float = 0.10e-3
    source_size_m: float = 0.08e-3
    algorithm: int = 6
    msh_file_version: float = 4.1


@dataclass(frozen=True)
class MaterialConfig:
    rho_water_kg_m3: float = 1000.0
    c_water_m_s: float = 1480.0
    rho_tissue_kg_m3: float = 1050.0
    c_tissue_m_s: float = 1540.0
    rho_absorber_kg_m3: float = 1050.0
    c_absorber_m_s: float = 1540.0
    mu_a_water_m_inv: float = 0.0
    mu_a_tissue_m_inv: float = 10.0
    mu_a_absorber_m_inv: float = 1000.0
    gamma_water: float = 0.0
    gamma_tissue: float = 0.10
    gamma_absorber: float = 0.10
    eta_th_water: float = 1.0
    eta_th_tissue: float = 1.0
    eta_th_absorber: float = 1.0


@dataclass(frozen=True)
class OpticalConfig:
    phi_peak_j_m2: float = 100.0
    sigma_m: float = 0.20e-3
    source_x_m: float = 0.0
    source_y_m: float = 5.5e-3
    laser_pulse_width_s: float = 10.0e-9
    require_initial_pressure_approximation: bool = True


@dataclass(frozen=True)
class SensorConfig:
    center_frequency_hz: float = 1.0e6
    fractional_bandwidth_fwhm: float = 0.80
    filter_order: int = 4
    adc_rate_hz: float | None = None
    noise_rms_pa: float = 0.0
    noise_seed: int = 20260910


@dataclass(frozen=True)
class TimeConfig:
    t_end_s: float = 8.0e-6
    dt_s: float = 20.0e-9
    save_every: int = 20


@dataclass(frozen=True)
class CaseConfig:
    case_name: str = "baseline_1MHz_debug"
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    materials: MaterialConfig = field(default_factory=MaterialConfig)
    optical: OpticalConfig = field(default_factory=OpticalConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    random_seed: int = 20260910

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseConfig":
        fields = {
            "geometry": GeometryConfig,
            "mesh": MeshConfig,
            "materials": MaterialConfig,
            "optical": OpticalConfig,
            "sensor": SensorConfig,
            "time": TimeConfig,
        }
        values: dict[str, Any] = {"case_name": data.get("case_name", cls.case_name),
                                  "random_seed": data.get("random_seed", cls.random_seed)}
        for name, typ in fields.items():
            raw = data.get(name, {})
            if not isinstance(raw, dict):
                raise TypeError(f"config section {name!r} must be an object")
            defaults = typ()
            kwargs = {k: raw.get(k, getattr(defaults, k)) for k in defaults.__dataclass_fields__}
            values[name] = typ(**kwargs)
        out = cls(**values)
        out.validate()
        return out

    def validate(self) -> None:
        g, m, o, s, t = self.geometry, self.mesh, self.optical, self.sensor, self.time
        if not g.domain_x_min_m < g.domain_x_max_m or not g.domain_y_min_m < g.domain_y_max_m:
            raise ValueError("domain bounds must be strictly increasing")
        if not (g.domain_x_min_m <= g.tissue_x_min_m < g.tissue_x_max_m <= g.domain_x_max_m):
            raise ValueError("tissue x bounds must be inside the water domain")
        if not (g.domain_y_min_m <= g.tissue_y_min_m < g.tissue_y_max_m <= g.domain_y_max_m):
            raise ValueError("tissue y bounds must be inside the water domain")
        if o.sigma_m <= 0 or g.absorber_radius_m <= 0:
            raise ValueError("source sigma and absorber radius must be positive")
        if not (g.tissue_x_min_m + g.absorber_radius_m < g.absorber_x_m < g.tissue_x_max_m - g.absorber_radius_m and
                g.tissue_y_min_m + g.absorber_radius_m < g.absorber_y_m < g.tissue_y_max_m - g.absorber_radius_m):
            raise ValueError("absorber disk must be completely contained in tissue")
        if not (g.domain_x_min_m <= o.source_x_m <= g.domain_x_max_m and
                g.domain_y_min_m <= o.source_y_m <= g.domain_y_max_m):
            raise ValueError("optical source centre must be inside the computational domain")
        if not (g.domain_x_min_m <= g.sensor_x_min_m < g.sensor_x_max_m <= g.domain_x_max_m):
            raise ValueError("sensor x interval must be inside domain")
        if abs(g.sensor_y_m - g.domain_y_min_m) > 1e-12:
            raise ValueError("the reference sensor must lie on the lower outer boundary")
        for name in ("rho_water_kg_m3", "rho_tissue_kg_m3", "rho_absorber_kg_m3",
                     "c_water_m_s", "c_tissue_m_s", "c_absorber_m_s"):
            if getattr(self.materials, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if o.phi_peak_j_m2 < 0:
            raise ValueError("fluence cannot be negative")
        if m.element_size_m <= 0 or m.interface_size_m <= 0 or m.source_size_m <= 0:
            raise ValueError("mesh sizes must be positive")
        if t.dt_s <= 0 or t.t_end_s <= 0 or t.t_end_s < t.dt_s:
            raise ValueError("time step and end time are inconsistent")
        if t.save_every < 1:
            raise ValueError("save_every must be >= 1")
        if s.center_frequency_hz <= 0 or not (0 < s.fractional_bandwidth_fwhm < 2):
            raise ValueError("sensor centre frequency/bandwidth are invalid")
        if s.filter_order < 1:
            raise ValueError("filter order must be >= 1")

        tau_s, tau_th = self.constraint_times()
        if o.require_initial_pressure_approximation and o.laser_pulse_width_s >= 0.1 * min(tau_s, tau_th):
            raise ValueError(
                "laser pulse is not short enough for the initial-pressure approximation: "
                f"tau_L={o.laser_pulse_width_s:.3e} s, limit={0.1*min(tau_s,tau_th):.3e} s"
            )

    def constraint_times(self) -> tuple[float, float]:
        """Return conservative stress and thermal confinement times (seconds)."""
        length = max(self.optical.sigma_m, 1e-15)
        c = max(self.materials.c_tissue_m_s, self.materials.c_water_m_s)
        # Thermal diffusivity for soft tissue, engineering reference value.
        alpha_th = 1.4e-7
        return length / c, length * length / (4.0 * alpha_th)

    def source_frequency_hz(self) -> float:
        c_min = min(self.materials.c_water_m_s, self.materials.c_tissue_m_s,
                    self.materials.c_absorber_m_s)
        return 3.0 * c_min / (2.0 * math.pi * self.optical.sigma_m)

    def recommended_mesh_size_m(self) -> float:
        # Fractional bandwidth is the total FWHM, so the upper edge is
        # fc * (1 + B/2), consistent with the receiver filter.
        f_tr = self.sensor.center_frequency_hz * (1.0 + self.sensor.fractional_bandwidth_fwhm / 2.0)
        return min(self.materials.c_water_m_s, self.materials.c_tissue_m_s,
                   self.materials.c_absorber_m_s) / (10.0 * max(self.source_frequency_hz(), f_tr))


def load_config(path: str | Path) -> CaseConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    cfg = CaseConfig.from_dict(data)
    return cfg
