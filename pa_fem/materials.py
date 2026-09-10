"""Material fields used by the conservative pressure formulation."""

from __future__ import annotations

from dataclasses import dataclass

from .config import CaseConfig
from .geometry import TAGS


@dataclass(frozen=True)
class Material:
    name: str
    rho_kg_m3: float
    c_m_s: float
    mu_a_m_inv: float
    gamma: float
    eta_th: float

    @property
    def bulk_modulus_pa(self) -> float:
        return self.rho_kg_m3 * self.c_m_s**2

    @property
    def inv_bulk_pa_inv(self) -> float:
        return 1.0 / self.bulk_modulus_pa

    @property
    def inv_rho_m3_kg(self) -> float:
        return 1.0 / self.rho_kg_m3


def material_map(cfg: CaseConfig) -> dict[int, Material]:
    m = cfg.materials
    return {
        TAGS.water: Material("water", m.rho_water_kg_m3, m.c_water_m_s,
                             m.mu_a_water_m_inv, m.gamma_water, m.eta_th_water),
        TAGS.tissue: Material("tissue", m.rho_tissue_kg_m3, m.c_tissue_m_s,
                              m.mu_a_tissue_m_inv, m.gamma_tissue, m.eta_th_tissue),
        TAGS.absorber: Material("absorber", m.rho_absorber_kg_m3, m.c_absorber_m_s,
                                m.mu_a_absorber_m_inv, m.gamma_absorber, m.eta_th_absorber),
    }

