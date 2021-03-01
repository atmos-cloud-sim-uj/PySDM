from PySDM.physics import si
from PySDM.initialisation import spectral_sampling, spectra
from PySDM.physics import formulae as phys
from PySDM.physics import constants as const
from chempy import Substance
import numpy as np
from PySDM.dynamics.aqueous_chemistry.aqueous_chemistry import AQUEOUS_COMPOUNDS


class Settings:
    DRY_RHO = 1800 * si.kg / (si.m ** 3)

    def __init__(self, dt, n_sd):
        self.system_type = 'closed'
        self.t_max = (2400 + 196) * si.s
        self.w = .5 * si.m/si.s
        self.g = 10 * si.m / si.s**2

        self.dt = dt
        self.n_sd = n_sd

        self.p0 = 950 * si.mbar
        self.T0 = 285.2 * si.K
        self.pv0 = .95 * phys.pvs(self.T0)

        self.kappa = .61  # TODO
        rho = 1  # TODO
        self.mass_of_dry_air = 44  # TODO

        self.r_dry, self.n_in_dv = spectral_sampling.ConstantMultiplicity(
            spectrum=spectra.Lognormal(
                norm_factor=566 / si.cm**3 / rho * self.mass_of_dry_air,
                m_mode=.08 * si.um / 2,
                s_geom=2
            )
        ).sample(n_sd)

        self.ENVIRONMENT_MOLE_FRACTIONS = {
            "SO2": 0.2 * const.ppb,
            "O3": 50 * const.ppb,
            "H2O2": 0.5 * const.ppb,
            "CO2": 360 * const.ppm,
            "HNO3": 0.1 * const.ppb,
            "NH3": 0.1 * const.ppb,
        }

        DRY_SUBSTANCE = Substance.from_formula("NH4HSO4")
        self.starting_amounts = {
            "moles_"+k: phys.volume(self.r_dry) * Settings.DRY_RHO / (DRY_SUBSTANCE.mass * si.gram / si.mole)
            if k in ("N_mIII", "S_VI", "H") else np.zeros(self.n_sd) for k in AQUEOUS_COMPOUNDS.keys()
        }

    @property
    def nt(self):
        nt = self.t_max / self.dt
        assert nt == int(nt)
        return int(nt)