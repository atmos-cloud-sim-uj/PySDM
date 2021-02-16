from ._moist import _Moist
from ..initialisation.r_wet_init import r_wet_init
from ..initialisation.multiplicities import discretise_n
from ..physics import formulae as phys
from ..state import arakawa_c
import numpy as np


class Kinematic1D(_Moist):
    def __init__(self, dt, mesh, thd_of_z, rhod_of_z):
        super().__init__(dt, mesh, [])
        self.thd0 = thd_of_z(mesh.dz * arakawa_c.z_scalar_coord(mesh.grid))
        self.rhod = rhod_of_z(mesh.dz * arakawa_c.z_scalar_coord(mesh.grid))

    def register(self, builder):
        super().register(builder)
        rhod = builder.core.Storage.from_ndarray(self.rhod)
        self._values["current"]["rhod"] = rhod
        self._tmp["rhod"] = rhod

    def get_qv(self) -> np.ndarray:
        return self.core.dynamics['EulerianAdvection'].solvers.advectee

    def get_thd(self) -> np.ndarray:
        return self.thd0

    def init_attributes(self, *,
                        spatial_discretisation,
                        spectral_discretisation,
                        kappa
                        ):
        super().sync()
        self.notify()

        attributes = {}
        with np.errstate(all='raise'):
            positions = spatial_discretisation.sample(self.mesh.grid, self.core.n_sd)
            attributes['cell id'], attributes['cell origin'], attributes['position in cell'] = \
                self.mesh.cellular_attributes(positions)

            r_dry, n_per_kg = spectral_discretisation.sample(self.core.n_sd)
            r_wet = r_wet_init(r_dry, self, attributes['cell id'], kappa)

            rhod = self['rhod'].to_ndarray()
            cell_id = attributes['cell id']
            domain_volume = np.prod(np.array(self.mesh.size))

        attributes['n'] = discretise_n(n_per_kg * rhod[cell_id] * domain_volume)
        attributes['volume'] = phys.volume(radius=r_wet)
        attributes['dry volume'] = phys.volume(radius=r_dry)

        return attributes

    def sync(self):
        super().sync()

    # TODO #418: common with 2D
    @property
    def dv(self):
        return self.mesh.dv
