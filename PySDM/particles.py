"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from PySDM.state.state import State
from PySDM.stats import Stats
from PySDM.physics.planet_data_factory import PlanetDataFactory

class Particles:

    def __init__(self, n_sd, backend, planet_name="Earth", stats=None):
        self.__n_sd = n_sd

        self.backend = backend
        self.environment = None
        self.state: (State, None) = None
        self.dynamics = {}
        self.products = {}
        self.observers = []

        self.n_steps = 0
        self.stats = stats or Stats()

        self.croupier = 'local'
        self.sorting_scheme = 'default'
        self.condensation_solver = None

        self.constants = PlanetDataFactory.create(planet_name)

    @property
    def n_sd(self) -> int:
        return self.__n_sd

    @property
    def dt(self) -> float:
        if self.environment is not None:
            return self.environment.dt

    @property
    def mesh(self):
        if self.environment is not None:
            return self.environment.mesh

    def permute(self, u01):
        if self.croupier == 'global':
            self.state.permutation_global(u01)
        elif self.croupier == 'local':
            self.state.permutation_local(u01)
        else:
            raise NotImplementedError()

    def normalize(self, prob, norm_factor, subs):
        self.backend.normalize(prob, self.state['cell id'], self.state.cell_start, norm_factor, self.dt/subs/self.mesh.dv)

    def find_pairs(self, cell_start, is_first_in_pair):
        self.state.find_pairs(cell_start, is_first_in_pair)

    def sum_pair(self, output, x, is_first_in_pair):
        self.state.sum_pair(output, x, is_first_in_pair)

    def max_pair(self, prob, is_first_in_pair):
        self.state.max_pair(prob, is_first_in_pair)

    def coalescence(self, gamma, adaptive, subs, adaptive_memory):
        return self.state.coalescence(gamma, adaptive, subs, adaptive_memory)

    def remove_precipitated(self):
        self.state.remove_precipitated()

    def condensation(self, kappa, rtol_x, rtol_thd, substeps, ripening_flags):
        particle_temperatures = \
            self.state["temperature"] if self.state.has_attribute("temperature") else \
            self.backend.array(0, dtype=float)

        self.backend.condensation(
                solver=self.condensation_solver,
                n_cell=self.mesh.n_cell,
                cell_start_arg=self.state.cell_start,
                v=self.state["volume"],
                particle_temperatures=particle_temperatures,
                n=self.state['n'],
                vdry=self.state["dry volume"],
                idx=self.state._State__idx,
                rhod=self.environment["rhod"],
                thd=self.environment["thd"],
                qv=self.environment["qv"],
                dv=self.environment.dv,
                prhod=self.environment.get_predicted("rhod"),
                pthd=self.environment.get_predicted("thd"),
                pqv=self.environment.get_predicted("qv"),
                kappa=kappa,
                rtol_x=rtol_x,
                rtol_thd=rtol_thd,
                r_cr=self.state["critical radius"],
                dt=self.dt,
                substeps=substeps,
                cell_order=np.argsort(substeps),  # TODO: check if better than regular order
                ripening_flags=ripening_flags
            )

    def run(self, steps):
        with self.stats:
            for _ in range(steps):
                for dynamic in self.dynamics.values():
                    dynamic()
                self.n_steps += 1
                for observer in self.observers:
                    observer.notify()
