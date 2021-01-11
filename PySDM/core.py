"""
Created at 09.11.2019
"""

import numpy as np

from PySDM.state.particles import Particles


class Core:

    def __init__(self, n_sd, backend):
        self.__n_sd = n_sd

        self.backend = backend
        self.environment = None
        self.particles: (Particles, None) = None
        self.dynamics = {}
        self.products = {}
        self.observers = []

        self.n_steps = 0

        self.sorting_scheme = 'default'
        self.condensation_solver = None

    @property
    def env(self):
        return self.environment

    @property
    def bck(self):
        return self.backend

    @property
    def Storage(self):
        return self.backend.Storage

    @property
    def Index(self):
        return self.backend.Index

    @property
    def IndexedStorage(self):
        return self.backend.IndexedStorage

    @property
    def PairIndicator(self):
        return self.backend.PairIndicator

    @property
    def PairwiseStorage(self):
        return self.backend.PairwiseStorage

    @property
    def Random(self):
        return self.backend.Random

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

    def normalize(self, prob, norm_factor, subs):
        self.backend.normalize(
            prob, self.particles['cell id'], self.particles.cell_idx,
            self.particles.cell_start, norm_factor, self.dt, self.mesh.dv, subs)

    def condensation(self, kappa, rtol_x, rtol_thd, substeps, ripening_flags):
        particle_temperatures = \
            self.particles["temperature"] if self.particles.has_attribute("temperature") else \
            self.Storage.empty(0, dtype=float)

        self.backend.condensation(
                solver=self.condensation_solver,
                n_cell=self.mesh.n_cell,
                cell_start_arg=self.particles.cell_start,
                v=self.particles["volume"],
                particle_temperatures=particle_temperatures,
                n=self.particles['n'],
                vdry=self.particles["dry volume"],
                idx=self.particles._Particles__idx,
                rhod=self.env["rhod"],
                thd=self.env["thd"],
                qv=self.env["qv"],
                dv=self.env.dv,
                prhod=self.env.get_predicted("rhod"),
                pthd=self.env.get_predicted("thd"),
                pqv=self.env.get_predicted("qv"),
                kappa=kappa,
                rtol_x=rtol_x,
                rtol_thd=rtol_thd,
                r_cr=self.particles["critical radius"],
                dt=self.dt,
                substeps=substeps,
                cell_order=np.argsort(substeps),  # TODO #341 check if better than regular order
                ripening_flags=ripening_flags
            )

    def run(self, steps):
        for _ in range(steps):
            for dynamic in self.dynamics.values():
                dynamic()
            self.n_steps += 1
            for observer in self.observers:
                observer.notify()
