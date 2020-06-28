"""
Created at 03.06.2020
"""


from PySDM.particles_builder import ParticlesBuilder


class Environment:
    def __init__(self, particles_builder: ParticlesBuilder, do_advection: bool = True):
        self.particles = particles_builder.particles
        self.environment = self.particles.environment
        self.do_advection = do_advection

    def __call__(self):
        if self.do_advection:
            self.environment.sync()

