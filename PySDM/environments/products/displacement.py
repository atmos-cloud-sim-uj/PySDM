"""
Created at 03.06.2020
"""

from ...product import Product
from PySDM.environments._moist import _Moist


class Displacement(Product):
    def __init__(self, particles_builder):
        particles = particles_builder.particles
        self.environment = particles.environment
        super().__init__(particles=particles,
                         description="Displacement",
                         name="z",
                         unit="m",
                         range=(0, 0),
                         scale="linear",
                         shape=particles.mesh.grid)

    def get(self):
        self.download_to_buffer(self.environment['z'])
        return self.buffer
