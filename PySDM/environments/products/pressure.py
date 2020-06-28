"""
Created at 03.06.2020
"""

from ...product import Product
from PySDM.environments._moist import _Moist


class Pressure(Product):
    def __init__(self, particles_builder):
        particles = particles_builder.particles
        self.environment = particles.environment
        super().__init__(particles=particles,
                         description="Pressure",
                         name="p",
                         unit="Pa",
                         range=(0, 0),
                         scale="linear",
                         shape=particles.mesh.grid)

    def get(self):
        self.download_to_buffer(self.environment['p'])
        return self.buffer
