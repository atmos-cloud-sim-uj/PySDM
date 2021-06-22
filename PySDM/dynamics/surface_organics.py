"""
Class for tracking and using organic and inorganic components of dry aerosol
"""


class SurfaceOrganics:

    def __init__(self):
        self.core = None

    def register(self, builder):
        self.core = builder.core

    def __call__(self):
        self.core.env.sync()
