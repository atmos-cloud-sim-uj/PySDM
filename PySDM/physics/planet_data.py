class PlanetData():
    def __init__(self, g0, Rd, R_str):
        self.g0 = g0
        self.Rd = Rd
        self.R_str = R_str

    def g(self, z):
        acc = self.g0 * (self.Rd / (self.Rd + z)) ** 2
        return acc