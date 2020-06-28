import pint
from scipy import constants as sci

from PySDM.physics.planet_data import PlanetData
from PySDM.physics._fake_unit_registry import FakeUnitRegistry
from PySDM.physics._flag import DIMENSIONAL_ANALYSIS

si = pint.UnitRegistry()
if not DIMENSIONAL_ANALYSIS:
    si = FakeUnitRegistry(si)


class PlanetDataFactory:
    @staticmethod
    def create(planet_name="Earth"):
        if planet_name == "Earth":
            r_str = sci.R * si.joule / si.kelvin / si.mole
            return PlanetData(sci.g * si.metre / si.second ** 2, r_str, r_str)

        elif planet_name == "Titan":
            g0 = 1.17 * si.centimeter / si.second ** 2
            rg = 2.8 * 10 ** 8 * si.centimeter
            r_str = sci.R * si.joule / si.kelvin / si.mole
            u = 16 * si.gram / si.mol
            rd = r_str / u
            return PlanetData(g0, rg, rd)

