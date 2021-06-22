"""
Collection of physical constants with dimensional analysis handled with
[Pint](https://pypi.org/project/Pint/)'s package `UnitRegistry` for test
purposes and mocked with `PySDM.physics.impl.fake_unit_registry.FakeUnitRegistry` by default.
"""

import pint
from scipy import constants as sci
from PySDM.physics.impl.fake_unit_registry import FakeUnitRegistry
from PySDM.physics.impl.flag import DIMENSIONAL_ANALYSIS
from chempy import Substance

si = pint.UnitRegistry()
if not DIMENSIONAL_ANALYSIS:
    si = FakeUnitRegistry(si)


def convert_to(value, unit):
    value /= unit


pi = sci.pi
pi_4_3 = pi * 4 / 3
three = 3
one_third = 1/3
two_thirds = 2/3

Md = (
        0.78 * Substance.from_formula('N2').mass * si.gram / si.mole +
        0.21 * Substance.from_formula('O2').mass * si.gram / si.mole +
        0.01 * Substance.from_formula('Ar').mass * si.gram / si.mole
)
Mv = Substance.from_formula('H2O').mass * si.gram / si.mole

R_str = sci.R * si.joule / si.kelvin / si.mole
eps = Mv / Md
Rd = R_str / Md
Rv = R_str / Mv

D0 = 2.26e-5 * si.metre ** 2 / si.second
D_exp = 1.81

K0 = 2.4e-2 * si.joules / si.metres / si.seconds / si.kelvins

p1000 = 1000 * si.hectopascals
c_pd = 1005 * si.joule / si.kilogram / si.kelvin
c_pv = 1850 * si.joule / si.kilogram / si.kelvin
T0 = sci.zero_Celsius * si.kelvin
g_std = sci.g * si.metre / si.second ** 2

c_pw = 4218 * si.joule / si.kilogram / si.kelvin

Rd_over_c_pd = Rd / c_pd

ARM_C1 = 6.1094 * si.hectopascal
ARM_C2 = 17.625 * si.dimensionless
ARM_C3 = 243.04 * si.kelvin

FWC_C0 = 6.115836990e000 * si.hPa
FWC_C1 = 0.444606896e000 * si.hPa / si.K
FWC_C2 = 0.143177157e-01 * si.hPa / si.K**2
FWC_C3 = 0.264224321e-03 * si.hPa / si.K**3
FWC_C4 = 0.299291081e-05 * si.hPa / si.K**4
FWC_C5 = 0.203154182e-07 * si.hPa / si.K**5
FWC_C6 = 0.702620698e-10 * si.hPa / si.K**6
FWC_C7 = 0.379534310e-13 * si.hPa / si.K**7
FWC_C8 = -.321582393e-15 * si.hPa / si.K**8

rho_w = 1 * si.kilograms / si.litres
sgm_w = 0.072 * si.joule / si.metre ** 2  # TODO #223 temperature dependence

p_tri = 611.73 * si.pascal
T_tri = 273.16 * si.kelvin
l_tri = 2.5e6 * si.joule / si.kilogram

# standard pressure and temperature (ICAO)
T_STP = T0 + 15 * si.kelvin
p_STP = 101325 * si.pascal
rho_STP = p_STP / Rd / T_STP

ppt = 1e-12
ppb = 1e-9
ppm = 1e-6
ROOM_TEMP = T_tri + 25 * si.K
M = si.mole / si.litre
H_u = M / p_STP
dT_u = si.K

# there are so few water ions instead of K we have K [H2O] (see Seinfeld & Pandis p 345)
K_H2O = 1e-14 * M * M
