from PySDM.physics import constants
from PySDM.physics.formulae import Formulae
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestFormulae:

    @staticmethod
    def test_pvs():
        with DimensionalAnalysis():
            # Arrange
            formulae = Formulae()
            si = constants.si
            sut = formulae.saturation_vapour_pressure.pvs_Celsius
            T = 300 * si.kelvins

            # Act
            pvs = sut(T)

            # Assert
            assert pvs.units == si.hectopascals

    @staticmethod
    def test_r_cr():
        with DimensionalAnalysis():
            # Arrange
            si = constants.si
            formulae = Formulae()
            sut = formulae.hygroscopicity.r_cr

            kp = .5
            rd = .1 * si.micrometre
            T = 300 * si.kelvins
            sgm = constants.sgm_w

            # Act
            r_cr = sut(kp, rd**3, T, sgm)

            # Assert
            assert r_cr.to_base_units().units == si.metres

    @staticmethod
    def test_lv():
        with DimensionalAnalysis():
            # Arrange
            si = constants.si
            T = 300 * si.kelvins

            formulae = Formulae()
            sut = formulae.latent_heat.lv

            # Act
            latent_heat = sut(T)

            # Assert
            assert latent_heat.check('[energy]/[mass]')

    @staticmethod
    def test___str__():
        # Arrange
        sut = Formulae()

        # Act
        result = str(sut)

        # Assert
        assert len(result) > 0
