from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute
from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.physics import constants as const

class DryVolumeStatic(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name='dry volume')

class DryVolumeDynamic(DerivedAttribute):
    def __init__(self, builder):
        self.core = builder.core
        self.moles_S_VI = builder.get_attribute('moles_S_VI')
        super().__init__(builder, name='dry volume', dependencies=(self.moles_S_VI,))

    def recalculate(self):
        dynamic = self.core.dynamics['AqueousChemistry']
        self.data.data[:] = self.moles_S_VI.data.data[:]
        self.data.data[:] *= dynamic.dry_molar_mass / dynamic.dry_rho

# TODO #223
# first attempt to add inorganic and organic dry volumes as extensive attributes
# and make DryVolumeOrgInorg a derived attribute
class DryVolumeInorganic(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name='dry volume inorganic')

class DryVolumeOrganic(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name='dry volume organic')

class DryVolumeOrgInorg(DerivedAttribute):
    def __init__(self, builder):
        self.volume_dry_org = builder.get_attribute('dry volume organic')
        self.volume_dry_inorg = builder.get_attribute('dry volume inorganic')
        dependencies = [self.volume_dry_org, self.volume_dry_inorg]
        super().__init__(builder, name='dry volume', dependencies=dependencies)

    def recalculate(self):
        self.data.idx = self.volume_dry_org.data.idx
        self.data.sum(self.volume_dry_org.get(), self.volume_dry_inorg.get())