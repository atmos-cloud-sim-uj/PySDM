from PySDM.attributes.impl.derived_attribute import DerivedAttribute
from PySDM.physics import constants as const

# TODO #223
# first attempt to make OrganicFraction a derived attribute
class OrganicFraction(DerivedAttribute):
    def __init__(self, builder):
        self.volume_dry_org = builder.get_attribute('dry volume organic')
        self.volume_dry_total = builder.get_attribute('dry volume')
        dependencies = [self.volume_dry_org, self.volume_dry_total]
        super().__init__(builder, name='organic fraction', dependencies=dependencies)

    def recalculate(self):
        self.data.idx = self.volume_dry_org.data.idx
        self.data.product(self.volume_dry_org.get(), 1/self.volume_dry_total.get())