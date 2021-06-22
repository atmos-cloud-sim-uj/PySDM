from PySDM.attributes.physics.multiplicities import Multiplicities
from PySDM.attributes.physics.volume import Volume
from PySDM.attributes.physics.dry_volume import DryVolumeOrganic, DryVolumeInorganic
from PySDM.attributes.physics.dry_volume import DryVolumeOrgInorg, DryVolumeDynamic, DryVolumeStatic
from PySDM.attributes.physics.organic_fraction import OrganicFraction
from PySDM.attributes.physics.radius import Radius
from PySDM.attributes.physics.dry_radius import DryRadius
from PySDM.attributes.physics.terminal_velocity import TerminalVelocity
from PySDM.attributes.numerics.cell_id import CellID
from PySDM.attributes.numerics.cell_origin import CellOrigin
from PySDM.attributes.numerics.position_in_cell import PositionInCell
from PySDM.attributes.physics.temperature import Temperature
from PySDM.attributes.physics.heat import Heat
from PySDM.attributes.physics.critical_volume import CriticalVolume
from PySDM.attributes.chemistry.mole_amount import MoleAmount
from PySDM.attributes.chemistry.concentration import Concentration
from PySDM.attributes.chemistry.pH import pH
from PySDM.attributes.chemistry.hydrogen_ion_concentration import HydrogenIonConcentration
from PySDM.physics.aqueous_chemistry.support import AQUEOUS_COMPOUNDS
from functools import partial

attributes = {
    'n': lambda _: Multiplicities,
    'volume': lambda _: Volume,
    'dry volume organic': lambda _: DryVolumeOrganic,
    'dry volume inorganic': lambda _: DryVolumeInorganic,
    # 'dry volume': lambda dynamics: DryVolumeDynamic if 'AqueousChemistry' in dynamics else (DryVolumeOrgInorg if 'SurfaceOrganics' in dynamics else DryVolumeStatic),
    'dry volume': lambda dynamics: DryVolumeDynamic if 'AqueousChemistry' in dynamics else DryVolumeStatic,
    'organic fraction': lambda _: OrganicFraction,
    'radius': lambda _: Radius,
    'dry radius': lambda _: DryRadius,
    'terminal velocity': lambda _: TerminalVelocity,
    'cell id': lambda _: CellID,
    'cell origin': lambda _: CellOrigin,
    'position in cell': lambda _: PositionInCell,
    'temperature': lambda _: Temperature,
    'heat': lambda _: Heat,
    'critical volume': lambda _: CriticalVolume,
    **{"moles_" + compound: partial(lambda _, c: MoleAmount(c), c=compound)
       for compound in AQUEOUS_COMPOUNDS.keys()},
    **{"conc_" + compound: partial(lambda _, c: Concentration(c), c=compound)
       for compound in AQUEOUS_COMPOUNDS.keys()},
    'pH': lambda _: pH,
    'conc_H': lambda _: HydrogenIonConcentration
}


def get_class(name, dynamics):
    return attributes[name](dynamics)
