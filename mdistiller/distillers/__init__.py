from ._base import Vanilla
from .KD import KD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .CAT_KD import CAT_KD
from .CAT_hcl_KD import CAT_hcl_KD
from .CAT_decoupled_KD import CAT_TEST_KD
from .transfer import transfer
from .SIMkd import SimKD
from .SIMkd_test_FrequencySmooth import SimKD as SimKD_frequencySmooth
from .SIMkd_test_FrequencySmooth_test import SimKD as SimKD_FE_gate
from .SFAKD import SFAKD
from .SFAKD_oblation1 import SFAKD as SFAKD_oblation1
from .SFAKD_oblation2 import SFAKD as SFAKD_oblation2
from .SFAKD_oblation3 import SFAKD as SFAKD_oblation3
from .SFAKD_oblation4 import SFAKD as SFAKD_oblation4
from .SFAKD_oblation4_1 import SFAKD as SFAKD_oblation4_1


distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
    "CAT_KD": CAT_KD,
    "CAT_hcl_KD": CAT_hcl_KD,
    "CAT_test_KD": CAT_TEST_KD,
    "SIMKD": SimKD,
    "SIMKD_FS": SimKD_frequencySmooth,
    "SIMKD_FS_GATE": SimKD_FE_gate,
    "SFAKD": SFAKD,
    "SFAKD_oblation1": SFAKD_oblation1,
    "SFAKD_oblation2": SFAKD_oblation2,
    "SFAKD_oblation3": SFAKD_oblation3,
    "SFAKD_oblation4": SFAKD_oblation4,
    "SFAKD_oblation4_1": SFAKD_oblation4_1,
    'transfer' :transfer,
}
