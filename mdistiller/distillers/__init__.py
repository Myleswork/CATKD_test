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
from .SIMkd_test import SimKD as SimKD_test
from .SIMkd_test_1 import SimKD as SimKD_test_1
from .SIMkd_test_FrequencySmooth import SimKD as SimKD_frequencySmooth
from .SIMkd_test_FrequencySmooth_gate import SimKD as SimKD_FE_gate


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
    "SIMKD_test": SimKD_test,
    "SIMKD_test_1": SimKD_test_1,
    "SIMKD_FE": SimKD_frequencySmooth,
    "SIMKD_GATE": SimKD_FE_gate,
    'transfer' :transfer,
}
