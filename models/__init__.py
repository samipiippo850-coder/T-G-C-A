from . import TGCA_Mechanism_V3
from . import TGCA_SOTA_Adapter_V2
from . import TGCA_CPTRD_V4_AUDIT
from . import TGCA_CPTRD_V4
from . import TGCA_CPTRD
from . import TGCA_SOTA_Adapter
# Minimal registry for PL1720 full16 experiments.

from . import DLinear
from . import PatchTST
from . import TimeBridge
from . import iTransformer

MODEL_DICT = {
    'TGCA_Mechanism_V3': TGCA_Mechanism_V3,
    'TGCA_SOTA_Adapter_V2': TGCA_SOTA_Adapter_V2,
    'TGCA_SOTA_Adapter': TGCA_SOTA_Adapter,
    'TGCA_CPTRD_V4_AUDIT': TGCA_CPTRD_V4_AUDIT,

    'TGCA_CPTRD_V4': TGCA_CPTRD_V4,

    'TGCA_CPTRD': TGCA_CPTRD,

    "DLinear": DLinear,
    "PatchTST": PatchTST,
    "TimeBridge": TimeBridge,
    "iTransformer": iTransformer,
}

MODEL_REQUIRES_CYCLE = set()
