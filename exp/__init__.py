from .exp_long_term_forecasting import Exp_Long_Term_Forecast
from .exp_long_term_forecasting_hf_amp import Exp_Long_Term_Forecast_HF_Amp
from .exp_long_term_forecasting_quantile import Exp_Long_Term_Forecast_Quantile
from .exp_long_term_forecasting_grouped import Exp_Long_Term_Forecast_Grouped
from .exp_long_term_forecasting_grouped_hf import Exp_Long_Term_Forecast_Grouped_HF
from .exp_long_term_forecasting_grouped_hf_turn import Exp_Long_Term_Forecast_Grouped_HF_Turn

EXP_DICT = {
    'long_term_forecast': Exp_Long_Term_Forecast,
    'long_term_forecast_hf_amp': Exp_Long_Term_Forecast_HF_Amp,
    'long_term_forecast_quantile': Exp_Long_Term_Forecast_Quantile,
    'long_term_forecast_grouped': Exp_Long_Term_Forecast_Grouped,
    'long_term_forecast_grouped_hf': Exp_Long_Term_Forecast_Grouped_HF,
    'long_term_forecast_grouped_hf_turn': Exp_Long_Term_Forecast_Grouped_HF_Turn,
}

from .exp_joint_tgca import Exp_Joint_TGCA
EXP_DICT["joint_tgca"] = Exp_Joint_TGCA

from .exp_joint_tgca_pcsafe import Exp_Joint_TGCA_PCSafe
EXP_DICT["joint_tgca_pcsafe"] = Exp_Joint_TGCA_PCSafe

from .exp_joint_tgca_casafe import Exp_Joint_TGCA_CASafe
EXP_DICT["joint_tgca_casafe"] = Exp_Joint_TGCA_CASafe

from .exp_joint_tgca_b1 import Exp_Joint_TGCA_B1
EXP_DICT["joint_tgca_b1"] = Exp_Joint_TGCA_B1
