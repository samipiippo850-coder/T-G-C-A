"""Reproducible ablations via class-level overrides; no shared source modifications."""
import os,runpy,sys,torch,hashlib
MODE=os.environ.get('TGCA_ABLATION_MODE','full')
MODES={'full','no_adapter','drop_trend','drop_local','drop_spectral','uniform_gate','no_router','no_direction','no_cap'}
if MODE not in MODES:raise ValueError('Unknown ablation '+MODE)
# Instantiate the same optional parameter set and consume the same RNG sequence for every mode.
os.environ['TGCA_COV_THERMO_ROUTER']='0';os.environ['TGCA_COV_HARD_MATRIX']='0';os.environ['TGCA_COV_HARD_PHASE_GUARD']='0'
os.environ['TGCA_COV_HARD_GATE_MODE']='low_power_free'
os.environ['TGCA_ERROR_ALIGNED_ROUTER']='1';os.environ['TGCA_ADAPTER_ENERGY_TEMP']='0';os.environ['TGCA_ADAPTER_LEARNABLE_ENERGY_TEMP']='0'
from models.TGCA_SOTA_Adapter_V2 import HeterogeneousResidualAdapter as A, Model as ModelWrapper
old_model_init=ModelWrapper.__init__
old_init=A.__init__;old_gate=A._moment_gate;old_forward=A.forward
old_cov=A._covariance_thermo_features
old_experts={'drop_trend':A._trend_expert,'drop_local':A._local_expert,'drop_spectral':A._spectral_expert}

def cov_features(self,*args,**kwargs):
 features,risk,diagnostics=old_cov(self,*args,**kwargs)
 return features,risk.reshape(-1,1),diagnostics

def init(self,*args,**kwargs):
 old_init(self,*args,**kwargs)
 if MODE=='no_router':self.error_aligned_router_enabled=False
 if MODE=='no_cap':self.max_rms=1e6
 if MODE=='no_cov':self.cov_thermo_enabled=False
 if MODE=='no_hard':self.cov_hard_matrix_enabled=False
 if MODE=='no_phase':self.cov_hard_phase_guard_enabled=False
 if MODE=='no_energy':self.energy_temp_enabled=False
 if MODE=='no_direction':
  # Preserve fixed positive directional amplitude; suppress additive signed head.
  self.direction_head.register_forward_hook(lambda mod,inp,out: torch.full_like(out,0.5493061443340549))
  self.signed_correction_head.register_forward_hook(lambda mod,inp,out: torch.zeros_like(out))

def gate(self,*args,**kwargs):
 w,features=old_gate(self,*args,**kwargs)
 if MODE=='uniform_gate':w=torch.full_like(w,1.0/w.shape[-1])
 elif MODE in old_experts:
  index={'drop_trend':0,'drop_local':1,'drop_spectral':2}[MODE]
  mask=torch.ones_like(w);mask[...,index]=0;w=w*mask;w=w/w.sum(dim=-1,keepdim=True).clamp_min(1e-12)
 return w,features

def forward(self,x_enc,base_prediction):
 if MODE=='no_adapter':return base_prediction
 return old_forward(self,x_enc,base_prediction)
A.__init__=init;A._moment_gate=gate;A.forward=forward;A._covariance_thermo_features=cov_features
if MODE in old_experts:
 method={'drop_trend':'_trend_expert','drop_local':'_local_expert','drop_spectral':'_spectral_expert'}[MODE]
 fn=old_experts[MODE]
 def disabled(self,*args,**kwargs):return torch.zeros_like(fn(self,*args,**kwargs))
 setattr(A,method,disabled)
def model_init(self,*args,**kwargs):
 old_model_init(self,*args,**kwargs)
 digest=hashlib.sha256()
 for key,value in self.state_dict().items():
  digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
 print('[ABLATION_INITIAL_STATE_SHA256]',digest.hexdigest(),flush=True)
ModelWrapper.__init__=model_init
print('[ABLATION_MODE]',MODE,'candidate_original cov=0 error_router=1 energy_temp=0 (historical_energy_unverified)',flush=True)
runpy.run_path(os.environ.get('TGCA_ABLATION_RUNPY','/root/DistDF-main/DistDF-main_competition/run.py'),run_name='__main__')
