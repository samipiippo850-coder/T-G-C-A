"""Opt-in epoch-boundary resume and explicitly recorded legacy weight recovery."""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil

import numpy as np
import torch

FIELDS = ('model', 'data', 'root_path', 'data_path', 'features', 'target',
          'freq', 'seq_len', 'label_len', 'pred_len', 'enc_in', 'dec_in',
          'c_out', 'd_model', 'n_heads', 'e_layers', 'd_layers', 'd_ff',
          'embed', 'batch_size', 'learning_rate', 'train_epochs', 'patience',
          'optim_type', 'lradj', 'pct_start', 'auxi_lambda', 'rec_lambda',
          'fix_seed', 'loss', 'min_lr')
# ===== TGCA_SPLIT_LR_V1 =====
# Branch-specific hyper-parameters that must not silently change on resume.
EXTRA_FIELDS = ('base_seq_len', 'base_learning_rate', 'adapter_learning_rate',
                # ===== TGCA_WEIGHT_DECAY_V1 =====
                'cf_weight_decay')
# ===== TGCA_SPLIT_LR_V1_END =====
ENV_FIELDS = ('TGCA_ERROR_ALIGNED_ROUTER', 'TGCA_ADAPTER_MAX_RMS',
              'TGCA_SIGNED_CORRECTION_RMS_CAP',
              # ===== TGCA_WEIGHT_DECAY_V1 =====
              'TGCA_ADAPTER_PATCH_TOKENS')

def enabled():
    return os.environ.get('TGCA_RESUME_TRAINING', '0') == '1'

def signature(args):
    values = {key: getattr(args, key, None) for key in FIELDS}
    # ===== TGCA_SPLIT_LR_V1 =====
    values.update({key: getattr(args, key, None) for key in EXTRA_FIELDS})
    # ===== TGCA_SPLIT_LR_V1_END =====
    return {'args': values,
            'env': {key: os.environ.get(key) for key in ENV_FIELDS}}

def signature_mismatch(saved, current):
    """Diff two signatures, tolerating keys a legacy state predates."""
    mismatch = {}
    for section in ('args', 'env'):
        saved_section = (saved or {}).get(section) or {}
        current_section = (current or {}).get(section) or {}
        for key in sorted(set(saved_section) | set(current_section)):
            if key not in saved_section:
                continue
            if key not in current_section or saved_section[key] != current_section[key]:
                mismatch[f'{section}.{key}'] = [saved_section.get(key),
                                                current_section.get(key)]
    return mismatch

def load(path):
    safe_numpy_types = [np._core.multiarray._reconstruct, np.ndarray, np.dtype,
                        type(np.dtype('uint32'))]
    with torch.serialization.safe_globals(safe_numpy_types):
        return torch.load(path, map_location='cpu', weights_only=True)

def prepare(exp, path, optimizer, scheduler, early_stopping, train_steps,
            vali_data=None, vali_loader=None, criterion=None):
    if not enabled():
        return 0
    checkpoint_dir = Path(path)
    state_path = checkpoint_dir / 'training_state.pt'
    if state_path.is_file():
        state = load(state_path)
        mismatch = signature_mismatch(state['signature'], signature(exp.args))
        if mismatch:
            raise RuntimeError('Resume configuration differs from saved training state: '
                               + json.dumps(mismatch, ensure_ascii=False))
        if state['train_steps'] != train_steps:
            raise RuntimeError('Resume training window/batch count differs')
        exp.model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        if scheduler.scheduler is not None:
            scheduler.scheduler.load_state_dict(state['scheduler'])
        scheduler.last_lr = state['last_lr']
        early_stopping.__dict__.update(state['early_stopping'])
        exp.step = state['step']
        random.setstate(state['random_state'])
        np.random.set_state(state['numpy_state'])
        torch.set_rng_state(state['torch_state'])
        if torch.cuda.is_available() and state['cuda_state'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_state'])
        epoch = int(state['epoch'])
        print(f'[TGCA_FULL_RESUME] completed_epochs={epoch} next_epoch={epoch + 1} '
              f'optimizer_restored=True scheduler_restored=True lr={scheduler.last_lr}', flush=True)
        return exp.args.train_epochs if early_stopping.early_stop else epoch

    legacy_root = os.environ.get('TGCA_RESUME_LEGACY_ROOT', '')
    output_root = os.environ.get('TGCA_RESUME_OUTPUT_ROOT', '')
    if not legacy_root or not output_root:
        return 0
    relative = checkpoint_dir.resolve().relative_to(Path(output_root).resolve())
    old_checkpoint_dir = Path(legacy_root).resolve() / relative
    old_log = old_checkpoint_dir.parent.parent / 'logs' / 'train.log'
    old_checkpoint = old_checkpoint_dir / 'checkpoint.pth'
    if not old_log.exists() and not old_checkpoint.exists():
        print('[TGCA_NEW_CANDIDATE] no prior run; first training start', flush=True)
        return 0
    if not old_checkpoint.is_file() or not old_log.is_file():
        raise RuntimeError('Incomplete legacy evidence: checkpoint and validation log required')
    records = re.findall(r'Epoch: (\d+), Steps: (\d+) \| Train Loss: (\S+) Vali Loss: (\S+)',
                         old_log.read_text(errors='replace'))
    finite = [(int(epoch), int(step), float(value)) for epoch, step, _, value in records
              if np.isfinite(float(value))]
    if not finite:
        raise RuntimeError('Legacy log has no finite validation checkpoint evidence')
    epoch, step, val = min(finite, key=lambda row: (row[2], -row[0]))
    if step != epoch * train_steps or epoch >= exp.args.train_epochs:
        raise RuntimeError('Legacy checkpoint progress is inconsistent with training budget')
    exp.model.load_state_dict(load(old_checkpoint), strict=True)
    if vali_loader is not None:
        observed = float(exp.vali(vali_data, vali_loader, criterion))
        print(f'[TGCA_RECOVERY_VALIDATION] logged={val} reloaded={observed}', flush=True)
        if not np.isclose(observed, val, rtol=1e-4, atol=1e-6):
            raise RuntimeError('Loaded legacy checkpoint does not reproduce its validation score')
    shutil.copy2(old_checkpoint, checkpoint_dir / 'checkpoint.pth')
    if scheduler.scheduler_type != 'TST':
        raise RuntimeError('Legacy schedule reconstruction only audited for TST/OneCycleLR')
    scheduler.scheduler.step(step)
    scheduler.lr_info(verbose=False)
    early_stopping.best_score = -val
    early_stopping.val_loss_min = val
    early_stopping.counter = 0
    exp.step = step
    provenance = {'recovery_mode': 'legacy_weights_optimizer_reset',
                  'checkpoint': str(old_checkpoint), 'log': str(old_log),
                  'checkpoint_sha256': hashlib.sha256(old_checkpoint.read_bytes()).hexdigest(),
                  'completed_epochs': epoch, 'next_epoch': epoch + 1,
                  'remaining_epochs': exp.args.train_epochs - epoch,
                  'best_validation_mse': val, 'optimizer_state_available': False,
                  'schedule_position_reconstructed': step, 'signature': signature(exp.args)}
    (checkpoint_dir / 'recovery_provenance.json').write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + '\n')
    print('[TGCA_LEGACY_WEIGHT_RECOVERY] ' + json.dumps(provenance), flush=True)
    best_train = next(train for e, s, train, value in reversed(records)
                      if int(e) == epoch and int(s) == step)
    print('[TGCA_RECOVERED_BEST] validation record from saved checkpoint', flush=True)
    print(f'Epoch: {epoch}, Steps: {step} | Train Loss: {best_train} Vali Loss: {val:.7f}', flush=True)
    save(exp, path, optimizer, scheduler, early_stopping, train_steps, epoch)
    return epoch

def save(exp, path, optimizer, scheduler, early_stopping, train_steps, epoch):
    if not enabled():
        return
    checkpoint_dir = Path(path)
    state = {'version': 1, 'signature': signature(exp.args), 'train_steps': train_steps,
             'model': exp.model.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.scheduler.state_dict() if scheduler.scheduler is not None else None,
             'last_lr': scheduler.last_lr, 'early_stopping': dict(early_stopping.__dict__),
             'epoch': int(epoch), 'step': exp.step,
             'random_state': random.getstate(), 'numpy_state': np.random.get_state(),
             'torch_state': torch.get_rng_state(),
             'cuda_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    temporary = checkpoint_dir / 'training_state.pt.tmp'
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, checkpoint_dir / 'training_state.pt')
    print(f'[TGCA_TRAIN_STATE_SAVED] epoch={epoch} step={exp.step}', flush=True)
