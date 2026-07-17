#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from datasets.public_ssvep import PublicSSVEPDataset
from utils.config import load_yaml,deep_get
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data_root'); p.add_argument('--subject',type=int,default=1); a=p.parse_args()
c=load_yaml(a.config); root=a.data_root or deep_get(c,'dataset.data_root')
ds=PublicSSVEPDataset(root,deep_get(c,'dataset.name'),deep_get(c,'dataset.sfreq'),deep_get(c,'dataset.t_start'),deep_get(c,'dataset.window_seconds'),deep_get(c,'dataset.mat_key','data'),True)
X,y=ds.load_subject(a.subject); ch=deep_get(c,'dataset.teacher_channel_indices')
print('\nDATASET CHECK PASSED')
print('data_root :',Path(root).expanduser().resolve()); print('X subject :',X.shape); print('labels    :',y.min(),y.max(),'classes=',len(np.unique(y))); print('teacher X :',X[:,ch,:].shape); print('indices   :',ch); print('names     :',deep_get(c,'dataset.teacher_channel_names'))
