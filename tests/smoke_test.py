from pathlib import Path
import sys,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.fbtcnn import FBT_CNN
from training.filterbank import apply_filterbank
x=np.random.default_rng(2026).normal(size=(8,9,250)).astype('float32')
fb=apply_filterbank(x,250,[(6,90),(14,90),(22,90),(30,90)])
m=FBT_CNN(9,250,40,4); z,f=m(torch.from_numpy(fb[:4]),return_features=True)
assert fb.shape==(8,4,9,250) and z.shape==(4,40) and f.shape==(4,32)
torch.nn.functional.cross_entropy(z,torch.tensor([0,1,2,3])).backward()
print('SMOKE TEST PASSED');print('filterbank',fb.shape,'logits',tuple(z.shape),'feature',tuple(f.shape),'params',m.count_params())
