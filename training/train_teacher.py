#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, logging, random, sys, time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from datasets.public_ssvep import PublicSSVEPDataset
from models.fbtcnn import FBT_CNN
from training.filterbank import load_or_create_cache
from utils.config import load_yaml, apply_cli_overrides, deep_get, resolve_project_path

LOG = logging.getLogger("cckd.teacher")

def parser():
    p=argparse.ArgumentParser(description="Module 1: FB-tCNN 9-channel teacher baseline")
    p.add_argument("--config", required=True)
    p.add_argument("--data_root")
    p.add_argument("--output_dir")
    p.add_argument("--cache_dir")
    p.add_argument("--n_subjects", type=int)
    p.add_argument("--n_folds", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--device")
    p.add_argument("--workers", type=int)
    p.add_argument("--seed", type=int)
    return p

def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def split_subjects(remaining, frac, seed):
    a=np.array(sorted(remaining.tolist())); rng=np.random.default_rng(seed); rng.shuffle(a)
    n=max(1, min(len(a)-1, int(round(len(a)*frac))))
    return a[n:], a[:n]

def make_loader(X,y,mask,batch,shuffle,workers):
    idx=np.flatnonzero(mask)
    ds=TensorDataset(torch.from_numpy(X[idx].astype(np.float32,copy=False)), torch.from_numpy(y[idx].astype(np.int64,copy=False)))
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=torch.cuda.is_available(),persistent_workers=workers>0)

def epoch(model,loader,device,opt=None,scaler=None):
    train=opt is not None; model.train(train); total=correct=0; loss_sum=0.0
    ctx=torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb,yb in loader:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
            if train: opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits=model(xb); loss=F.cross_entropy(logits,yb)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); opt.step()
            loss_sum+=loss.item()*yb.numel(); correct+=(logits.argmax(1)==yb).sum().item(); total+=yb.numel()
    return {"loss":loss_sum/max(total,1),"acc":correct/max(total,1)}

def main():
    args=parser().parse_args(); cfg=apply_cli_overrides(load_yaml(args.config),args)
    out=resolve_project_path(deep_get(cfg,"output.output_dir"),PROJECT_ROOT); cache=resolve_project_path(deep_get(cfg,"output.cache_dir"),PROJECT_ROOT)
    out.mkdir(parents=True,exist_ok=True); cache.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.StreamHandler(),logging.FileHandler(out/"training.log",encoding="utf-8")])
    seed=int(deep_get(cfg,"runtime.seed",2026)); seed_all(seed)
    devname=deep_get(cfg,"runtime.device","auto"); device=torch.device("cuda" if torch.cuda.is_available() else "cpu") if devname=="auto" else torch.device(devname)
    data_root=Path(deep_get(cfg,"dataset.data_root")).expanduser().resolve()
    LOG.info("config=%s",Path(args.config).resolve()); LOG.info("data_root=%s",data_root); LOG.info("device=%s",device)
    ds=PublicSSVEPDataset(data_root,deep_get(cfg,"dataset.name"),deep_get(cfg,"dataset.sfreq",250),deep_get(cfg,"dataset.t_start",0.14),deep_get(cfg,"dataset.window_seconds",1.0),deep_get(cfg,"dataset.mat_key","data"),verbose=False)
    X,y,sid=ds.load_all_subjects(int(deep_get(cfg,"dataset.n_subjects")))
    ch=list(map(int,deep_get(cfg,"dataset.teacher_channel_indices")))
    if max(ch)>=X.shape[1]: raise IndexError(f"teacher channel index exceeds raw channels: max={max(ch)}, raw={X.shape[1]}")
    X=X[:,ch,:]; bands=[tuple(map(float,b)) for b in deep_get(cfg,"filterbank.bands_hz")]
    Xfb=load_or_create_cache(X,cache,deep_get(cfg,"dataset.name"),float(deep_get(cfg,"dataset.sfreq")),bands,ch)
    LOG.info("loaded raw_selected=%s filterbank=%s labels=%d subjects=%d",X.shape,Xfb.shape,len(np.unique(y)),len(np.unique(sid)))
    subjects=np.unique(sid); nf=deep_get(cfg,"evaluation.n_folds")
    if nf is not None: subjects=subjects[:int(nf)]
    results=[]; params=feature_dim=None
    for fi,test_sub in enumerate(subjects,1):
        rem=np.unique(sid); rem=rem[rem!=test_sub]
        tr_sub,va_sub=split_subjects(rem,float(deep_get(cfg,"training.val_fraction_subjects",.15)),seed+fi)
        tr=np.isin(sid,tr_sub); va=np.isin(sid,va_sub); te=sid==test_sub
        assert not np.any(tr&va) and not np.any(tr&te) and not np.any(va&te)
        batch=int(deep_get(cfg,"training.batch_size",64)); workers=int(deep_get(cfg,"runtime.workers",2))
        tl=make_loader(Xfb,y,tr,batch,True,workers); vl=make_loader(Xfb,y,va,batch,False,workers); ql=make_loader(Xfb,y,te,batch,False,workers)
        model=FBT_CNN(n_channels=Xfb.shape[2],n_samples=Xfb.shape[3],n_classes=int(deep_get(cfg,"dataset.n_classes",40)),n_subbands=Xfb.shape[1],branch_filters=int(deep_get(cfg,"model.branch_filters",16)),fusion_filters=int(deep_get(cfg,"model.fusion_filters",32)),temporal_stride=int(deep_get(cfg,"model.temporal_stride",5)),local_kernel=int(deep_get(cfg,"model.local_kernel",5)),dropout=float(deep_get(cfg,"model.dropout",.4))).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=float(deep_get(cfg,"training.learning_rate",1e-3)),weight_decay=float(deep_get(cfg,"training.weight_decay",1e-4)))
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=int(deep_get(cfg,"training.reduce_lr_patience",7)),min_lr=float(deep_get(cfg,"training.min_lr",1e-6)))
        amp=bool(deep_get(cfg,"training.amp",True)); scaler=torch.amp.GradScaler("cuda") if device.type=="cuda" and amp else None
        best=-1.; state=None; best_ep=0; wait=0; hist=[]; start=time.time()
        for ep in range(1,int(deep_get(cfg,"training.epochs",150))+1):
            a=epoch(model,tl,device,opt,scaler); b=epoch(model,vl,device); sched.step(b["acc"]); hist.append({"epoch":ep,"train":a,"val":b})
            if b["acc"]>best+float(deep_get(cfg,"training.min_delta",1e-4)):
                best=b["acc"]; best_ep=ep; state=copy.deepcopy(model.state_dict()); wait=0
            else: wait+=1
            if ep==1 or ep%int(deep_get(cfg,"runtime.log_every",10))==0: LOG.info("fold=%d testS=%s ep=%d train=%.4f val=%.4f best=%.4f",fi,test_sub,ep,a["acc"],b["acc"],best)
            if wait>=int(deep_get(cfg,"training.early_stopping_patience",20)): break
        model.load_state_dict(state); tst=epoch(model,ql,device)
        ck=out/f"fbtcnn_teacher_fold{fi:02d}.pt"
        torch.save({"model_state":model.state_dict(),"config":cfg,"fold":fi,"test_subject":int(test_sub),"train_subjects":tr_sub.tolist(),"val_subjects":va_sub.tolist(),"best_epoch":best_ep,"best_val_acc":best,"test":tst},ck)
        row={"fold":fi,"test_subject":int(test_sub),"best_epoch":best_ep,"best_val_acc":float(best),"test_acc":float(tst["acc"]),"test_loss":float(tst["loss"]),"time_sec":time.time()-start,"checkpoint":str(ck)}
        if bool(deep_get(cfg,"evaluation.save_history",False)): row["history"]=hist
        results.append(row); params=model.count_params(); feature_dim=model.get_feature_dim(); LOG.info("DONE fold=%d test=%.4f",fi,tst["acc"])
    acc=np.array([r["test_acc"] for r in results]); summary={"project":"CCKD-SSVEP","version":"0.1.0","module":"FB-tCNN teacher baseline","dataset":deep_get(cfg,"dataset.name"),"data_root":str(data_root),"mean_test_acc":float(acc.mean()),"std_test_acc":float(acc.std()),"min_test_acc":float(acc.min()),"max_test_acc":float(acc.max()),"n_folds":len(results),"params":params,"feature_dim":feature_dim,"folds":results,"config":cfg}
    with (out/"teacher_results.json").open("w",encoding="utf-8") as f: json.dump(summary,f,indent=2)
    LOG.info("SUMMARY %.4f +/- %.4f | saved=%s",acc.mean(),acc.std(),out)
if __name__=="__main__": main()
