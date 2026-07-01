#!/usr/bin/env python3
"""Does CLAMP=1 actually yield first-quadrant FINAL output (after legalize+repair+
remove_overlap), or do the post-place passes nudge blocks back negative?"""
import os, sys
import numpy as np, torch
from litetestLoader import FloorplanDatasetLiteTest
sys.path.insert(0, "/home/pop/2026_EDA_contest/electro")
import electro_parallel as ep

ds = FloorplanDatasetLiteTest("../")
SUB = [0, 40, 60, 80, 99]

def build_tp(cons, polys, n):
    tp = torch.full((n, 4), -1.0)
    for i in range(n):
        v = polys[i][polys[i][:, 0] != -1]
        if len(v) == 0: continue
        mn = v.min(dim=0).values; mx = v.max(dim=0).values
        x,y,w,h = float(mn[0]),float(mn[1]),float(mx[0]-mn[0]),float(mx[1]-mn[1])
        if cons[i,1]!=0: tp[i]=torch.tensor([x,y,w,h])
        elif cons[i,0]!=0: tp[i,2]=w; tp[i,3]=h
    return tp

def run(tid, env):
    for k,v in env.items(): os.environ[k]=str(v)
    for k in ("ELECTRO_CLAMP","ELECTRO_NONNEG","ELECTRO_CLAMP_START"):
        if k not in env: os.environ.pop(k, None)
    s = ds[tid]; area_t,b2b,p2b,pins,cons = s['input']; polys,_=s['label']
    n = int((area_t!=-1).sum().item()); tp=build_tp(cons,polys,n)
    cn = cons[:n].numpy()
    P = {"n":n,"area":area_t,"b2b":b2b,"p2b":p2b,"pins":pins,"cons":cons,"tp":tp,
         "iters":600,"lr":0.02,"device":"cpu","init":None,
         "is_pre":(cn[:,1]!=0).astype(bool),
         "clust_id":cn[:,3].astype(int) if cn.shape[1]>3 else np.zeros(n,int),
         "bcode":cn[:,4].astype(int) if cn.shape[1]>4 else np.zeros(n,int),
         "rounds":3,"nonneg":env.get("ELECTRO_NONNEG","0")=="1"}
    x,y,w,h = ep.run_start(0, P)
    return float(min(x)), float(min(y))

print(f"{'tid':>4} {'baseline':>16} {'CLAMP=1':>16} {'CLAMP+NONNEG':>16}")
for tid in SUB:
    b = run(tid, {})
    c = run(tid, {"ELECTRO_CLAMP":"1"})
    cn = run(tid, {"ELECTRO_CLAMP":"1","ELECTRO_NONNEG":"1"})
    print(f"{tid:>4} ({b[0]:6.1f},{b[1]:6.1f}) ({c[0]:6.2f},{c[1]:6.2f}) "
          f"({cn[0]:6.2f},{cn[1]:6.2f})")
