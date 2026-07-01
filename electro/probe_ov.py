#!/usr/bin/env python3
"""Can a strong pairwise-overlap penalty make eDensity place() self-legalize
(low overlap) at high util inside the clamp box?  Sweep OV1."""
import math, os, sys
import numpy as np, torch
from litetestLoader import FloorplanDatasetLiteTest
sys.path.insert(0, "/home/pop/2026_EDA_contest/electro")
import analytical_place as ap
from legalize import verify_overlap
ds = FloorplanDatasetLiteTest("../")

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

for tid in [0, 60, 99]:
    s = ds[tid]; area_t,b2b,p2b,pins,cons = s['input']; polys,_=s['label']
    n = int((area_t!=-1).sum().item()); tp=build_tp(cons,polys,n)
    tot = float(sum(area_t[i] for i in range(n) if area_t[i]>0))
    os.environ["ELECTRO_EDENSITY"]="2"; os.environ["ELECTRO_EDENSITY_UTIL"]="0.85"
    for ov1 in [2.5, 10, 30, 80]:
        os.environ["ELECTRO_OV1"]=str(ov1); os.environ["ELECTRO_OV0"]=str(min(ov1,2.0))
        out,_ = ap.place(n,area_t,b2b,p2b,pins,cons,tp,iters=600,lr=0.02,seed=0,device="cpu")
        x=np.array([o[0] for o in out]);y=np.array([o[1] for o in out])
        w=np.array([o[2] for o in out]);h=np.array([o[3] for o in out])
        bb=(max(x+w)-min(x))*(max(y+h)-min(y)); util=tot/bb
        ov=verify_overlap(x,y,w,h); ovpct=100*ov/tot
        print(f"tid {tid:>3} ov1={ov1:>5} | util={util:.3f} ov%={ovpct:5.2f} "
              f"minxy=({min(x):.1f},{min(y):.1f})")
    print()
