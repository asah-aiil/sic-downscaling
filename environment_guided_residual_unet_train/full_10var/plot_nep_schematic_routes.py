#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Node-to-node NEP schematic routes using land-clearance-aware A* paths."""
from __future__ import annotations
import argparse, heapq
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt

SRC="EPSG:3413"
VIEW="+proj=stere +lat_0=90 +lat_ts=70 +lon_0=112.5 +datum=WGS84 +units=m +no_defs"
TO_VIEW=Transformer.from_crs(SRC,VIEW,always_xy=True)
TO_GEO=Transformer.from_crs(SRC,"EPSG:4326",always_xy=True)
LAND="/home/ubuntu/sic_dsc/data/dataset/target/Bremen_2025.nc"

NODES={"G_west":(16,70),"K_west":(22,72),"K_kara":(60,77),"R_kara":(58,71),"K_laptev":(105,77),"K_mid":(140,74),"K_east":(142,77),"K_chukchi":(175,71),"K_bering":(193,66),"G_pacific_north":(190,60),"G_pacific_south":(170,58)}
LINKS=[("G_pacific_north","K_bering"),("K_bering","K_chukchi"),("K_chukchi","K_mid"),("K_mid","K_east"),("K_east","K_laptev"),("K_laptev","K_kara"),("K_kara","K_west"),("K_west","G_west"),("G_pacific_south","K_chukchi"),("K_chukchi","K_laptev"),("K_laptev","R_kara"),("R_kara","K_west")]

def args():
    p=argparse.ArgumentParser(); p.add_argument("--land-file",default=LAND,help="Bremen grid and static land mask; no SIC or shipping mask is read"); p.add_argument("--clearance-km",type=float,default=35.0); p.add_argument("--clearance-weight",type=float,default=6.0); p.add_argument("--hard-clearance",action="store_true",help="Enforce clearance as a hard barrier; off by default to keep narrow straits connected"); p.add_argument("--snap-km",type=float,default=220.0); p.add_argument("--arrow-width",type=float,default=3.2); p.add_argument("--output",default="/home/ubuntu/sic_dsc/fig/nep_schematic_routes_astar.svg"); p.add_argument("--view-xmin",type=float,default=-2500000); p.add_argument("--view-xmax",type=float,default=3300000); p.add_argument("--view-ymin",type=float,default=-2400000); p.add_argument("--view-ymax",type=float,default=600000); p.add_argument("--no-nodes",action="store_true"); return p.parse_args()

def read_land(path,x,y):
 with xr.open_dataset(path) as ds:
  name="z" if "z" in ds else next(k for k,v in ds.data_vars.items() if v.ndim>=2); da=ds[name]
  for d in tuple(da.dims):
   if d not in ("x","y"): da=da.isel({d:0})
  da=da.notnull().transpose("y","x")
  if da.shape!=(len(y),len(x)): da=da.interp(x=x,y=y,method="nearest")
  land=~np.asarray(da.values,bool)
 xx,yy=np.meshgrid(x,y); _,lat=TO_GEO.transform(xx,yy); land &= lat<88.0; return land

def nearest_safe(x,y,lon,lat,allowed,snap_km):
 xx,yy=np.meshgrid(x,y); gx,gy=TO_GEO.transform(xx,yy); d=((gx-lon+180)%360-180)**2+(gy-lat)**2; d[~allowed]=np.inf
 if not np.isfinite(d).any(): raise RuntimeError("no safe ocean cells")
 idx=np.unravel_index(np.argmin(d),d.shape)
 if np.sqrt(np.nanmin(d))*111>snap_km: print(f"warning: node {lon}E,{lat}N snapped farther than {snap_km:g} km")
 return tuple(int(v) for v in idx)

def astar(start,goal,allowed,weight):
 h,w=allowed.shape; sy,sx=start; gy,gx=goal
 if not allowed[sy,sx] or not allowed[gy,gx]: return []
 dist=distance_transform_edt(allowed); heap=[(0.,0.,start)]; best={start:0.}; prev={}; moves=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
 while heap:
  _,cost,u=heapq.heappop(heap)
  if u==goal:
   p=[]
   while u in prev: p.append(u); u=prev[u]
   return [start]+p[::-1]
  if cost>best.get(u,np.inf): continue
  y0,x0=u
  for dy,dx in moves:
   v=(y0+dy,x0+dx)
   if not(0<=v[0]<h and 0<=v[1]<w and allowed[v]): continue
   step=1.4142 if dy and dx else 1.; nc=cost+step*(1+weight/max(dist[v],1.)**2)
   if nc<best.get(v,np.inf): best[v]=nc; prev[v]=u; heapq.heappush(heap,(nc+np.hypot(v[0]-gy,v[1]-gx),nc,v))
 return []

def main():
 a=args()
 with xr.open_dataset(a.land_file,decode_times=False) as ds: x=np.asarray(ds.x.values); y=np.asarray(ds.y.values)
 land=read_land(Path(a.land_file),x,y); dx=float(np.median(np.abs(np.diff(x)))); dy=float(np.median(np.abs(np.diff(y)))); dist=distance_transform_edt(~land,sampling=(dy,dx)); allowed=(~land)&(dist>=a.clearance_km*1000) if a.hard_clearance else (~land); cells={k:nearest_safe(x,y,*v,allowed,a.snap_km) for k,v in NODES.items()}; mode="hard" if a.hard_clearance else "soft"; print(f"land-only routing; {mode} clearance={a.clearance_km:g} km; no SIC/shipping mask")
 fig=plt.figure(figsize=(10.5,5.2),dpi=300,frameon=False); ax=fig.add_axes([0,0,1,1])
 for i,(u,v) in enumerate(LINKS):
  path=astar(cells[u],cells[v],allowed,a.clearance_weight)
  if not path: print(f"WARNING no path: {u}->{v}"); continue
  rr,cc=np.asarray(path).T; px,py=TO_VIEW.transform(x[cc],y[rr]); col="#36b9e8" if i%2==0 else "#e87918"; ax.plot(px,py,color=col,linewidth=a.arrow_width,alpha=.82,solid_capstyle="round",zorder=4); j0=max(0,len(px)-max(4,len(px)//9)); ax.add_patch(FancyArrowPatch((px[j0],py[j0]),(px[-1],py[-1]),arrowstyle="-|>",mutation_scale=18,linewidth=0,color=col,alpha=.9,zorder=5))
 if not a.no_nodes:
  for lon,lat in NODES.values(): px,py=TO_VIEW.transform(((lon+180)%360)-180,lat); ax.scatter([px],[py],s=20,c="#fff",edgecolors="#374151",linewidths=.7,zorder=6)
 ax.set_xlim(a.view_xmin,a.view_xmax); ax.set_ylim(a.view_ymin,a.view_ymax); ax.set_aspect("equal"); ax.set_axis_off(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,transparent=True,bbox_inches=None,pad_inches=0); fig.savefig(out.with_suffix('.pdf'),transparent=True,bbox_inches=None,pad_inches=0); fig.savefig(out.with_suffix('.png'),dpi=300,transparent=True,bbox_inches=None,pad_inches=0); plt.close(fig); print("saved=",out)

if __name__=='__main__': main()
