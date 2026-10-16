"""Paired independent-period connectivity evaluation using existing checkpoints.

Four-neighbour connected components give exactly the reachability of an
eight-neighbour A* that allows a diagonal only when both side cells are open.
Paths/distances are not estimated here. A* spot checks validate the binary
decisions. Source data and checkpoints are read-only.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from scipy.ndimage import label
import torch
import torch.nn.functional as F
import xarray as xr

BASE=Path('local-path/environment_guided_residual_unet_train/full_10var')
sys.path.insert(0,str(BASE))
import plot_full10_nep_path_buffer as h

P4='local-path/code/checkpoints/20260910_075706_envgated_pr_tos_sic_mask_year_fixed_sic_residual_2014_2023_T1_d_rmse_fixed/best_model.pth'

def writer(path,rows):
    if not rows:return
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def longest(rows,key):
    best=run=0;last=None
    for row in sorted(rows,key=lambda r:r['date']):
        date=datetime.strptime(row['date'],'%Y-%m-%d')
        if last is not None and date-last!=timedelta(days=1):run=0
        run=run+1 if row[key] else 0;best=max(best,run);last=date
    return best

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--max-days',type=int);p.add_argument('--threshold',type=float,default=.15);a=p.parse_args()
    if not 0<=a.threshold<=1:raise ValueError('Threshold must lie in [0,1]')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta=json.loads((Path(h.CACHE)/'meta.json').read_text())
    X=np.load(Path(h.CACHE)/meta.get('x_file','X.npy'),mmap_mode='r');Y=np.load(Path(h.CACHE)/meta.get('y_file','Y.npy'),mmap_mode='r')
    indices10=h.select_channels(meta['channel_names']);indices4=[indices10[i] for i in (4,7,8,9)]
    rows=[r for r in h.read_index(Path(h.CACHE)) if r['date'][:4] in ('2024','2025')]
    if a.max_days:rows=rows[:a.max_days]
    ck10=h.safe_load(Path(h.CHECKPOINT));ck4=h.safe_load(Path(P4))
    cfg=ck4['config']
    if list(cfg['INPUT_VARS'])!=['pr','tos','sic']:raise RuntimeError('Wrong four-channel model')
    model10=h.build_model(ck10,device)
    keys=('MODEL_OUTPUT_MODE','SIC_CHANNEL_INDEX','MASK_CHANNEL_INDEX','ENV_CHANNEL_INDICES','SIC_RESIDUAL_SCALE','SIC_BASELINE_INTERPOLATION','ENV_GATE_INITIAL_BIAS','ENV_GATE_SCALE','SIC_RESIDUAL_ZERO_INIT')
    previous={k:getattr(h.Config,k,None) for k in keys}
    try:
        for k in keys:
            if k in cfg:setattr(h.Config,k,cfg[k])
        model4=h.UNetWithResidual(in_channels=4,out_channels=1)
    finally:
        for k,v in previous.items():setattr(h.Config,k,v)
    model4.load_state_dict(h.checkpoint_state(ck4),strict=True);model4.to(device).eval()
    del ck10,ck4
    with xr.open_dataset(h.MASK,decode_times=False) as ds:
        x=ds.x.values;y=ds.y.values;route=ds.mask_all_shipping_seas.transpose('y','x').values>.5
    land=h.land_mask(Path(h.LAND),x,y);vx,vy=h.view_mesh(x,y)
    viewport=(vx>=-1950000)&(vx<=2800000)&(vy>=-1950000)&(vy<=-350000)
    domains={'shipping_mask':route&~land,'no_shipping_mask':viewport&~land}
    ends={k:(h.nearest_xy(x,y,-174.,61.5,v),h.nearest_xy(x,y,20.,72.,v)) for k,v in domains.items()}
    orthogonal_connectivity=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
    for domain,allowed in domains.items():
        start,goal=ends[domain]
        static_labels,_=label(allowed,structure=orthogonal_connectivity)
        if not static_labels[start] or static_labels[start]!=static_labels[goal]:
            raise RuntimeError(
                f'{domain} is disconnected even without sea ice under the no-corner-cut rule; '
                'repair the static route domain before interpreting daily blockage'
            )
    settings=dict(checkpoint4=P4,checkpoint10=h.CHECKPOINT,cache=h.CACHE,cache_meta=meta,threshold=a.threshold,comparison_rule='finite SIC <= threshold',input_indices4=indices4,input_indices10=indices10,postprocess='clamp [0,1]; no dynamic output-mask multiplication; no low-value zeroing',domain_cells={k:int(v.sum()) for k,v in domains.items()},endpoints=ends,land_file=h.LAND,route_mask_file=h.MASK,viewport=[-1950000,2800000,-1950000,-350000],inference='CUDA autocast as historical full10 evaluator',connectivity='8-neighbour A* diagonals require both orthogonal side cells open; reachability equals 4-neighbour components',date_count=len(rows),source_note='Existing September checkpoints and independent cache, not the new daily-mean reprocessing experiment')
    (out/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2,default=str))
    results=[];checks=[]
    for number,row in enumerate(rows,1):
        sid=int(row['sample_id']);date=row['date'][:10]
        i10=torch.from_numpy(np.array(X[sid,indices10],copy=True))[None].to(device)
        i4=torch.from_numpy(np.array(X[sid,indices4],copy=True))[None].to(device)
        with torch.inference_mode(),torch.amp.autocast(device_type=device.type,enabled=device.type=='cuda'):
            p10=model10(i10);p4=model4(i4)
        truth=np.asarray(Y[sid,0],dtype=np.float32)
        fields={'truth':truth,'bilinear':F.interpolate(i10[:,8:9].float(),size=truth.shape,mode='bilinear',align_corners=False).cpu().numpy()[0,0],'full10':p10.float().clamp(0,1).cpu().numpy()[0,0],'cmcc4':p4.float().clamp(0,1).cpu().numpy()[0,0]}
        for domain,allowed in domains.items():
            start,goal=ends[domain];record=dict(year=int(date[:4]),date=date,domain=domain,threshold=a.threshold)
            for name,field in fields.items():
                valid=allowed&np.isfinite(field)&(field<=a.threshold)
                if valid[start] and valid[goal]:
                    labels,_=label(valid,structure=orthogonal_connectivity);ok=bool(labels[start] and labels[start]==labels[goal]);del labels
                else:ok=False
                record[name+'_passable']=int(ok)
                if domain=='no_shipping_mask' and date in ('2024-08-19','2024-10-12') and name in ('truth','bilinear','cmcc4'):
                    astar_ok=h.astar_path(valid,start,goal,clearance_weight=0.) is not None
                    checks.append(dict(date=date,domain=domain,field=name,label_ok=int(ok),astar_ok=int(astar_ok)))
                    if astar_ok!=ok:raise RuntimeError('A* equivalence failed')
            results.append(record)
        if number%20==0 or number==len(rows):
            writer(out/'daily_comparison.csv',results)
            print(f'{number}/{len(rows)} {date} '+str(results[-1]),flush=True)
    summaries=[]
    for domain in domains:
        for year in (2024,2025):
            rr=[r for r in results if r['domain']==domain and r['year']==year]
            if not rr:continue
            truth_days=sum(r['truth_passable'] for r in rr);truth_win=longest(rr,'truth_passable')
            for method in ('bilinear','full10','cmcc4'):
                key=method+'_passable';days=sum(r[key] for r in rr);win=longest(rr,key)
                s=dict(domain=domain,year=year,method=method,evaluated_days=len(rr),truth_navigable_days=truth_days,navigable_days=days,days_error=days-truth_days,truth_longest_window=truth_win,longest_window=win,window_error=win-truth_win,false_closed=sum(r['truth_passable']==1 and r[key]==0 for r in rr),false_open=sum(r['truth_passable']==0 and r[key]==1 for r in rr),corrected_false_closed_vs_bilinear=sum(r['truth_passable']==1 and r['bilinear_passable']==0 and r[key]==1 for r in rr),corrected_false_open_vs_bilinear=sum(r['truth_passable']==0 and r['bilinear_passable']==1 and r[key]==0 for r in rr),new_false_closed_vs_bilinear=sum(r['truth_passable']==1 and r['bilinear_passable']==1 and r[key]==0 for r in rr),new_false_open_vs_bilinear=sum(r['truth_passable']==0 and r['bilinear_passable']==0 and r[key]==1 for r in rr))
                summaries.append(s)
    writer(out/'summary.csv',summaries);writer(out/'astar_checks.csv',checks)
    changes=[r for r in results if r['bilinear_passable']!=r['cmcc4_passable'] or r['bilinear_passable']!=r['full10_passable'] or r['truth_passable']!=r['bilinear_passable']]
    writer(out/'changed_or_wrong_dates.csv',changes)
    legacy=[]
    for domain in domains:
        for year in (2024,2025):
            suffix='_nomask' if domain=='no_shipping_mask' else ''
            path=Path(f'local-path/bremen_full10_bilinear_astar{suffix}_{year}.csv')
            if not path.exists():continue
            old=list(csv.DictReader(path.open(encoding='utf-8-sig')))
            new={r['date']:r for r in results if r['domain']==domain and r['year']==year}
            for r in old:
                if abs(float(r['threshold'])-a.threshold)>1e-8:continue
                if r['date'] not in new:continue
                v=new[r['date']]
                for oldkey,newkey in [('bilinear_passable','bilinear_passable'),('downscaled_passable','full10_passable'),('truth_passable','truth_passable')]:
                    if oldkey in r and int(r[oldkey])!=v[newkey]:legacy.append(dict(domain=domain,year=year,date=r['date'],field=oldkey,old=r[oldkey],new=v[newkey]))
    writer(out/'legacy_mismatches.csv',legacy)
    (out/'validation.json').write_text(json.dumps(dict(astar_checks=len(checks),legacy_mismatch_count=len(legacy)),indent=2))
    print('SUMMARY '+json.dumps(summaries),flush=True);print('LEGACY_MISMATCHES '+json.dumps(legacy),flush=True)

if __name__=='__main__':main()
