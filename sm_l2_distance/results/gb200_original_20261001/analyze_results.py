import csv,json,re
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
out=Path(__file__).resolve().parent
rm=json.loads((out/'rm_topology.json').read_text())
summary=[]
for gpu in (0,1):
 run=out/f'gpu{gpu}';g=next(g for g in rm['gpus'] if g['minor']==gpu);topo={s['smid']:s for s in g['sms']}
 rows=list(csv.DictReader((run/'results/distance.csv').open()));ids=sorted({int(r['sm_a']) for r in rows});idx={s:i for i,s in enumerate(ids)};n=len(ids)
 assert n==152 and len(rows)==n*n and set(ids)==set(topo)
 D=np.zeros((n,n))
 for r in rows:D[idx[int(r['sm_a'])],idx[int(r['sm_b'])]]=float(r['mean_abs_diff'])
 assert np.isfinite(D).all() and np.array_equal(D,D.T) and (np.diag(D)==0).all()
 categories={k:[] for k in ('same_tpc','same_gpc','same_die_different_gpc','cross_die')}
 for i in range(n):
  for j in range(i+1,n):
   a,b=topo[ids[i]],topo[ids[j]]
   if a['gpcId']==b['gpcId']:
    categories['same_gpc'].append(D[i,j])
    if a['globalTpcId']==b['globalTpcId']:categories['same_tpc'].append(D[i,j])
   elif a['ugpuId']==b['ugpuId']:categories['same_die_different_gpc'].append(D[i,j])
   else:categories['cross_die'].append(D[i,j])
 stats={k:dict(pairs=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p10=float(np.percentile(v,10)),p90=float(np.percentile(v,90))) for k,v in categories.items()}
 pred=fcluster(linkage(squareform(D),method='average'),t=2,criterion='maxclust')-1;truth=np.array([topo[s]['ugpuId'] for s in ids]);agreement=max(np.mean(pred==truth),np.mean(1-pred==truth))
 means=[float(r['mean_latency']) for r in csv.DictReader((run/'results/sm_info.csv').open())]
 x=dict(gpu=gpu,sm_count=n,passes=5,first_pass_coverage='152/152' if 'Unique SMs: 152 / 152' in (run/'run.log').read_text() else 'unknown',pair_statistics=stats,two_cluster_agreement_with_rm=float(agreement),mean_latency_across_sms=float(np.mean(means)))
 (run/'analysis_rm.json').write_text(json.dumps(x,indent=2)+'\n');summary.append(x)
 print(json.dumps(x,indent=2))
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
