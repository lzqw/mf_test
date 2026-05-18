import argparse, pickle, csv
from pathlib import Path
import numpy as np

def load(path):
    try:
        return pickle.load(open(path,'rb'))
    except Exception:
        return []

def first_last(rows,key):
    if not rows: return (np.nan,np.nan)
    return (rows[0].get(key,np.nan), rows[-1].get(key,np.nan))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='logs/safetygym_long')
    ap.add_argument('--out',default='safetygym_long_summary.txt')
    args=ap.parse_args()
    root=Path(args.root)
    results=[]
    for eval_pkl in root.rglob('eval_metrics.pkl'):
        run_dir=eval_pkl.parent
        train_pkl=run_dir/'train_metrics.pkl'
        ev=load(eval_pkl)
        tr=load(train_pkl) if train_pkl.exists() else []
        row={'path':str(run_dir), 'num_eval':len(ev)}
        for k in ['return_','cost_return','success_rate','goal_dist_min','goal_dist_final','goal_dist_mean','goal_met_any_rate','goal_reached_by_dist_any_rate','raw_action_norm','APR','FAR',
                  'raw_return_','raw_success_rate','raw_cost_return','raw_goal_dist_min','raw_goal_dist_final',
                  'filtered_return_','filtered_success_rate','filtered_cost_return','filtered_goal_dist_min','filtered_goal_dist_final','filtered_FAR','filtered_APR',
                  'eval_stopped_on_goal_rate','hit_step_mean']:
            a,b=first_last(ev,k); row[f'first_{k}']=a; row[f'last_{k}']=b
        if tr:
            last=tr[-1]
            for k in ['q1_loss','q2_loss','qp_loss','vp_loss','policy_loss','alpha','g_loss']:
                row[f'train_last_{k}']=last.get(k,np.nan)
        results.append(row)
    txt=Path(args.out)
    lines=[]
    for r in sorted(results,key=lambda x:x['path']):
        lines.append(f"path: {r['path']}")
        lines.append(f"  num eval: {r['num_eval']}")
        for k in ['return_','cost_return','success_rate','goal_dist_min','goal_dist_final','goal_dist_mean','goal_met_any_rate','goal_reached_by_dist_any_rate','raw_action_norm','APR','FAR',
                  'raw_return_','raw_success_rate','raw_cost_return','raw_goal_dist_min','raw_goal_dist_final',
                  'filtered_return_','filtered_success_rate','filtered_cost_return','filtered_goal_dist_min','filtered_goal_dist_final','filtered_FAR','filtered_APR',
                  'eval_stopped_on_goal_rate','hit_step_mean']:
            lines.append(f"  {k}: {r.get('first_'+k,np.nan)} -> {r.get('last_'+k,np.nan)}")
        lines.append('')
    txt.write_text('\n'.join(lines))
    csv_path=txt.parent/'summary.csv'
    if results:
        keys=sorted({k for r in results for k in r.keys()})
        with csv_path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); [w.writerow(r) for r in results]

if __name__=='__main__':
    main()
