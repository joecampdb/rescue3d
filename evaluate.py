"""Frozen-model, common-scene evaluation and shared-trace architecture audits."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from models import load,forward,ENCODERS,MEMORIES
from experiment import BASE,episode,Predictor,INPUT_KEYS,make_data
from world import rotation,PROFILES


def trace_audit(p,encoder,kind,data):
    # Same first 48 observations from each held-out classical trajectory.
    obs={k:data[k][:,:48] for k in INPUT_KEYS};live=data['live'][:,:48]
    labels=data['labels'][:,:48];obs_jax={k:jnp.asarray(v) for k,v in obs.items()}
    predict=jax.jit(lambda p,x:forward(p,x,encoder,kind))
    pred,rate=predict(p,obs_jax);pred=np.asarray(pred)
    reset={**obs_jax,'reset':jnp.ones_like(obs_jax['reset'])}
    no_memory=np.asarray(predict(p,reset)[0])
    R=rotation(771);rotated={k:(v@R.T if k in ['points','query','cue','delta'] else v) for k,v in obs.items()}
    rotated_pred=np.asarray(predict(p,rotated)[0])
    differences=np.abs(pred-rotated_pred)
    available=np.linalg.norm(obs['cue'],axis=-1)>0
    age=np.full(live.shape[0],100);delayed=np.zeros_like(live,bool)
    for t in range(live.shape[1]):
        age=np.where(obs['reset'][:,t]>0,100,age+1);age=np.where(available[:,t],0,age)
        delayed[:,t]=(age>=2)&(age<=15)&(live[:,t]>0)
    valid=live.astype(bool)
    if not delayed.any():raise RuntimeError('No delayed-cue audit examples')
    target_error=lambda y:float(np.mean((y[...,0][delayed]-labels[...,0][delayed])**2))
    actual=labels[...,1][valid].astype(bool);risk=pred[...,1][valid]>0
    tpr=float(np.mean(risk[actual]));tnr=float(np.mean(~risk[~actual]))
    return dict(delayed_target_mse=target_error(pred),reset_target_mse=target_error(no_memory),
                delayed_observations=int(delayed.sum()),occupancy_balanced_accuracy=(tpr+tnr)/2,
                rotation_mean_abs_logit_difference=float(differences[valid].mean()),
                rotation_max_abs_logit_difference=float(differences[valid].max()),
                spike_fraction=float(np.asarray(rate)[valid].mean()) if kind=='spiking' else None)


def summaries(records):
    rows=[]
    for name in sorted(set(r['model'] for r in records)):
        for profile in PROFILES:
            group=[r for r in records if r['model']==name and r['profile']==profile]
            if not group:continue
            row=dict(model=name,profile=profile,episodes=len(group),successes=sum(r['success'] for r in group),
                     returned=sum(r['returned'] for r in group),targets_found=sum(r['targets_found'] for r in group),
                     targets_total=3*len(group))
            for key in ['energy_used','collisions','false_reports','loop_p95_ms']:
                row[key]=float(np.mean([r[key] for r in group]))
            # Cluster by scene: do not pretend three network initializations create new scenes.
            scene_rates=[np.mean([r['success'] for r in group if r['seed']==s]) for s in sorted(set(r['seed'] for r in group))]
            rng=np.random.default_rng(12345);samples=rng.choice(scene_rates,(4000,len(scene_rates)),replace=True).mean(1)
            row['scene_bootstrap_success_95']=np.quantile(samples,[.025,.975]).tolist()
            rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,default=BASE/'outputs/training')
    p.add_argument('--out',type=Path,default=BASE/'outputs/evaluation');p.add_argument('--scenes',type=int,default=8)
    p.add_argument('--first-seed',type=int,default=2000)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'manifest.json').exists():raise RuntimeError('Use a fresh evaluation directory')
    manifest=json.loads((args.training/'manifest.json').read_text())
    if manifest['status']!='completed':raise RuntimeError('Training must complete before evaluation')
    seeds=list(range(args.first_seed,args.first_seed+args.scenes))
    ev=dict(scene_seeds=seeds,model_seeds=manifest['seeds'],profiles=list(PROFILES),
            backend=jax.default_backend(),jax=jax.__version__,python=platform.python_version(),
            status='running',started=time.time(),training=str(args.training),
            source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in BASE.glob('*.py')})
    (args.out/'manifest.json').write_text(json.dumps(ev,indent=2));records=[];audits=[]
    heldout=BASE/'outputs/heldout_traces.npz'
    if not heldout.exists():make_data(heldout,scenes=8,first_seed=1500)
    data=dict(np.load(heldout,allow_pickle=False))
    try:
        for profile in PROFILES:
            for seed in seeds:
                r=episode(seed,profile);r.update(model='classical',model_seed=None);records.append(r)
            print('classical',profile,sum(r['success'] for r in records if r['profile']==profile),'/',len(seeds),flush=True)
        (args.out/'episodes.json').write_text(json.dumps(records,indent=2))
        for model_seed in manifest['seeds']:
            for encoder in ENCODERS:
                for kind in MEMORIES:
                    name=f'{encoder}_{kind}';params=load(args.training/f'{name}_{model_seed}.npz')
                    predictor=Predictor(params,encoder,kind)
                    audit=trace_audit(params,encoder,kind,data)
                    audits.append(dict(model=name,model_seed=model_seed,**audit))
                    (args.out/'audits.json').write_text(json.dumps(audits,indent=2))
                    for profile in PROFILES:
                        group=[]
                        for seed in seeds:
                            predictor.reset();r=episode(seed,profile,predictor)
                            r.update(model=name,model_seed=model_seed);records.append(r);group.append(r)
                        print(model_seed,name,profile,f"success {sum(r['success'] for r in group)}/{len(seeds)} targets {sum(r['targets_found'] for r in group)}/{3*len(seeds)}",flush=True)
                        (args.out/'episodes.json').write_text(json.dumps(records,indent=2))
                    # One fixed scene replay per topology, first training seed only.
                    if model_seed==manifest['seeds'][0]:
                        predictor.reset();r=episode(42,'nominal',predictor,record=True)
                        (args.out/f'replay_{name}.json').write_text(json.dumps(r,separators=(',',':')))
        (args.out/'summary.json').write_text(json.dumps(summaries(records),indent=2))
        ev.update(status='completed',finished=time.time(),episodes=len(records))
    except BaseException as exc:
        ev.update(status='failed',error=repr(exc));raise
    finally:(args.out/'manifest.json').write_text(json.dumps(ev,indent=2))


if __name__=='__main__':main()
