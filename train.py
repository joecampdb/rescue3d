"""Same data, minibatches, optimizer updates, storage cap and final checkpoint."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse
import hashlib
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from models import ENCODERS,MEMORIES,init,trainer,metadata,save
from experiment import BASE,INPUT_KEYS


def batch(data,rng,batch_size=12,length=16):
    episode=rng.integers(len(data['lengths']),size=batch_size)
    start=np.array([rng.integers(max(int(data['lengths'][i])-length,1)) for i in episode])
    index=start[:,None]+np.arange(length)[None,:]
    obs={k:data[k][episode[:,None],index].copy() for k in INPUT_KEYS}
    # Identical full-SO(3) augmentation for both encoder families, fixed per sequence.
    for b in range(batch_size):
        q,r=np.linalg.qr(rng.normal(size=(3,3)));q=q@np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q)<0:q[:,0]*=-1
        for key in ['points','query','cue','delta']:obs[key][b]=obs[key][b]@q.T
    labels=data['labels'][episode[:,None],index]
    live=data['live'][episode[:,None],index]
    available=np.linalg.norm(obs['cue'],axis=-1)>0
    known=np.zeros(batch_size,bool);supervise=np.zeros_like(live)
    for t in range(length):
        known&=~obs['reset'][:,t].astype(bool);known|=available[:,t];supervise[:,t]=known*live[:,t]
    return {k:jnp.asarray(v) for k,v in obs.items()},jnp.asarray(labels),jnp.asarray(supervise)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=str,default=str(BASE/'outputs/train.npz'))
    p.add_argument('--out',type=str,default=str(BASE/'outputs/training'));p.add_argument('--steps',type=int,default=500)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,23,37]);p.add_argument('--require-gpu',action='store_true')
    args=p.parse_args();from pathlib import Path
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Choose a fresh training output directory')
    if args.require_gpu and jax.default_backend()!='gpu':raise RuntimeError('GPU required')
    data=dict(np.load(args.data,allow_pickle=False))
    manifest=dict(seeds=args.seeds,steps=args.steps,batch_size=12,sequence_length=16,learning_rate=.001,
                  checkpoint='fixed final update; no test or validation selection',backend=jax.default_backend(),
                  devices=[str(d) for d in jax.devices()],jax=jax.__version__,status='running',started=time.time(),
                  source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in BASE.glob('*.py')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));histories=[]
    print('Training on',jax.devices(),flush=True)
    try:
        for seed in args.seeds:
            for encoder in ENCODERS:
                for kind in MEMORIES:
                    params,width=init(seed,kind);oi,update,op=trainer(encoder,kind);state=oi(params)
                    info=metadata(params,width,kind)
                    assert info['dense_macs']<=info['mac_cap']
                    rng=np.random.default_rng(seed+10000);started=time.perf_counter();curve=[]
                    for step in range(args.steps):
                        obs,labels,supervise=batch(data,rng)
                        state,loss=update(step,state,obs,labels,supervise)
                        if step%100==0 or step==args.steps-1:
                            curve.append(dict(step=step,loss=float(loss)))
                            print(seed,encoder,kind,step,f'loss={float(loss):.4f}',flush=True)
                    save(out/f'{encoder}_{kind}_{seed}.npz',op(state))
                    histories.append(dict(seed=seed,encoder=encoder,memory=kind,**info,curve=curve,seconds=time.perf_counter()-started))
                    (out/'training.json').write_text(json.dumps(histories,indent=2))
                    del state,params,update
                    jax.clear_caches()
        manifest.update(status='completed',finished=time.time())
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));raise
    finally:(out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
