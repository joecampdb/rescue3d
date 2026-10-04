"""Training data and closed-loop execution. Oracle labels are excluded from inputs."""
import json
from pathlib import Path
import time
from functools import lru_cache
import numpy as np
from world import World,DIRECTIONS,BUDGET,rotation,SIZE
from controller import Planner

BASE=Path(__file__).resolve().parent
INPUT_KEYS=['points','mask','cue','delta','query','reset']


def model_input(obs,R=None):
    R=np.eye(3,dtype=np.float32) if R is None else R
    return dict(points=np.asarray(obs['points']@R.T,np.float32),mask=obs['mask'],
                cue=np.asarray(obs['cue']@R.T,np.float32),delta=np.asarray(obs['delta']@R.T,np.float32),
                query=np.asarray(DIRECTIONS@R.T,np.float32),reset=np.float32(bool(obs['new_reports'])))


def episode(seed,profile='nominal',predict=None,record=False,collect=False):
    w=World(seed,profile);planner=Planner(w.base);last=None;frames=[];data=[];times=[];spikes=[]
    R=rotation(seed+88123) if profile=='rotated' else np.eye(3,dtype=np.float32)
    while not w.done:
        start=time.perf_counter();obs=w.observe(last==6);planner.integrate(obs)
        inputs=model_input(obs,R);estimate=None
        if predict is not None:
            estimate,rate=predict(inputs);spikes.append(rate)
        action=planner.choose(obs,w.energy,w.t,estimate)
        if collect:
            data.append(dict(**inputs,labels=obs['labels'],live=np.float32(1)))
        if record and w.t%4==0:
            frames.append(dict(t=w.t,pose=w.pose.tolist(),estimate=w.estimate.tolist(),
                              points=(obs['points']+w.pose)[obs['mask']>0].tolist(),
                              known=np.argwhere(planner.odds<-.2)[:,::-1].tolist(),
                              found=sorted(w.found),reports=sorted(w.reports),energy=w.energy,
                              mode=planner.mode,action=action,cue=obs['cue'].tolist()))
        w.act(action);last=action;times.append((time.perf_counter()-start)*1000)
    result=w.result();result.update(loop_p95_ms=float(np.percentile(times,95)),loop_median_ms=float(np.median(times)),
                                   spike_fraction=float(np.mean(spikes)) if spikes else None)
    if collect:result['training_data']=data
    if record:result['replay']=dict(blocked=np.argwhere(w.blocked)[:,::-1].tolist(),targets=w.targets,
                                   base=w.base,shape=SIZE,trail=w.trail,frames=frames,result=result.copy())
    return result


@lru_cache(maxsize=16)
def compiled_step(encoder,kind):
    import jax
    from models import online
    return jax.jit(lambda p,s,x:online(p,s,x,encoder,kind))


class Predictor:
    def __init__(self,p,encoder,kind):
        import jax
        from models import online,initial_state
        self.p=p;self.kind=kind;self.state=initial_state()
        self.step=compiled_step(encoder,kind)
        dummy=dict(points=np.zeros((64,3),np.float32),mask=np.zeros(64,np.float32),cue=np.zeros(3,np.float32),
                   delta=np.zeros(3,np.float32),query=DIRECTIONS.astype(np.float32),reset=np.float32(0))
        _,y,_=self.step(p,self.state,dummy);y.block_until_ready()

    def reset(self):
        from models import initial_state
        self.state=initial_state()

    def __call__(self,obs):
        self.state,y,rate=self.step(self.p,self.state,obs)
        y=np.array(y);y[:,1]=1/(1+np.exp(-y[:,1]))
        return y,float(rate)


def make_data(out,scenes=64,first_seed=200):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise RuntimeError('Dataset already exists')
    arrays={k:[] for k in INPUT_KEYS+['labels','live']};lengths=[]
    for seed in range(first_seed,first_seed+scenes):
        result=episode(seed,collect=True);rows=result.pop('training_data');lengths.append(len(rows))
        for key in arrays:
            a=np.stack([r[key] for r in rows]);pad=np.zeros((BUDGET+1-len(a),*a.shape[1:]),a.dtype)
            arrays[key].append(np.concatenate([a,pad]))
        if seed%8==0:print(f'dataset scene {seed}, {len(rows)} observations',flush=True)
    np.savez_compressed(out,**{k:np.stack(v) for k,v in arrays.items()},lengths=lengths)
    out.with_suffix('.json').write_text(json.dumps(dict(seeds=list(range(first_seed,first_seed+scenes)),
        observations=sum(lengths),profile='nominal',collection_policy='classical observed-map/cue planner',
        targets='privileged direction to nearest unreported target and adjacent-voxel occupancy; training only'),indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=BASE/'outputs/train.npz');p.add_argument('--scenes',type=int,default=64)
    args=p.parse_args();make_data(args.data,args.scenes)
