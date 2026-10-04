"""Post-hoc float64 diagnostic, not a replacement for the budgeted float32 runs."""
import json
import numpy as np
import jax
import jax.numpy as jnp
from models import load,encode,temporal
from experiment import BASE,INPUT_KEYS


def main():
    jax.config.update('jax_enable_x64',True)
    d=dict(np.load(BASE/'outputs/heldout_traces.npz',allow_pickle=False))
    obs={k:jnp.asarray(d[k][:,:48],dtype=jnp.float64) for k in INPUT_KEYS}
    rng=np.random.default_rng(771);R,r=np.linalg.qr(rng.normal(size=(3,3)));R=R@np.diag(np.sign(np.diag(r)))
    if np.linalg.det(R)<0:R[:,0]*=-1
    rotated={k:(v@R.T if k in ['points','query','cue','delta'] else v) for k,v in obs.items()}
    rows=[]
    for kind in ['ff','gru','reservoir','spiking']:
        p=load(BASE/f'outputs/training_v2/invariant_{kind}_11.npz')
        p=jax.tree_util.tree_map(lambda x:x.astype(jnp.float64),p)
        def fn(p,obs):
            x=encode(p,obs,'invariant')
            def step(s,item):
                h,reset=item;s=jnp.where(reset[...,None,None]>0,0.,s)
                state,y,_=temporal(p,s,h,kind);return state,y
            _,y=jax.lax.scan(step,jnp.zeros((x.shape[0],6,16),dtype=x.dtype),
                             (x.swapaxes(0,1),obs['reset'].swapaxes(0,1)))
            return y.swapaxes(0,1)
        f=jax.jit(fn);a=np.array(f(p,obs));b=np.array(f(p,rotated))
        diff=np.abs(a-b)[d['live'][:,:48]>0]
        rows.append(dict(model=f'invariant_{kind}',seed=11,float64_mean_abs_logit_difference=float(diff.mean()),
                         float64_max_abs_logit_difference=float(diff.max())))
    out=dict(scope='Post-hoc numerical diagnostic only: seed 11 geometric models; same trained weights promoted to float64, orthogonal rotation computed in float64. This doubles storage and is outside the matched float32 budget.',rows=rows)
    (BASE/'outputs/precision_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':main()
