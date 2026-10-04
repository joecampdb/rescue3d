"""Budget-matched point-set encoders crossed with four temporal cores in JAX."""
import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

ENCODERS=['xyz','invariant']
MEMORIES=['ff','gru','reservoir','spiking']
WEIGHT_CAP=4096
STATE_CAP=96
MAC_CAP=2_000_000


def init(seed,kind):
    def build(width):
        rng=np.random.default_rng(seed)
        def dense(a,b,scale=1.):
            return dict(w=jnp.asarray(rng.normal(0,scale/np.sqrt(a),(a,b)),dtype=jnp.float32),b=jnp.zeros(b))
        p=dict(point1=dense(6,width),point2=dense(width,8),embed=dense(24,16),risk=dense(16,1,.2))
        if kind=='gru':p.update(gates=dense(32,32),candidate=dense(32,16))
        elif kind=='ff':p['core']=dense(16,16)
        elif kind=='reservoir':
            p['fixed_in']=dense(16,16)
            w=rng.normal(size=(16,16));w*=.8/max(abs(np.linalg.eigvals(w)))
            p['fixed_rec']=dict(w=jnp.asarray(w,dtype=jnp.float32))
        elif kind=='spiking':p.update(current=dense(16,8),recurrent=dense(8,8,.2))
        else:raise ValueError(kind)
        p['target']=dense(24 if kind=='spiking' else 32,1,.2)
        return p
    # Each additional encoder channel contributes exactly 6+1+8 weights.
    # Compute the width arithmetically; do not allocate 274 discarded GPU models.
    width=1+(WEIGHT_CAP-count(build(1)))//15
    return build(width),width


def count(p):return sum(x.size for x in jax.tree_util.tree_leaves(p))
def dense(p,x):return x@p['w']+p.get('b',0.)


def point_features(points,mask,query,encoder):
    # points: [..., P,3], query: [...,6,3]; translation removed at sensor origin.
    points=points/5.
    pp=jnp.broadcast_to(points[...,None,:,:],(*points.shape[:-2],6,points.shape[-2],3))
    qq=jnp.broadcast_to(query[..., :,None,:],pp.shape)
    if encoder=='xyz':return jnp.concatenate([pp,qq],-1)
    radius=jnp.linalg.norm(pp,axis=-1)
    dot=jnp.sum(pp*qq,-1)
    centroid=jnp.sum(points*mask[...,None],axis=-2)/jnp.maximum(mask.sum(-1,keepdims=True),1.)
    cm=centroid[...,None,None,:]
    return jnp.stack([radius,dot,jnp.sqrt(jnp.maximum(radius**2-dot**2,1e-8)),
                      jnp.sum(pp*cm,-1),jnp.linalg.norm(pp-cm,axis=-1),dot**2],-1)


def encode(p,obs,encoder):
    f=point_features(obs['points'],obs['mask'],obs['query'],encoder)
    h=jnp.tanh(dense(p['point2'],jax.nn.relu(dense(p['point1'],f))))
    mask=obs['mask'][...,None,:,None]
    mean=(h*mask).sum(-2)/jnp.maximum(mask.sum(-2),1.)
    mx=jnp.max(jnp.where(mask>0,h,-1e4),-2)
    mx=jnp.where(obs['mask'].sum(-1)[...,None,None]>0,mx,0.)
    cue=obs['cue'];delta=obs['delta'];q=obs['query']
    if encoder=='xyz':
        v=jnp.concatenate([cue,delta],-1)
        vec=jnp.broadcast_to(v[...,None,:],(*v.shape[:-1],6,6))
    else:
        cd=jnp.sum(cue[...,None,:]*q,-1);dd=jnp.sum(delta[...,None,:]*q,-1)
        cn=jnp.broadcast_to(jnp.linalg.norm(cue,axis=-1)[...,None],cd.shape)
        dn=jnp.broadcast_to(jnp.linalg.norm(delta,axis=-1)[...,None],dd.shape)
        vec=jnp.stack([cd,cn,jnp.sqrt(jnp.maximum(cn*cn-cd*cd,1e-8)),dd,dn,jnp.sqrt(jnp.maximum(dn*dn-dd*dd,1e-8))],-1)
    flags=jnp.stack([jnp.linalg.norm(cue,axis=-1),obs['mask'].mean(-1)],-1)
    flags=jnp.broadcast_to(flags[...,None,:],(*flags.shape[:-1],6,2))
    return jnp.tanh(dense(p['embed'],jnp.concatenate([mean,mx,vec,flags],-1)))


@jax.custom_jvp
def spike(v):return (v>0).astype(v.dtype)


@spike.defjvp
def spike_jvp(primals,tangents):
    v,=primals;dv,=tangents
    return spike(v),dv/(1+5*jnp.abs(v))**2


def initial_state(batch=()):return jnp.zeros((*batch,6,16),dtype=jnp.float32)


def temporal(p,state,x,kind):
    activity=jnp.zeros(x.shape[:-2])
    if kind=='ff':h=jnp.tanh(dense(p['core'],x));new=jnp.zeros_like(state)
    elif kind=='gru':
        z,r=jnp.split(jax.nn.sigmoid(dense(p['gates'],jnp.concatenate([x,state],-1))),2,-1)
        candidate=jnp.tanh(dense(p['candidate'],jnp.concatenate([x,r*state],-1)))
        h=(1-z)*state+z*candidate;new=h
    elif kind=='reservoir':
        # Trainable encoder/readout, fixed input/recurrent reservoir matrices.
        h=.65*state+.35*jnp.tanh(dense(p['fixed_in'],x)+dense(p['fixed_rec'],state));new=h
    else:
        membrane,previous=jnp.split(state,2,-1);current=dense(p['current'],x)
        spikes=[]
        for _ in range(4):
            voltage=.90*membrane+current+dense(p['recurrent'],previous)
            previous=spike(voltage-.5)
            membrane=voltage-.5*jax.lax.stop_gradient(previous)
            spikes.append(previous)
        h=jnp.mean(jnp.stack(spikes),0);new=jnp.concatenate([membrane,previous],-1)
        activity=jnp.mean(jnp.stack(spikes),axis=(0,-1,-2))
    output=jnp.concatenate([jnp.tanh(dense(p['target'],jnp.concatenate([x,h],-1))),dense(p['risk'],x)],-1)
    return new,output,activity


def online(p,state,obs,encoder,kind):
    state=jnp.where(obs['reset'][...,None,None]>0,0.,state)
    return temporal(p,state,encode(p,obs,encoder),kind)


def forward(p,obs,encoder,kind):
    x=encode(p,obs,encoder)
    def step(state,data):
        h,reset=data;state=jnp.where(reset[...,None,None]>0,0.,state)
        new,y,rate=temporal(p,state,h,kind)
        return new,(y,rate)
    _,(y,rates)=jax.lax.scan(step,initial_state((x.shape[0],)),(x.swapaxes(0,1),obs['reset'].swapaxes(0,1)))
    return y.swapaxes(0,1),rates.swapaxes(0,1)


def trainer(encoder,kind):
    oi,ou,op=optimizers.adam(.001)
    def loss(p,obs,label,supervise):
        y,_=forward(p,obs,encoder,kind)
        target=(((y[...,0]-label[...,0])**2)*supervise[...,None]).sum()/jnp.maximum(supervise.sum()*6,1.)
        logits=y[...,1];actual=label[...,1]
        risk=jnp.mean((1+2*actual)*(jax.nn.softplus(logits)-actual*logits))
        return target+.3*risk
    @jax.jit
    def update(i,state,obs,label,supervise):
        value,g=jax.value_and_grad(loss)(op(state),obs,label,supervise)
        if kind=='reservoir':
            for name in ['fixed_in','fixed_rec']:g[name]=jax.tree_util.tree_map(jnp.zeros_like,g[name])
        norm=jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree_util.tree_leaves(g)))
        g=jax.tree_util.tree_map(lambda x:x*jnp.minimum(1.,1/(norm+1e-8)),g)
        return ou(i,g,state),value
    return oi,update,op


def metadata(p,width,kind):
    fixed=sum(x.size for name,v in p.items() if name.startswith('fixed_') for x in v.values())
    # Dense matrix-vector MACs, excluding nonlinearities and invariant-feature arithmetic.
    point=6*64*(6*width+width*8)
    common=6*(24*16+16)
    core={'ff':6*16*16,'gru':6*3*32*16,'reservoir':6*2*16*16,'spiking':6*(16*8+4*8*8)}[kind]
    read=6*(24 if kind=='spiking' else 32)
    return dict(parameters=count(p),trainable_parameters=count(p)-fixed,fixed_parameters=fixed,
                weight_bytes=4*count(p),persistent_state_floats=0 if kind=='ff' else STATE_CAP,
                encoder_width=width,dense_macs=point+common+core+read,
                weight_cap=WEIGHT_CAP,state_cap=STATE_CAP,mac_cap=MAC_CAP)


def save(path,p):np.savez(path,**{f'{k}__{n}':np.asarray(v) for k,values in p.items() for n,v in values.items()})
def load(path):
    p={}
    with np.load(path,allow_pickle=False) as f:
        for key in f.files:
            layer,name=key.split('__');p.setdefault(layer,{})[name]=jnp.asarray(f[key])
    return p
