"""3-D voxel flight, occluded point returns, and explicitly synthetic life-sign cues."""
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np

DIRECTIONS=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],dtype=np.int32)
SIZE=(13,13,9)  # x,y,z; world arrays are indexed z,y,x
POINTS=64
BUDGET=360
CELL_M=.5
RANGE=5.
z=1-2*(np.arange(POINTS-6)+.5)/(POINTS-6)
az=np.arange(POINTS-6)*np.pi*(3-np.sqrt(5))
RAYS=np.concatenate([DIRECTIONS,np.stack([np.sqrt(1-z*z)*np.cos(az),np.sqrt(1-z*z)*np.sin(az),z],1)]).astype(np.float32)
SAMPLES=np.arange(.2,RANGE+.01,.2,dtype=np.float32)
OFFSETS=np.floor(RAYS[:,None,:]*SAMPLES[None,:,None]+.5).astype(np.int32)


@dataclass(frozen=True)
class Stress:
    name:str='nominal'
    dropout:float=.10
    range_sigma:float=.03
    cue_dropout:float=.25
    cue_sigma:float=.12
    ghost:float=0.
    cue_outlier:float=.05
    odom_sigma:float=0.
    miss_detection:float=.05
    false_detection:float=0.


PROFILES={
    'nominal':Stress(),
    'dropout':Stress('dropout',dropout=.65,cue_dropout=.75,miss_detection=.40),
    'multipath':Stress('multipath',range_sigma=.10,ghost=.20,cue_outlier=.40,false_detection=.35),
    'drift':Stress('drift',odom_sigma=.09),
    'combined':Stress('combined',dropout=.65,range_sigma=.10,cue_dropout=.75,cue_sigma=.25,ghost=.20,cue_outlier=.40,odom_sigma=.09,miss_detection=.40,false_detection=.35),
    'rotated':Stress('rotated'),
}


def in_bounds(p):return all(0<=int(p[i])<SIZE[i] for i in range(3))
def at(a,p):return a[int(p[2]),int(p[1]),int(p[0])]
def put(a,p,value):a[int(p[2]),int(p[1]),int(p[0])]=value

POSITIONS=[(x,y,z) for z in range(SIZE[2]) for y in range(SIZE[1]) for x in range(SIZE[0])]
def flat(p):return int(p[0])+SIZE[0]*(int(p[1])+SIZE[1]*int(p[2]))
NEIGHBORS=[tuple(flat(np.array(p)+d) for d in DIRECTIONS if in_bounds(np.array(p)+d)) for p in POSITIONS]


class Parents:
    def __init__(self,indices):self.indices=indices
    def get(self,position):
        i=self.indices[flat(position)]
        return POSITIONS[i] if i>=0 else None


def flood(free,start):
    distance=np.full(free.size,-1,np.int16);parent=np.full(free.size,-1,np.int16);start=tuple(map(int,start))
    if not in_bounds(start):return distance.reshape(free.shape),Parents(parent)
    start=flat(start);distance[start]=0;q=deque([start]);free_flat=free.ravel()
    while q:
        p=q.popleft();d=int(distance[p])
        for nxt in NEIGHBORS[p]:
            if free_flat[nxt] and distance[nxt]<0:
                distance[nxt]=d+1;parent[nxt]=p;q.append(nxt)
    return distance.reshape(free.shape),Parents(parent)


def line(a,b):
    a=np.asarray(a);b=np.asarray(b);steps=max(int(np.max(np.abs(b-a)))*3,1)
    return np.unique(np.floor(a+(b-a)*np.linspace(0,1,steps+1)[:,None]+.5).astype(int),axis=0)


def rotation(seed):
    rng=np.random.default_rng(seed);q,r=np.linalg.qr(rng.normal(size=(3,3)))
    q=q@np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q)<0:q[:,0]*=-1
    return q.astype(np.float32)


class World:
    def __init__(self,seed,profile='nominal'):
        self.seed=int(seed);self.stress=PROFILES[profile];self.base=(1,1,1)
        self.blocked,self.targets,self.decoys=self.generate()
        self.reachable=flood(~self.blocked,self.base)[0]>=0
        self.pose=np.array(self.base,dtype=int);self.estimate=self.pose.astype(float)
        self.t=0;self.energy=BUDGET;self.found=set();self.reports=set();self.collisions=0
        self.delta=np.zeros(3);self.done=False;self.trail=[self.pose.tolist()]
        self.discovery={};self.listen_count=0;self.false_reports=0

    def generate(self):
        rng=np.random.default_rng(self.seed)
        for _ in range(50):
            grid=np.zeros(SIZE[::-1],bool)
            grid[[0,-1],:,:]=True;grid[:,[0,-1],:]=True;grid[:,:,[0,-1]]=True
            grid[:,:,6]=True;grid[:,6,:]=True;grid[4,:,:]=True
            # Multiple portals, including holes in the intermediate floor.
            for lo,hi in [(1,6),(7,12)]:
                for zz in [1,5]:
                    y=int(rng.integers(lo,hi-1));grid[zz:zz+3,y:y+2,6]=False
                    x=int(rng.integers(lo,hi-1));grid[zz:zz+3,6,x:x+2]=False
                for ylo in [1,7]:
                    x=int(rng.integers(lo,hi-1));y=int(rng.integers(ylo,ylo+4))
                    grid[4,y:y+2,x:x+2]=False
            for _ in range(15):
                x,y,z=(int(rng.integers(2,SIZE[i]-3)) for i in range(3))
                w,h,d=rng.integers(1,3,3);grid[z:z+d,y:y+h,x:x+w]=True
            grid[1:3,1:3,1:3]=False
            reachable=flood(~grid,self.base)[0]>=0
            if reachable.sum()<500:continue
            candidates=np.argwhere(reachable)[:,::-1];rng.shuffle(candidates)
            targets=[]
            for p in candidates:
                if np.linalg.norm(p-self.base)>5 and all(np.linalg.norm(p-np.array(t))>4 for t in targets):targets.append(tuple(map(int,p)))
                if len(targets)==3:break
            if len(targets)==3:
                decoys=[tuple(map(int,p)) for p in candidates[-4:]]
                return grid,targets,decoys
        raise RuntimeError('No suitable 3D world')

    def visible(self,p,q):return all(not at(self.blocked,c) for c in line(p,q))

    def observe(self,listening=False):
        s=self.stress;rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.t,271]))
        cells=self.pose[None,None,:]+OFFSETS
        cells=np.clip(cells,0,np.array(SIZE)-1)
        hits=self.blocked[cells[:,:,2],cells[:,:,1],cells[:,:,0]]
        hit=hits.any(1);distance=np.where(hit,SAMPLES[np.argmax(hits,1)],RANGE)
        distance=np.clip(distance+rng.normal(0,s.range_sigma,POINTS),.2,RANGE)
        ghost=rng.random(POINTS)<s.ghost
        distance[ghost]=rng.uniform(.5,RANGE,ghost.sum());hit[ghost]=True
        valid=rng.random(POINTS)>=s.dropout
        cloud=(RAYS*distance[:,None]).astype(np.float32)
        mask=(valid&hit).astype(np.float32)
        cue=np.zeros(3,dtype=np.float32)
        if listening:
            self.listen_count+=1
            remaining=[p for i,p in enumerate(self.targets) if i not in self.found]
            if remaining:
                target=min(remaining,key=lambda p:np.linalg.norm(np.array(p)-self.pose))
                vec=np.array(target)-self.pose;dist=np.linalg.norm(vec)
                barriers=sum(at(self.blocked,p) for p in line(self.pose,target))
                if dist<=12 and rng.random()<(1-s.cue_dropout)*np.exp(-.10*barriers):
                    cue=vec/max(dist,1e-6)+rng.normal(0,s.cue_sigma,3)
                    if rng.random()<s.cue_outlier:cue=rng.normal(size=3)
                    cue=(cue/max(np.linalg.norm(cue),1e-6)).astype(np.float32)
        new=[]
        for i,target in enumerate(self.targets):
            if i not in self.reports and np.linalg.norm(self.pose-target)<=1.25 and self.visible(self.pose,target) and rng.random()>s.miss_detection:
                self.reports.add(i);self.found.add(i);new.append(i);self.discovery[str(i)]=self.t
        for j,target in enumerate(self.decoys):
            identifier=100+j
            if identifier not in self.reports and np.linalg.norm(self.pose-target)<=1.25 and rng.random()<s.false_detection:
                self.reports.add(identifier);new.append(identifier);self.false_reports+=1
        remaining=[p for i,p in enumerate(self.targets) if i not in self.found]
        # Labels below are privileged training/evaluator fields; never model inputs.
        target=min(remaining,key=lambda p:np.linalg.norm(np.array(p)-self.pose)) if remaining else self.base
        direction=np.array(target)-self.pose;direction=direction/max(np.linalg.norm(direction),1e-6)
        blocked_labels=np.array([at(self.blocked,self.pose+d) for d in DIRECTIONS],dtype=np.float32)
        return dict(points=cloud,mask=mask,cue=cue,delta=self.delta.astype(np.float32),
                    ranges=distance,valid=valid,hit=hit,estimate=self.estimate.copy(),new_reports=new,
                    labels=np.stack([DIRECTIONS@direction,blocked_labels],-1).astype(np.float32))

    def act(self,action):
        if self.done:return
        if action==7:self.done=True;return
        self.energy-=1
        old=self.pose.copy()
        if action<6:
            proposed=self.pose+DIRECTIONS[action]
            if not in_bounds(proposed) or at(self.blocked,proposed):self.collisions+=1
            else:self.pose=proposed
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.t,711]))
        self.delta=(self.pose-old).astype(float)+rng.normal(0,self.stress.odom_sigma,3)
        self.estimate+=self.delta
        self.estimate=np.clip(self.estimate,0,np.array(SIZE)-1)
        self.trail.append(self.pose.tolist());self.t+=1
        if self.energy<=0:self.done=True

    def result(self):
        home=tuple(self.pose)==self.base
        return dict(seed=self.seed,profile=self.stress.name,targets_found=len(self.found),
                    reports=len(self.reports),false_reports=self.false_reports,returned=home,
                    success=bool(home and len(self.found)==3),energy_used=BUDGET-self.energy,
                    ticks=self.t,collisions=self.collisions,listen_actions=self.listen_count,
                    discovery=self.discovery)
