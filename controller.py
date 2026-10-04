"""Shared mapping and frontier navigation; neural modules estimate directional utility."""
import numpy as np
from world import SIZE,DIRECTIONS,RAYS,at,put,flood,in_bounds,line


class Planner:
    def __init__(self,base):
        self.base=tuple(base);self.odds=np.zeros(SIZE[::-1],np.float32)
        self.visits=np.zeros_like(self.odds);self.searched=np.zeros_like(self.odds,bool)
        self.returning=False;self.goal=None;self.cue_memory=np.zeros(3);self.last_listen=-99
        self.reports=set();self.mode='explore';self.last_action=6

    def integrate(self,obs):
        pos=np.floor(obs['estimate']+.5).astype(int)
        # Vectorized ray integration; one free vote per ray and voxel.
        samples=np.arange(0,5.21,.2)
        cells=np.floor(obs['estimate']+RAYS[:,None,:]*samples[None,:,None]+.5).astype(int)
        changed=np.concatenate([np.ones((64,1),bool),np.any(np.diff(cells,axis=1)!=0,axis=-1)],axis=1)
        enddist=obs['ranges']+.12*obs['hit']
        valid=obs['valid'][:,None] & (samples[None,:]<=enddist[:,None]) & changed
        valid &= ((cells>=0)&(cells<np.array(SIZE))).all(-1)
        selected=cells[valid];np.add.at(self.odds,(selected[:,2],selected[:,1],selected[:,0]),-.35)
        ends=np.floor(obs['estimate']+RAYS*enddist[:,None]+.5).astype(int)
        hit=obs['valid']&obs['hit']&((ends>=0)&(ends<np.array(SIZE))).all(-1)
        ends=ends[hit];np.add.at(self.odds,(ends[:,2],ends[:,1],ends[:,0]),1.5)
        np.clip(self.odds,-5,5,out=self.odds);put(self.odds,pos,-5);put(self.visits,pos,at(self.visits,pos)+1)
        for d in np.vstack([np.zeros(3,int),DIRECTIONS]):
            p=pos+d
            if in_bounds(p) and at(self.odds,p)<-.2:put(self.searched,p,True)
        self.reports.update(obs['new_reports'])
        if obs['new_reports']:self.cue_memory*=0
        if np.linalg.norm(obs['cue'])>0:self.cue_memory=.5*self.cue_memory+.5*obs['cue']
        else:self.cue_memory*=.98

    def choose(self,obs,energy,tick,estimate=None):
        pos=tuple(np.floor(obs['estimate']+.5).astype(int));free=self.odds<-.2
        distance,parent=flood(free,pos);home=int(at(distance,self.base))
        if energy<max(home,20)+15 or len(self.reports)>=3:self.returning=True
        if self.returning:
            self.mode='return';goal=self.base
            if pos==goal:return 7
            if home<0:
                # No invented route through unseen space: record an incomplete return.
                self.mode='no known home route';return 7
        else:
            if tick-self.last_listen>=8:
                self.last_listen=tick;self.mode='listen';return 6
            unknown=np.abs(self.odds)<=.2
            adj=np.zeros_like(self.odds)
            for axis in range(3):
                adj+=np.roll(unknown,1,axis).astype(float)+np.roll(unknown,-1,axis).astype(float)
            candidates=np.argwhere(free & (distance>0) & ((adj>0)|~self.searched))[:,::-1]
            if not len(candidates):self.returning=True;return self.choose(obs,energy,tick,estimate)
            d=distance[candidates[:,2],candidates[:,1],candidates[:,0]]
            delta=candidates-np.array(pos);norm=np.maximum(np.linalg.norm(delta,axis=1),1.)
            signal=self.cue_memory if estimate is None else DIRECTIONS.T@estimate[:,0]/2
            score=(1.+adj[candidates[:,2],candidates[:,1],candidates[:,0]]+2*(~self.searched[candidates[:,2],candidates[:,1],candidates[:,0]]))/(d+3.)
            score-=.2*self.visits[candidates[:,2],candidates[:,1],candidates[:,0]]
            score+=1.2*(delta@signal)/norm
            if estimate is not None:
                primary=np.argmax(delta@DIRECTIONS.T,axis=1)
                score-=.15*estimate[primary,1]
            goal=tuple(candidates[int(np.argmax(score))]);self.mode='search'
        self.goal=goal;current=goal
        for _ in range(np.prod(SIZE)):
            prev=parent.get(current)
            if prev is None:return 6
            if prev==pos:
                offset=np.array(current)-pos
                return int(np.where((DIRECTIONS==offset).all(1))[0][0])
            current=prev
        raise RuntimeError('Invalid parent chain')
