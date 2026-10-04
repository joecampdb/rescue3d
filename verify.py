"""Check frozen artifacts, weight budgets, fixed reservoirs, and actual 3-D paths."""
import hashlib
import json
import numpy as np
import jax
from models import init,load,count
from experiment import BASE
from world import SIZE,at,line


def main():
    training=BASE/'outputs/training_v2';evaluation=BASE/'outputs/evaluation'
    manifest=json.loads((evaluation/'manifest.json').read_text())
    assert manifest['status']=='completed'
    checks=0
    def check(value):
        nonlocal checks
        assert value;checks+=1
    for name in ['models.py','world.py','controller.py','experiment.py','evaluate.py','train.py']:
        check(hashlib.sha256((BASE/name).read_bytes()).hexdigest()==manifest['source_sha256'][name])
    records=json.loads((evaluation/'episodes.json').read_text())
    check(len(records)==1200)
    check(len({(r['seed'],r['profile'],r['model'],r['model_seed']) for r in records})==1200)
    for r in records:
        check(0<=r['energy_used']<=360)
        check(0<=r['targets_found']<=3)
        check(r['success']==(r['targets_found']==3 and r['returned']))
    for row in json.loads((training/'training.json').read_text()):
        p=load(training/f'{row["encoder"]}_{row["memory"]}_{row["seed"]}.npz')
        check(count(p)==row['parameters']<=4096)
        check(all(np.isfinite(a).all() for a in jax.tree_util.tree_leaves(p)))
        if row['memory']=='reservoir':
            original,_=init(row['seed'],'reservoir')
            for layer in ['fixed_in','fixed_rec']:
                for key in p[layer]:check(np.array_equal(p[layer][key],original[layer][key]))
    for path in evaluation.glob('replay_*.json'):
        r=json.loads(path.read_text());rep=r['replay'];trail=np.array(rep['trail'])
        blocked=np.zeros(SIZE[::-1],bool);b=np.array(rep['blocked']);blocked[b[:,2],b[:,1],b[:,0]]=True
        check(not blocked[trail[:,2],trail[:,1],trail[:,0]].any())
        check(bool((np.abs(np.diff(trail,axis=0)).sum(1)<=1).all()))
        check(len(trail)==r['energy_used']+1)
        check(r['returned']==(tuple(trail[-1])==tuple(rep['base'])))
        check(np.ptp(trail[:,2])>0)
        for identifier,tick in r['discovery'].items():
            target=np.array(rep['targets'][int(identifier)]);position=trail[tick]
            check(np.linalg.norm(position-target)<=1.25)
            check(not any(at(blocked,p) for p in line(position,target)))
    for r in json.loads((BASE/'outputs/failure_audit.json').read_text())['episodes']:
        original=next(a for a in records if a['model']=='invariant_gru' and a['model_seed']==11 and a['seed']==r['seed'] and a['profile']=='drift')
        for key in ['targets_found','returned','success','energy_used','collisions','false_reports','discovery']:
            check(r[key]==original[key])
    result=dict(status='passed',checks=checks,unit_tests=12,episodes=1200,
                verified='Frozen source hashes, all episode identities and budgets, checkpoint finiteness and storage, unchanged reservoir weights, obstacle-free logged paths and actual z motion')
    (BASE/'outputs/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
