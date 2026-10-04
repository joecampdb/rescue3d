"""Explain drift failures by replaying frozen G-GRU seed 11; no policy changes."""
import json
import numpy as np
from models import load
from experiment import BASE,Predictor,model_input
from world import World
from controller import Planner


def main():
    predictor=Predictor(load(BASE/'outputs/training_v2/invariant_gru_11.npz'),'invariant','gru')
    rows=[]
    for seed in range(2000,2008):
        w=World(seed,'drift');planner=Planner(w.base);predictor.reset();last=None
        while not w.done:
            obs=w.observe(last==6);planner.integrate(obs);estimate,_=predictor(model_input(obs))
            action=planner.choose(obs,w.energy,w.t,estimate);w.act(action);last=action
        rows.append(dict(**w.result(),terminal_mode=planner.mode,true_position=w.pose.tolist(),
                         estimated_position=w.estimate.tolist(),
                         position_error_cells=float(np.linalg.norm(w.estimate-w.pose)),remaining_energy=w.energy,
                         termination='budget exhausted' if w.energy==0 else planner.mode))
    output=dict(scope='Frozen-model replay diagnosis for geometric GRU, training seed 11, eight drift scenes; no adaptation',episodes=rows)
    (BASE/'outputs/failure_audit.json').write_text(json.dumps(output,indent=2))
    for r in rows:print(r['seed'],r['targets_found'],r['termination'],r['true_position'],np.round(r['estimated_position'],2).tolist(),flush=True)


if __name__=='__main__':main()
