"""Run a classical agent, or a locally trained model, in a fresh scene."""
import argparse
import json
from pathlib import Path
from experiment import BASE, Predictor, episode
from models import load, ENCODERS, MEMORIES
from world import PROFILES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoder',choices=ENCODERS,default='invariant')
    p.add_argument('--memory',choices=MEMORIES,default='gru')
    p.add_argument('--model-seed',type=int,choices=[11,23,37],default=11)
    p.add_argument('--scene',type=int,default=42)
    p.add_argument('--profile',choices=list(PROFILES),default='nominal')
    p.add_argument('--trained',action='store_true',help='Load a locally trained neural model')
    p.add_argument('--training',type=Path,default=BASE/'outputs/training_v2')
    p.add_argument('--out',type=Path,default=BASE/'outputs/local_demo.json')
    a=p.parse_args()
    if a.out.exists():raise RuntimeError('Choose a fresh --out path')
    predictor=None
    if a.trained:
        weights=a.training/f'{a.encoder}_{a.memory}_{a.model_seed}.npz'
        if not weights.exists():
            p.error(f'Missing {weights}. Run experiment.py and train.py first; see README.')
        predictor=Predictor(load(weights),a.encoder,a.memory)
    result=episode(a.scene,a.profile,predictor,record=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,separators=(',',':')))
    print(json.dumps({k:v for k,v in result.items() if k!='replay'},indent=2))
    print(f'Recorded trajectory: {a.out}')


if __name__=='__main__':main()
