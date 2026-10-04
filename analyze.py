"""Generate an evidence-backed comparison without selecting or retraining models."""
import json
from pathlib import Path
import numpy as np
from experiment import BASE
from render import IDS,MODELS


def main():
    root=BASE/'outputs/evaluation'
    rows=json.loads((root/'summary.json').read_text());records=json.loads((root/'episodes.json').read_text())
    audits=json.loads((root/'audits.json').read_text());training=json.loads((BASE/'outputs/training_v2/training.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='completed'
    profiles=['nominal','dropout','multipath','drift','combined','rotated']
    get=lambda m,p:next(r for r in rows if r['model']==m and r['profile']==p)
    lines=['RESCUE3D: MATCHED-STORAGE PERCEPTION AND MEMORY COMPARISON',
           '3 October 2026 | Python/JAX | 24 trained networks | 1,200 evaluation episodes','',
           'WHAT WAS ACTUALLY COMPARED',
           'Two point-set encoders crossed with four temporal cores, inside one common',
           'mapping/search/return controller. This is a modular supervised-learning',
           'comparison, not end-to-end RL, learned SLAM, or a neuromorphic hardware test.',
           'The geometric (G) model uses invariant distances and dot products; it is not',
           'a reproduction of EGNN. C is a conventional XYZ point-set encoder. Both',
           'receive the same full-SO(3) training augmentation.','',
           'COMPLETE MISSION SUCCESS: all three REAL targets and return to TRUE base',
           'Each learned entry pools 8 layouts x 3 training seeds (24 episodes).',
           'Classical has 8 episodes: no artificial duplication over training seeds.','',
           f'{"Model":12s}'+''.join(f'{p:>13s}' for p in profiles)]
    for m in MODELS:
        lines.append(f'{IDS.get(m,"Classical"):12s}'+''.join(f'{get(m,p)["successes"]:>9}/{get(m,p)["episodes"]:<3}' for p in profiles))
    lines+=['','ACTUAL TARGETS FOUND (misses and false reports kept separate)','',f'{"Model":12s}'+''.join(f'{p:>13s}' for p in profiles)]
    for m in MODELS:
        lines.append(f'{IDS.get(m,"Classical"):12s}'+''.join(f'{get(m,p)["targets_found"]:>9}/{get(m,p)["targets_total"]:<3}' for p in profiles))
    lines+=['','RETURN RATE','',f'{"Model":12s}'+''.join(f'{p:>13s}' for p in profiles)]
    for m in MODELS:
        lines.append(f'{IDS.get(m,"Classical"):12s}'+''.join(f'{get(m,p)["returned"]:>9}/{get(m,p)["episodes"]:<3}' for p in profiles))
    lines+=['','Every model in nominal conditions: targets, return, action use, and collisions.']
    for m in MODELS:
        r=get(m,'nominal')
        lines.append(f'{IDS.get(m,"Classical"):10s}: {r["targets_found"]}/{r["targets_total"]} targets; {r["returned"]}/{r["episodes"]} home; mean {r["energy_used"]:.1f} actions; {r["collisions"]:.2f} collision attempts.')
    lines+=['','MATCHED-TRACE DIAGNOSTICS (mean across 3 training seeds)',
           'Direction MSE is measured 2-15 observations after a cue. Lower is better.',
           'Reset MSE zeroes neural memory each observation but leaves current input.',
           'The occupancy head tests only six adjacent voxels, not semantic recognition.',
           f'{"Model":10s} {"MSE":>8s} {"Reset MSE":>10s} {"Occ. bal.acc.":>14s} {"Rotation max*":>14s}']
    diagnostic=[]
    for m in MODELS[1:]:
        a=[r for r in audits if r['model']==m]
        avg={k:float(np.mean([r[k] for r in a])) for k in ['delayed_target_mse','reset_target_mse','occupancy_balanced_accuracy','rotation_max_abs_logit_difference']}
        diagnostic.append(dict(model=m,**avg))
        lines.append(f'{IDS[m]:10s} {avg["delayed_target_mse"]:8.4f} {avg["reset_target_mse"]:10.4f} {avg["occupancy_balanced_accuracy"]:14.3%} {avg["rotation_max_abs_logit_difference"]:14.3g}')
    lines+=['* Mean of each seed\'s largest output-logit difference under joint rotation.',
            'The raw occupancy logit is unbounded; logit change is not a success-rate change.','']
    precision=json.loads((BASE/'outputs/precision_audit.json').read_text())
    lines+=['FINITE-PRECISION AUDIT',
            'The invariant feature construction has exact algebraic symmetry, but float32',
            'rounding and spiking thresholds can produce nonzero numerical discrepancies.',
            'A separate seed-11 float64 diagnostic, using the same trained weights and a',
            'rotation computed in float64, gave the following maximum logit differences:']
    for r in precision['rows']:lines.append(f'{IDS[r["model"]]:10s}: {r["float64_max_abs_logit_difference"]:.3g}')
    lines+=['This doubles storage and is OUTSIDE the matched resource budget. It diagnoses',
            'numerics without changing any scored float32 checkpoint or mission result.','']
    lines+=['PAIRED GEOMETRIC-MINUS-CONVENTIONAL DIFFERENCES',
            'Metric: additional real targets per episode; nominal condition.',
            'Scene-cluster percentile bootstrap, 4,000 draws; exploratory intervals.']
    paired=[]
    for kind in ['ff','gru','reservoir','spiking']:
        deltas=[]
        for scene in manifest['scene_seeds']:
            subset=[r for r in records if r['seed']==scene and r['profile']=='nominal']
            a=np.mean([r['targets_found'] for r in subset if r['model']==f'invariant_{kind}'])
            b=np.mean([r['targets_found'] for r in subset if r['model']==f'xyz_{kind}'])
            deltas.append(a-b)
        rng=np.random.default_rng(231);samples=rng.choice(deltas,(4000,len(deltas)),replace=True).mean(1)
        ci=np.quantile(samples,[.025,.975]);paired.append(dict(memory=kind,mean=float(np.mean(deltas)),ci95=ci.tolist()))
        lines.append(f'{kind:10s}: {np.mean(deltas):+.3f} targets; interval [{ci[0]:+.3f}, {ci[1]:+.3f}]')
    lines+=['','RESOURCE ACCOUNTING',
            'Shared caps: 4,096 stored float32 weights, 96 temporal floats, 2 M dense MACs,',
            '64 point slots, 360 actions. All models: 500 updates x 12 sequences x 16 steps.',
            'Actual stored parameter counts differ by only 0.27%; MAC counts are unequal.']
    for kind in ['ff','gru','reservoir','spiking']:
        row=next(r for r in training if r['memory']==kind)
        lines.append(f'{kind:10s}: {row["parameters"]} stored / {row["trainable_parameters"]} learned parameters; {row["persistent_state_floats"]} temporal floats; {row["dense_macs"]:,} dense MACs.')
    lines+=['The common map/search arrays and pathfinding workspace are additional memory',
            'for every agent. FF is memoryless only in its neural core. Encoder widths',
            'vary to meet the total weight budget, so this does not isolate temporal',
            'mechanism while holding every perception layer fixed. The reservoir trains',
            'its encoder/readouts but not its input or recurrent reservoir weights.',
            'Dense MACs omit activation, geometric-feature, data-movement and planning costs.',
            'Transient activation/workspace memory is not matched: the largest single',
            'point-MLP activation is approximately 204-344 KiB. Total backend peak RAM',
            'was not measured. Equal storage/state caps do not imply identical hardware cost.','',
            'RUNTIME',
            'Training: RTX 4070, WSL JAX 0.6.2. Evaluation: native Windows JAX 0.7.1 CPU.',
            'Values below average per-episode p95 loop times in nominal scenes; they are',
            'not pooled p95 measurements, inference-only latency, or hardware power.']
    for m in MODELS:lines.append(f'{IDS.get(m,"Classical"):10s}: {get(m,"nominal")["loop_p95_ms"]:.2f} ms')
    lines+=['','SENSING FAILURE SUMMARY']
    for profile in profiles:
        group=[r for r in records if r['profile']==profile]
        lines.append(f'{profile:10s}: {sum(r["success"] for r in group)}/{len(group)} complete; {sum(r["returned"] for r in group)}/{len(group)} home; {sum(r["false_reports"] for r in group)} false reports.')
    failures=json.loads((BASE/'outputs/failure_audit.json').read_text())['episodes']
    lines+=['','DRIFT FAILURE DIAGNOSIS: frozen G-GRU, training seed 11']
    for r in failures:
        lines.append(f'Scene {r["seed"]}: {r["targets_found"]}/3 targets, {r["remaining_energy"]} units left, {r["position_error_cells"]:.2f}-cell position error; terminal state: {r["termination"]}.')
    lines+=['Seven of these eight episodes ended with no route to base in the corrupted',
            'estimated map; one stopped at the estimated base while physically elsewhere.',
            'This diagnoses a shared mapping/localization failure rather than proving',
            'that any particular memory architecture cannot solve rescue navigation.']
    # Rank only to describe this finite pilot, not select a winning checkpoint.
    ranked=sorted(MODELS[1:],key=lambda m:get(m,'nominal')['successes'],reverse=True)
    best_count=get(ranked[0],'nominal')['successes'];best=[IDS[m] for m in ranked if get(m,'nominal')['successes']==best_count]
    lines+=['','INTERPRETATION',
            f'Highest nominal learned-model completion count: {", ".join(best)} ({best_count}/24).',
            f'Classical nominal control: {get("classical","nominal")["successes"]}/8.',
            'A rank in eight procedural layouts is not a general architectural ranking.',
            'Read the target and return tables separately: a robot may detect targets',
            'and still fail the mission because its estimated route does not reach base.',
            'Localization and false-report handling are shared system bottlenecks; a',
            'rotation-consistent frontend does not by itself repair either one.',
            'Use the memory-reset audit to assess a neural state contribution on identical',
            'histories, rather than attributing all autonomous-route differences to memory.',
            'High adjacent-occupancy accuracy is a narrow sensor task with cardinal rays.',
            'It is not evidence of recognizing victims or semantic hierarchies in 3-D.',
            'The SNN is an implemented spiking temporal core, but any hardware-energy',
            'advantage remains unmeasured. Rate and MAC statistics do not establish it.','',
            'STATISTICAL AND PHYSICAL LIMITS',
            'Only eight distinct evaluation layouts. Three training seeds quantify some',
            'optimization variability, not 24 independent environments. Bootstrap scene',
            'intervals are descriptive and unstable at this sample size; no multiple-',
            'comparison significance claim or population-level superiority is made.',
            'Both perception families use the same rotation augmentation. Models train',
            'on short classical-policy traces and execute much longer autonomous runs.',
            'Distribution shift and the fixed 500-update training cap can disadvantage',
            'an architecture; these results are not its optimized performance ceiling.',
            'Worlds are voxelized, static, generated to have reachable targets, and use',
            'point-agent motion. Ghost ranges/bearings are synthetic corruption proxies,',
            'not a multipath wave solver. Pose drift is a random walk, not full SLAM.',
            'The rotated condition changes coordinates, not gravity or physical attitude.',
            'The coordinate frame is fixed across a history: transporting recurrent state',
            'between changing body orientations is not tested. The geometric scalar',
            'features are also reflection-invariant and do not retain chirality.',
            'Single-agent testing does not evaluate communication loss or team dynamics.',
            'No continuum hierarchy, renormalization-equivariance, topos, stack, sheaf,',
            'biological interface, or material-device efficiency has been demonstrated.','',
            'NEXT EXPERIMENTS',
            '1. Add observable loop closure/localization uncertainty and verify return',
            '   under drift; require independent confirmation of suspicious target reports.',
            '2. Increase disjoint building families and training seeds; freeze a larger',
            '   evaluation protocol before tuning. Compare learning curves at several budgets.',
            '3. Run fixed-frontend and fixed-MAC controls to separate memory mechanism',
            '   from capacity allocation; add learned vector-equivariant temporal states.',
            '4. Add realistic sensor propagation, finite-body collision geometry, dynamic',
            '   obstructions, communication failures, and eventually a physics simulator.',
            '5. Define a coarse-graining map and semantic feature labels before testing',
            '   multiscale consistency. Changing point count alone is insufficient.',
            '6. Map the measured workload to candidate neuromorphic devices, then measure',
            '   latency/power on hardware rather than converting GPU MACs into joules.','',
            'EVIDENCE',
            'All per-episode results, checkpoints, initialization seeds and trace audits',
            'are retained in outputs/. See README.md for exact assumptions, equations,',
            'source references and reproduction commands. The earlier planar benchmark',
            'and existing LaTeX paper were not edited; their percentages are not directly',
            'comparable to this different task and budget.']
    (BASE/'COMPARISON.txt').write_text('\n'.join(lines),encoding='utf-8')
    (root/'analysis.json').write_text(json.dumps(dict(diagnostics=diagnostic,paired_geometry=paired),indent=2))
    print('\n'.join(lines[:34]))


if __name__=='__main__':main()
