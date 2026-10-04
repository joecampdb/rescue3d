# Rescue3D

**Can a search robot use geometry and memory to find targets and get home when its sensors are unreliable?**

This Python/JAX research prototype makes that question testable. An agent explores an unfamiliar three-dimensional voxel building, reports three hidden targets, and attempts to return to base within 360 actions. It receives partial, noisy point clouds and intermittent directional cues. The complete world is available to the simulator for scoring, not to the deployed agent.

This repository contains the simulation, eight neural architectures, training and evaluation code, tests, and scripts that generate comparison figures, a network topology legend, and an MP4. **Datasets, trained weights, evaluation records, and media are generated locally and are not bundled.**

## For a co-founder: what this establishes

The project investigates whether geometric representations and temporal memory improve a robot's search decisions under limited sensing and storage. It separates perception from planning so that the neural modules can be compared using the same map, controller, training data, and mission budget.

The business-relevant question is complete mission success: finding all targets **and returning to the true base**. Finding a target alone does not demonstrate reliable navigation. Localization errors can corrupt the remembered map even when perception works well.

This is a controlled software experiment, not a deployable rescue robot or a validated human detector. Its targets and sensing are synthetic. It does not establish a neuromorphic energy advantage, biological compatibility, or continuous multiscale semantic understanding.

## Results: completed pilot experiment

The completed 3 October 2026 run evaluated **24 trained networks plus a classical controller over 1,200 episodes**. Training used an RTX 4070; closed-loop evaluation ran on CPU. These are measured simulation outcomes, not projected hardware performance.

**Complete success means finding all three real targets AND returning to the true base within the action budget.** Each neural entry below pools eight held-out layouts and three training seeds (24 episodes). The classical controller has eight episodes per condition, without duplication over training seeds.

| Model | Nominal | Missing readings | False readings | Pose drift | Combined errors | Rotated coordinates |
|---|---:|---:|---:|---:|---:|---:|
| Classical controller | 7/8 | 6/8 | 2/8 | 0/8 | 0/8 | 7/8 |
| Conventional + feed-forward | 11/24 | 9/24 | 4/24 | 0/24 | 0/24 | 10/24 |
| Conventional + GRU | 16/24 | 7/24 | 1/24 | 1/24 | 1/24 | 19/24 |
| Conventional + reservoir | 10/24 | 10/24 | 1/24 | 1/24 | 1/24 | 13/24 |
| Conventional + spiking | 17/24 | 6/24 | 1/24 | 0/24 | 0/24 | 17/24 |
| Geometric + feed-forward | 16/24 | 11/24 | 2/24 | 0/24 | 0/24 | 16/24 |
| **Geometric + GRU** | **24/24** | **20/24** | **6/24** | 1/24 | 0/24 | **24/24** |
| Geometric + reservoir | 22/24 | 10/24 | 4/24 | 0/24 | 0/24 | 22/24 |
| **Geometric + spiking** | **24/24** | 15/24 | 5/24 | 0/24 | 1/24 | **24/24** |

“False readings” is the synthetic `multipath` profile; “rotated” applies a consistent coordinate rotation to neural inputs, not a physical change in robot attitude. Conventional and geometric models both received rotation augmentation during training.

### What the results mean

- **Geometry plus memory was promising in this pilot.** Geometric GRU and spiking models completed all nominal missions. The geometric GRU retained 83.3% completion with missing readings, versus 62.5% for geometric spiking and 75% for the classical controller.
- **Localization was the critical failure.** Under pose drift, the geometric GRU found 64 of 72 targets but returned in only 1 of 24 missions. The geometric spiking model found 66 of 72 targets but returned in none. Better detection alone did not solve navigation.
- **Memory contributed on identical observation histories.** Resetting geometric GRU state at every observation increased delayed-cue direction mean squared error from 0.1686 to 0.3350; for geometric spiking, from 0.2186 to 0.3513. These are prediction diagnostics, not mission-success percentages.
- **No neuromorphic energy claim follows.** Spiking dynamics were simulated, but hardware power was not measured. Floating-point rounding can also change spike decisions.

### Nominal mission efficiency

Actions include movement and listening. Means below include every nominal episode, including missions that missed targets; compare action counts alongside success, not in isolation.

| Model | Real targets found | Returned home | Mean actions |
|---|---:|---:|---:|
| Classical controller | 23/24 | 8/8 | 155.9 |
| Conventional + feed-forward | 51/72 | 24/24 | 289.3 |
| Conventional + GRU | 64/72 | 24/24 | 252.9 |
| Conventional + reservoir | 52/72 | 24/24 | 284.2 |
| Conventional + spiking | 64/72 | 24/24 | 249.3 |
| Geometric + feed-forward | 63/72 | 24/24 | 225.1 |
| Geometric + GRU | 72/72 | 24/24 | 93.4 |
| Geometric + reservoir | 70/72 | 24/24 | 187.4 |
| Geometric + spiking | 72/72 | 24/24 | 116.0 |

**Evidence limits:** there are only eight distinct test layouts, reused across architectures and training seeds. These counts do not establish population-level superiority or a general architecture ranking. All models share a mapper/planner; storage and state caps are matched, but computation and total RAM are not exactly equal. The geometric representation does not implement continuum feature hierarchies or renormalization equivariance.

These summaries were checked against the retained local per-episode evaluation records and trace audits. Raw records and checkpoints are not bundled in this repository; the reproduction commands below generate them. Backend/version differences can change individual trajectories.

## Run a simulation immediately

Use Python 3.11. The default demo runs on CPU and needs no training or downloaded weights.

```bash
git clone https://github.com/joecampdb/rescue3d.git
cd rescue3d
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip install -r requirements-cpu.txt
python demo.py
python demo.py --profile drift --out outputs/local_drift.json
python -m unittest discover -s tests -v
```

The demo runs the classical baseline, prints target counts, return status, actions, and timing, and records its trajectory in `outputs/local_demo.json`. Choose a fresh output filename for repeat runs. Other sensing profiles are `dropout`, `multipath`, `combined`, and `rotated`.

## What is being compared?

Two point-cloud encoders are crossed with four memory types, giving eight neural models plus a classical baseline.

| Component | Conventional choice | Geometric choice |
|---|---|---|
| Perception | Learned processing of XYZ point coordinates | Engineered distances and dot products followed by learned layers |
| Geometry | Learns from rotation-augmented examples | Designed for consistent rotation of geometric inputs and query directions |

| Memory | How it uses history |
|---|---|
| Feed-forward (`ff`) | No neural temporal state |
| GRU (`gru`) | Learned recurrent state and gates |
| Reservoir (`reservoir`) | Fixed recurrent dynamics; trained encoder and readout |
| Spiking (`spiking`) | Thresholded leaky integrate-and-fire dynamics; surrogate-gradient training |

Every model also uses the same external map and search history. A feed-forward neural module does not make the whole robot memoryless. The spiking model retains a dense perception frontend and does not emulate a specific material or chip.

The geometric encoder is a restricted symmetry-aware design, not a general SE(3)-equivariant graph network. Rotating a complete observation consistently is different from physically rotating a robot with gravity, changing visibility, and body dynamics.

## Sensing assumptions to challenge

- Up to 64 range-return points per observation provide partial geometry. Darkness alone does not remove these synthetic active-range observations.
- Intermittent directional cues supply information about targets. These are stipulated sensor signals, not realistic human recognition through rubble.
- Missing returns, false readings, and localization drift can be varied independently or together. The multipath profile is an error proxy, not simulated wave propagation.
- A point agent moves along six lattice directions through static obstacles. Flight, contact mechanics, communications, extraction, and moving debris are outside the model.
- Ground-truth labels are used during supervised training and evaluation scoring; the controller operates on observations and its estimated map.

## Resource budgets

| Resource | Comparison rule |
|---|---|
| Stored neural weights | At most 4,096 float32 parameters |
| Neural temporal state | At most 96 float32 values |
| Dense operations | At most 2 million estimated multiply-accumulates per observation |
| Sensing and mission | 64 point slots; 360 actions |
| Training | Same observations, rotation augmentation, and update schedule |

These are common caps, not exact equality in compute or total RAM. The shared map, transient activations, and software runtime are additional costs. An action is an abstract budget unit, not a measured amount of energy. CPU/GPU timing does not establish chip-level neuromorphic efficiency.

## Train and compare all models

From a fresh checkout, run the following in order:

```bash
python experiment.py
python train.py --out outputs/training_v2
python evaluate.py --training outputs/training_v2
python precision_audit.py
python failure_audit.py
python analyze.py
python render.py
python verify.py
```

The default dataset covers 64 procedural training scenes. Training uses three random seeds and eight architectures. Evaluation compares these 24 trained models and a classical control on eight held-out layouts across six sensor conditions, totaling 1,200 episodes. Layouts recur across models and training seeds: those episodes are not 1,200 independent buildings.

Training and evaluation take longer than the single-agent demo. The supplied requirements install CPU JAX; GPU training requires a compatible JAX/CUDA environment. An RTX 4070 is sufficient for this small benchmark, but CPU operation remains supported. Numerical results may vary across JAX versions and hardware, particularly near spike thresholds and controller ties.

The MP4 renderer requires **FFmpeg on PATH**. After training, generate just the topology legend without FFmpeg using:

```bash
python render.py --topology-only
```

Generated deliverables include:

- `outputs/figures/network_topologies.png`: network topology legend.
- `outputs/figures/simulation_comparison.png`: comparison across sensing conditions.
- `outputs/figures/eight_models_3d.mp4`: trajectories of all eight neural variants in a shared scene.
- `outputs/evaluation/episodes.json`: individual mission outcomes.

To run a trained model after the training step:

```bash
python demo.py --trained --encoder invariant --memory gru --out outputs/local_gru.json
python demo.py --trained --encoder xyz --memory spiking --out outputs/local_xyz_spiking.json
```

Scripts protect important existing outputs. Use their `--help` options for alternative destinations; audit and analysis scripts use the standard paths above. `verify.py` requires the completed experiment artifacts, whereas unit tests can run immediately.

## Code map

| File | Purpose |
|---|---|
| [world.py](world.py) | World generation, ray sensing, corruption, movement, and scoring |
| [controller.py](controller.py) | Shared map, search goals, pathfinding, return logic |
| [models.py](models.py) | Point encoders and four memory cores |
| [experiment.py](experiment.py) | Observation/action loop and training-data collection |
| [demo.py](demo.py) | Single-agent demonstration |
| [train.py](train.py) | Supervised learning under resource caps |
| [evaluate.py](evaluate.py) | Held-out missions and common-trace audits |
| [analyze.py](analyze.py) | Statistical summaries and memory diagnostics |
| [render.py](render.py) | Comparison figures, topology legend, and trajectory video |
| [precision_audit.py](precision_audit.py) | Floating-point symmetry diagnostic |
| [failure_audit.py](failure_audit.py) | Return-to-base failure inspection |
| [verify.py](verify.py) | Artifact consistency and resource checks |
| [tests/](tests/) | Simulation and architecture unit tests |

## How to interpret the experiment

Compare target finding, successful return, complete mission success, collisions, and computation separately. Report uncertainty across distinct layouts and training seeds. Inspect failures instead of ranking architectures from a single attractive trajectory. Use the shared-trace audits to distinguish prediction quality from changes in the visited trajectory.

Next research steps are uncertainty-aware localization, verified target reports, broader scene families, changing body orientation, longer histories, and controls with fixed frontends and more closely matched computation. A claim of renormalization equivariance additionally needs an explicit coarse-graining operation and a task defined across scales; neither is implemented here.
