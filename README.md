# ModuBalance-MuJoCo

**Distributed Intelligent Modular Balance Control in MuJoCo**

ModuBalance-MuJoCo is a small, reproducible feasibility experiment for Physical-AI-inspired modular robotics. It asks one deliberately narrow research question:

> Can a physically connected module use only local physical measurements to recover pitch balance after initial, payload, and moving-foundation disturbances?

This is not a walking robot and it is not yet a magnetic-joint simulation. The hinge motor is an **idealized intelligent connector**. Starting with an ideal active connector makes the control hypothesis testable before magnetic-force, latch, and multi-module complexity is introduced.

## What the experiment tests

Cube B is treated as a local intelligent agent. A named sensor interface gives it connector angle/rate, module orientation/angular velocity, measured actuator torque, and connector force. The agent does not receive MuJoCo's full state, the payload-active flag, the experiment phase, or a message saying that a disturbance occurred. It reacts only to measured physical consequences.

Two experiments make the controller's contribution falsifiable:

- `pd`: local intelligent PD connector control.
- `off`: passive/no intelligent connector torque.

The comparison and payload sweep quantify whether the controller helps and where its operating envelope ends.

## Physical model

The MJCF model contains a ground plane, X-axis sliding platform, rigid Cube A, hinged Cube B, ideal torque motor, platform position servo, local IMU/connector sensors, and a visible payload location.

```text
                    equivalent payload load
                              ↓
                       ┌───────────┐
                       │  Cube B   │  local agent
                       └─────●─────┘
                             ↑
                    controlled Y hinge
                             ↓
                       ┌───────────┐
                       │  Cube A   │
                       └─────●─────┘
                             │
                  ┌────────────────────┐  → X
                  │  movable platform  │
                  └────────────────────┘
════════════════════════════════════════════════ ground
```

All values use SI units internally.

| Quantity | Default | Reason |
|---|---:|---|
| Physics timestep | 0.002 s | 500 Hz fixed-step dynamics |
| Controller period | 0.010 s | 100 Hz local control |
| Gravity | 9.81 m/s² downward | Earth gravity |
| Platform | 0.70 × 0.45 × 0.08 m, 12 kg | Stable translating foundation |
| Cube A | 0.24 m side, 3 kg | Rigid lower module |
| Cube B | 0.24 m side, 2 kg | Upper intelligent module |
| Hinge | Y-axis, ±35°, 0.35 N·m·s/rad damping | Pitch-only first experiment |
| Initial pitch | 5° | Recoverable initial disturbance |
| Payload | 0.50 kg equivalent at +0.06 m X offset | Produces both force and pitch moment |
| Base target speed | 0.05 m/s | Slow foundation motion |
| Base velocity ramp | 0.75 s half-cosine | Finite, smooth acceleration |
| PD gains | Kp = 45 N·m/rad, Kd = 7.5 N·m·s/rad | Stable tested baseline |
| Connector limit | ±12 N·m | Explicit actuator saturation |
| Balance band | ±2° | Proof-of-concept criterion |
| Required dwell | 1.0 s | Continuous time inside the balance band |

The XML values and their experiment-level overrides are in `models/modular_balance.xml` and `config/default.yaml`.

## Local intelligent connector

`IntelligentModuleAgent` accepts a `ConnectorPolicy`. The current `PDConnectorController` implements

```text
e(t) = θdesired - θ(t)
τraw(t) = Kp e(t) - Kd θ̇(t)
τ(t) = clip(τraw(t), -τmax, +τmax)
```

where `θdesired = 0`. Module pitch and its local Y angular rate drive the controller. The policy returns commanded torque, error, and saturation status. `OffConnectorController` implements the identical interface and returns zero torque.

This boundary is intentional: a future adaptive controller, MPC policy, learned policy, or magnetic connector can implement `compute_action(observation)` without rewriting the simulator. Multiple `IntelligentModuleAgent` instances can later run separate policies.

## Experiment timeline

1. **0–5 s — initial recovery:** Cube B starts at approximately 5° pitch and attempts to return to zero.
2. **5–10 s — payload disturbance:** an off-centre downward load starts abruptly. The agent is not told that it started.
3. **10–15 s — moving foundation:** the platform follows a half-cosine velocity ramp toward 0.05 m/s and then moves at constant target velocity.

### Payload implementation

Runtime attachment of a free body can add constraint impulses and topology bookkeeping unrelated to the control question. This version therefore uses the allowed equivalent-load approach. At 5 s it applies the physically equivalent wrench

```text
Fz = -m_payload g
Ty = offset_x m_payload g
```

to Cube B at the actual off-centre `payload_site`. This changes MuJoCo dynamics—it is not a display-only parameter. A zero-mass red block rests 2 mm above Cube B and becomes opaque when the wrench activates. Payload sweep values are equivalent supported masses in kilograms.

## Measurements and outputs

Every 0.002 s row records:

- time, Cube B pitch and roll;
- hinge angle and angular velocity;
- commanded and measured/applied actuator torque;
- connector force magnitude and Y torque;
- base position, velocity, target position, and target velocity;
- payload-active flag;
- controller error/output and saturation flag.

One experiment writes `results/<controller>/experiment.csv`, `evaluation.json`, and:

- `pitch_response.png`
- `controller_torque.png`
- `connector_measurements.png`
- `base_motion.png`
- `joint_response.png`

Pitch plots include payload/base event markers and the ±2° balance band.

## Quantitative PASS/FAIL

The evaluator reports phase peak pitch, final error, initial/payload/base settling time, total and moving-base RMS pitch, maximum commanded torque, and saturation. Settling means the measured absolute pitch is at or below the configured tolerance for a continuous configured dwell period before that phase ends. No status is hard-coded.

- Initial recovery passes if a valid dwell interval begins during 0–5 s.
- Payload recovery passes if a valid dwell interval begins during 5–10 s.
- Moving-base recovery passes if a valid dwell interval begins during 10–15 s.
- Overall feasibility passes only when all three phases pass.

A disturbance that never leaves the tolerance band can have a near-zero settling time; that correctly means it remained balanced under the stated criterion.

## Project structure

```text
ModuBalance-MuJoCo/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── compare.py
├── sweep.py
├── config/
│   └── default.yaml
├── models/
│   └── modular_balance.xml
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── configuration.py
│   ├── controller.py
│   ├── disturbances.py
│   ├── evaluation.py
│   ├── logger.py
│   ├── plotting.py
│   ├── sensors.py
│   └── simulation.py
├── tests/
│   ├── test_controller.py
│   └── test_evaluation.py
└── results/
    └── .gitkeep
```

## Installation

Python 3.10–3.13 is recommended. From the project directory on Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The viewer requires a working desktop/OpenGL environment. Headless runs do not open a window.

## Running

Interactive controlled experiment:

```bash
python main.py
```

Headless controlled experiment:

```bash
python main.py --headless --controller pd
```

Passive baseline:

```bash
python main.py --headless --controller off
```

Duration or payload override:

```bash
python main.py --headless --duration 15
python main.py --headless --payload-mass 0.75
```

The simulation always uses the configured fixed physics timestep. Viewer synchronization affects wall-clock presentation only, never the simulated dynamics.

## Baseline comparison

After the two normal runs, load their CSV files and create an overlaid pitch plot:

```bash
python compare.py
```

If either CSV is absent, `compare.py` runs that experiment headlessly. Force both to rerun with:

```bash
python compare.py --rerun
```

The output is `results/comparison/pitch_pd_vs_off.png`. A useful experiment should show materially different OFF and PD trajectories.

## Payload sweep

Run the default 0.10, 0.25, 0.50, 0.75, 1.00, and 1.50 kg equivalent loads:

```bash
python sweep.py
```

Or choose values:

```bash
python sweep.py --masses 0.2 0.4 0.8 1.2
```

Outputs are `results/payload_sweep.csv`, `payload_vs_peak_pitch.png`, `payload_vs_settling_time.png`, and `payload_vs_rms_error.png`. Per-run raw data lives under `results/sweep_runs/`.

## Tests

```bash
pytest
```

The tests verify PD sign/damping/saturation and the continuous-dwell settling rule. The required end-to-end validation commands are:

```bash
python main.py --headless --controller pd
python main.py --headless --controller off
python compare.py
python sweep.py
pytest
```

## Interpreting results

- Pitch should begin around 5°, change dynamically, and recover under PD.
- At 5 s, the payload should create a new nonzero equilibrium and connector torque.
- At 10 s, finite platform acceleration should create a small pitch transient and velocity tracking response.
- PD torque should be nonzero; OFF torque should remain zero.
- Saturation is a measured event, not automatically a failure. It signals that the requested policy output exceeded the ideal connector limit.
- Payload-sweep failure near the edge of the ±2° band is useful: it identifies the current controller/actuator operating envelope.

## Limitations

- Only pitch is actuated. Roll is measured and logged but not controlled.
- Cube A is rigidly mounted to the translating platform; the model does not include feet, compliance, contact-rich walking, or structural reconfiguration.
- The payload is an equivalent external wrench, not a colliding object or runtime-welded body.
- The ideal motor can command positive or negative hinge torque directly. It does not model field strength, air gap, coil dynamics, heating, magnetic saturation, current limits, or a latch.
- The fixed PD controller has no integral action, so an off-centre constant load creates a small steady-state pitch error.
- One agent controls one connector. This establishes an architectural boundary but does not yet demonstrate multi-agent coordination.
- The present pass criterion tests balance recovery, not energy efficiency, robustness to sensor noise/delay, or hardware feasibility.

## Planned research sequence

```text
ideal active connector
        ↓
prove distributed balancing feasibility
        ↓
introduce actuator limits
        ↓
replace connector with magnetic-force model
        ↓
multiple independently controlled magnets
        ↓
mechanical latch
        ↓
multiple intelligent modules
        ↓
structural reconfiguration
        ↓
two-legged/wheeled robot
```

The recommended immediate next step is a connector plant interface that converts local commands into force at several spatially separated magnetic elements, with gap-dependent force limits and current dynamics. Keep the existing local observation, baseline, logging, and evaluation unchanged so the ideal and magnetic connectors can be compared fairly.

## Phase 2: multi-agent, magnetic, topology, and stress experiments

Phase 2 extends—rather than replaces—the original experiment. `python main.py` still runs the Phase-1 viewer, while `--experiment` selects additive research stages.

### Clone and run in a VS Code terminal

On Windows PowerShell:

```powershell
git clone https://github.com/AbdulAhadDilshad/ModuBalance-MuJoCo.git
cd ModuBalance-MuJoCo
code .
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the original graphical simulation:

```powershell
.\.venv\Scripts\python.exe main.py --controller pd
```

Run Phase-2 headlessly:

```powershell
.\.venv\Scripts\python.exe main.py --headless --experiment three_module --controller distributed
```

On macOS/Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

### Three-agent architecture

The new model stacks Cube A, B, and C with co-located roll/pitch hinge pairs at AB and BC. Neutral geometry gaps are 0 m at platform/A, A/B, and B/C, with a 2 mm payload clearance. Each of A, B, and C owns a separate `DistributedModuleAgent`.

The only policy input is an immutable `Phase2LocalObservation` containing:

- local pitch and roll;
- local pitch and roll rate;
- local connector angle and rate;
- local connector force and torque;
- the local connector-connected flag;
- messages explicitly delivered by `CommunicationChannel`.

Nearest-neighbour links are strictly `A↔B↔C`. A message contains only `pitch_error`, `angular_velocity`, and `connector_load`. The channel has no MuJoCo model/data reference and unit tests prove that A messages cannot be delivered directly to C. Delay, dropout, and Gaussian message noise are configurable.

Run the baselines:

```powershell
.\.venv\Scripts\python.exe main.py --headless --experiment three_module --controller centralized
.\.venv\Scripts\python.exe main.py --headless --experiment three_module --controller distributed
.\.venv\Scripts\python.exe main.py --headless --experiment three_module --communication none
.\.venv\Scripts\python.exe main.py --headless --experiment three_module --controller passive
```

`--mode centralized`, `--mode distributed`, and `--mode passive` are equivalent shortcuts that select the three-module experiment.

### Four-element magnetic connector

`IdealTorqueConnector` retains direct motor torque as the scientific upper-bound baseline. `MagneticConnector` uses four independently commanded attraction pairs at each interface:

```text
M1 front-left   (+0.08, +0.08 m)    M2 front-right (+0.08, -0.08 m)
M3 rear-left    (-0.08, +0.08 m)    M4 rear-right  (-0.08, -0.08 m)
```

The force magnitude is

```text
F_i = u_i F_max / (1 + (d_i / d0)^2),   0 <= u_i <= 1
```

with nominal `F_max = 18 N` per element and `d0 = 0.020 m`. Forces are applied at the actual upper/lower MJCF sites with equal magnitude and opposite direction. MuJoCo therefore produces moments from `r × F`; no unexplained corrective torque is inserted. Front/rear command differences control pitch, and left/right differences control roll.

```powershell
.\.venv\Scripts\python.exe main.py --headless --experiment magnetic
.\.venv\Scripts\python.exe main.py --headless --experiment magnetic --latch on
.\.venv\Scripts\python.exe compare_connectors.py
```

### Mechanical latch

Each connector has an initially inactive MuJoCo weld equality. A latch remains open while alignment is poor. It locks only after angle is within 1° and angular speed within 0.08 rad/s continuously for 0.5 s. It unlocks above 3° or 0.35 rad/s. This dwell plus hysteresis prevents chatter. Once locked, magnetic holding commands scale to 25% of their unlocked value.

### Runtime topology experiments

Topology is logged globally but distributed policies receive only their own connection flag and active communication links.

```powershell
.\.venv\Scripts\python.exe main.py --headless --experiment remove_module
.\.venv\Scripts\python.exe main.py --headless --experiment add_module
.\.venv\Scripts\python.exe main.py --headless --experiment reposition_module
```

- Remove deactivates BC, removes C's dynamic mass contribution, hides it, breaks the B–C communication link, and applies a small disconnect recoil.
- Add begins with C dynamically inactive, then restores its physical mass and BC link with a 3° attachment misalignment.
- Reposition shifts Cube C and its local sites by +60 mm in X, applies the equivalent shifted-C gravity moment, and introduces a 4° reattachment error.

These operations occur in one running MuJoCo simulation. Agents are not sent an experiment-phase or topology-event message.

### Stress testing

```powershell
.\.venv\Scripts\python.exe stress_test.py --trials 20
```

The one-factor Monte Carlo design varies payload, base velocity, impulse force, magnet strength, friction, sensor noise, communication delay, and packet loss. A deterministic payload/base-velocity grid supplies the 2-D recovery heatmap. Seeds and every test parameter are recorded in `results/stress/stress_results.csv`.

Plots include payload/velocity/magnet/friction/noise success rates, payload peak pitch, communication-delay settling time, and the payload–velocity heatmap.

### Communication comparison

```powershell
.\.venv\Scripts\python.exe compare_communication.py
```

This runs centralized, nearest-neighbour distributed, local-only distributed, and 50 ms delayed distributed control using identical dynamics and disturbances.

### Phase-2H feasibility prototypes

These are standing/slow-motion prototypes, not locomotion controllers:

```powershell
.\.venv\Scripts\python.exe main.py --headless --experiment two_leg
.\.venv\Scripts\python.exe main.py --headless --experiment wheeled
```

The two-leg rig evaluates standing attitude, COM projection, foot contacts, payload shift, and small forward/lateral disturbances. It does not generate gait. The wheeled prototype uses a smooth 0.05 m/s foundation trajectory and tests pitch control under payload and force disturbances; it is not a high-speed vehicle model.

### Phase-2 outputs and limitations

Every three-module run writes a wide `experiment.csv`, `evaluation.json`, module pitch/roll plots, connector-force plot, all magnet commands, and latch/topology states. Logs include every module's attitude/rates/action, every connector's force/torque/latch/active state, all eight magnet commands and forces, communication receive counts, topology state, saturation, separation, and effort.

The runtime add/remove mechanism uses dynamic activation/deactivation of Cube C's mass and BC participation because the stable stack model retains hinge topology. It is a robust topology-feasibility surrogate, not a free-flying docking simulation. The two-leg model is a support-pivot standing rig; walking, gait generation, actuator current/thermal dynamics, magnetic finite-element fields, and hardware latch impact mechanics remain future work.
