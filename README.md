# PHYSAI-RL-ROBOT-X

**Simulation-first embodied AI: hybrid VLA planning + RL execution for robust robot manipulation**

> *A modular robotics pipeline combining open-source Vision-Language-Action model planning with reinforcement learning control, benchmarked under noise, latency, and failure recovery conditions.*

---

## Motivation

Modern robot foundation models (OpenVLA, GR00T) excel at task understanding and instruction following, but remain fragile under continuous low-level motor control. Pure RL controllers master low-level execution but generalize poorly across task variants. This project closes that gap with a **hierarchical hybrid architecture**:

```
Instruction
  → Scene Perception (YOLO + depth)
  → World State Encoder
  → Task Planner (LLM / VLA)
  → Skill Router
  → RL Executor (PPO / SAC)
  → Safety Checker
  → Reward & Diagnostics
  → Failure Memory → Improvement Loop
```

---

## System Architecture

```
physai-rl-robot-x/
├── envs/                    # Simulation backends + gym wrappers
│   ├── isaac/               # Isaac Sim + MockIsaacEnv
│   ├── wrappers/            # NoiseWrapper, LatencyWrapper, GymWrapper
│   └── task_definitions/    # PickAndPlace, SortObjects, StackBlocks
│
├── src/
│   ├── perception/          # Object detection (YOLO) + scene parsing
│   ├── state_encoder/       # World state → RL-ready tensor
│   ├── planner/             # LLMPlanner, VLAPlanner, MockLLMPlanner
│   ├── skill_router/        # Maps plan steps → RL controller calls
│   ├── control/             # RLExecutor, PPO builder, SAC builder
│   ├── safety/              # SafetyChecker: bounds, collision, limits
│   ├── memory/              # TrajectoryStore, FailureLogger
│   ├── diagnostics/         # MetricsComputer, EpisodeLogger
│   ├── training/            # Trainer (rl_only, hybrid, curriculum)
│   └── evaluation/          # BenchmarkRunner, NoiseSuite
│
├── configs/                 # Hydra/YAML configs (sim, planner, RL, tasks, eval)
├── scripts/                 # Shell scripts for training & benchmarking
├── tests/                   # Unit, integration, regression, safety tests
└── results/                 # Figures, tables, logs, videos
```

---

## Research Question

> **Can a hybrid system that combines open-source VLA/LLM task planning with RL-based skill execution outperform RL-only and planner+scripted baselines on task success, recovery rate, and robustness in realistic simulation?**

### Sub-questions
- Does VLA/LLM planning improve task generalization across instruction variants?
- Does RL execution stabilize control under continuous noise and perturbation?
- How much does planner latency degrade the pipeline under tight control timing?
- Which failures are perception, planning, or control bottlenecks?
- Can failure memory improve future episode performance?

---

## Task Suite

| Task | Difficulty | Key Challenge |
|------|-----------|---------------|
| Pick & Place | Easy | Grasp accuracy, goal positioning |
| Sort Objects | Medium | Multi-object planning, category matching |
| Stack Blocks | Medium | Precise placement, order enforcement |
| Drawer Insert | Hard | Articulated manipulation |
| Recovery | Hard | Disturbance handling, replanning |

---

## Baselines

| Baseline | Description |
|----------|-------------|
| **RL Only (PPO)** | End-to-end PPO, no planner |
| **RL Only (SAC)** | End-to-end SAC, no planner |
| **Scripted + RL** | Hardcoded skill sequence + RL execution |
| **Hybrid (LLM + PPO)** | LLM/VLA planner + RL executor + recovery |

---

## Noise & Latency Stress Tests

The benchmark suite injects configurable noise and delays at four levels:

| Channel | None | Low | Medium | High |
|---------|------|-----|--------|------|
| Camera noise std | 0.0 | 0.02 | 0.05 | 0.10 |
| Actuation noise std | 0.0 | 0.01 | 0.03 | 0.06 |
| Observation delay (steps) | 0 | 2 | 5 | 10 |
| Action delay (steps) | 0 | 1 | 3 | 5 |
| Frame drop prob | 0.0 | 0.02 | 0.05 | 0.10 |

---

## Primary Metrics

- **Task success rate** — fraction of episodes where goal is achieved
- **Average completion time** — wall-clock seconds per episode
- **Collision count** — average per-episode collisions
- **Recovery success rate** — success rate on episodes requiring replanning
- **Replan count** — average number of replanning invocations
- **Reward convergence** — training curve quality
- **Robustness under noise** — success rate degradation across noise levels
- **Planner latency** — mean / P95 inference time in ms

---

## Stack

| Component | Technology |
|-----------|-----------|
| Simulation | NVIDIA Isaac Sim (+ MockIsaacEnv for CI) |
| RL training | stable-baselines3 (PPO, SAC) |
| High-level planning | OpenAI API / local LLM / MockLLMPlanner |
| VLA integration | OpenVLA (openvla/openvla-7b via HuggingFace) |
| Object detection | YOLOv8 (ultralytics) + mock detector |
| Config management | Hydra + OmegaConf |
| Experiment tracking | Weights & Biases + TensorBoard |
| Data format | LeRobot-compatible JSONL trajectories |

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/aryanputta/neurosymbolic-robot-x
cd neurosymbolic-robot-x
bash scripts/setup_env.sh

# 2. Run the unit + integration tests (no GPU required)
pytest tests/ -v

# 3. Train the RL-only PPO baseline (mock env, no Isaac Sim required)
physai-train --task pick_and_place --algorithm ppo --mode rl_only --total-timesteps 500000

# 4. Evaluate with noise injection
physai-eval --task pick_and_place --noise-level medium --n-episodes 50

# 5. Run the full benchmark suite
physai-benchmark
```

---

## Training Phases

| Phase | Description |
|-------|-------------|
| 1 | Build simulator scene + single task environment |
| 2 | Train RL-only PPO baseline |
| 3 | Add LLM planner interface with structured skill plans |
| 4 | Connect planner → RL executor pipeline |
| 5 | Add failure recovery + replanning |
| 6 | Latency + noise stress testing |
| 7 | Multi-task curriculum evaluation |
| 8 | Ablation study + benchmark tables |

---

## Ablation Study

The benchmark framework supports ablation over:
- No memory (disable failure logger)
- No replanning (single-shot planning only)
- No RL stabilization (scripted motion only)
- No perception noise
- No latency
- Planner-only (VLA / LLM without RL executor)
- RL-only (no planner)
- Different RL algorithms (PPO vs. SAC vs. TD3)
- Different skill granularity (coarse vs. fine skill decomposition)

---

## Failure Analysis

The `FailureLogger` and `FailureClassifier` automatically categorize each episode failure:

| Category | Description | Common Fix |
|----------|-------------|-----------|
| Perception | Object not found / wrong position | Raise detection confidence threshold |
| Planning | Hallucinated target / invalid skill | Add schema validation + skill whitelist |
| Control | Skill timeout / grasp failure | Tune RL reward; increase max steps |
| Safety | Collision / out-of-bounds | Tighten workspace bounds in planner |
| Timeout | Episode time limit exceeded | Shorten plan; increase control Hz |

---

## Expected Results

| Condition | RL Only | Hybrid |
|-----------|---------|--------|
| Clean sim | Competitive on narrow tasks | Better on multi-step, novel instructions |
| Medium noise | Degraded | More robust (replanning helps) |
| High latency | Severely degraded | Partially resilient (plan caching) |
| Recovery task | Poor | Significantly better |
| Zero-shot new task | Fails | Generalizes via language |

---

## Limitations

1. **Planner latency**: LLM calls add 50–500 ms. Mitigated by plan caching and reduced call frequency.
2. **Sim-to-real gap**: Results are simulation-only. Domain randomization is configurable but not validated on real hardware.
3. **VLA model weight size**: OpenVLA-7B requires ~14 GB VRAM. Use mock planner for resource-constrained runs.
4. **RL sample efficiency**: PPO requires ~1M steps for simple tasks. SAC is more sample-efficient.
5. **Perception cascade**: Detection errors propagate to planning. Confidence thresholds are configurable.

---

## Resume Bullet

*Designed a modular embodied AI pipeline in simulation combining open-source VLA/LLM task planning with RL skill execution, safety-constrained control, and failure-driven improvement loops — evaluated across pick-and-place, sorting, and stacking tasks under sensor noise, actuation delay, and replanning stress tests using Isaac Sim + stable-baselines3.*

---

## Citation / References

- [OpenVLA](https://github.com/openvla/openvla) — Open vision-language-action model for robot manipulation
- [LeRobot](https://github.com/huggingface/lerobot) — HuggingFace robotics tooling and datasets
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim) — Physically based robotics simulation
- [NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/gr00t) — Open robot foundation model platform
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO, SAC, TD3 implementations

---

## License

MIT License — see [LICENSE](LICENSE) for details.
