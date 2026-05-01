# Sim-to-real for robotics in 2026: a Spot-flavored field guide

**The reality gap is now better diagnosed than solved**, and the much-hyped "one model for all robots" is, as of April 2026, still a useful research direction rather than a deployable reality. The strongest sim-to-real recipe for a PPO Spot policy in JAX/Flax + MJX combines four ingredients that have repeatedly held up across labs: massively parallel PPO with a privileged critic, a small student actor on proprio history, modest domain randomization once the simulator is well-tuned, and a real-data residual (delta-action or actuator network) to close the last gap. Generalist VLAs (RT-X, Octo, OpenVLA, π₀, CrossFormer, GR00T) genuinely share trunks across robots but **almost always retain embodiment-specific input/output projectors**, and none currently runs the high-frequency torque control that legged hardware demands. Below, the three questions are addressed in turn, with concrete papers, what they actually showed, and where claims are softer than they look. Confidence flags: **High** = multiple independent replications; **Medium** = single strong paper or close-knit lab line; **Low** = recent preprint, marketing-adjacent, or unverified. Several arXiv IDs in the 2511–2603 range are very recent preprints whose numbers should be treated as preliminary.

---

## 1) Where sim-to-real actually breaks

### The dominant failure mode is dynamics, not pixels

For legged robots, the **single best-supported claim** in the field is that **actuator and joint dynamics dominate the reality gap**, not vision or kinematics. This goes back to **Hwangbo et al., *Learning agile and dynamic motor skills for legged robots*, Science Robotics 2019** (arXiv 1901.08652), which introduced the actuator network — a small MLP mapping joint position-error history to applied torque, trained on real ANYmal data. Every successor paper either uses this trick or argues about how to replace it. **Tan et al. 2018** (arXiv 1804.10332) had already named "inaccurate actuator models and lack of latency modeling" as the two biggest causes of failure on Google's quadruped, and the diagnosis has held.

The most useful 2025-2026 reads on this for your stack are three papers. **Bjelonic, Tischhauser, Hutter — *PACE: Towards Bridging the Gap*, arXiv 2509.06342** (ETH RSL, 2025) shows that a minimal **4n+1 parameter set per robot** (per-joint armature, viscous damping, Coulomb friction, joint bias, plus a global delay) captures most of the legged sim-to-real gap on 13 different quadrupeds, including ANYmal and Tytan, with **no domain randomization** required and only ~20 s of in-air encoder data. They report a 32% cost-of-transport reduction on ANYmal and demonstrate that URDF-only models miss joint-level Δq/Δq̇ phase portraits by roughly ×5. **Sobanbabu et al. — *SPI-Active*, CoRL 2025** (arXiv 2505.14266) reports 42–63% lower tracking error than DR baselines on Unitree Go2 and G1 humanoid using sample-based system ID with active exploration, again without torque sensors. **Fey, Margolis, Peticco, Agrawal — *UAN: Unsupervised Actuator Network*, arXiv 2502.10894** (MIT Improbable AI, 2025) extends Hwangbo's actuator net to harmonic-drive joints without torque supervision, which is **directly relevant to Spot** because Boston Dynamics doesn't expose torques to users.

### The most Spot-specific paper

**Miller, Yu, Brauckmann, Farshidian — *High-Performance Reinforcement Learning on Spot*, ICRA 2025** (arXiv 2504.17857, RAI Institute / Boston Dynamics) is the single highest-priority read for your project. It is the **first published end-to-end RL policy on Spot**, built on Isaac Lab + the Spot RL Researcher Development Kit, reaching 5.2 m/s gallop with a flight phase (more than 3× the factory controller). Critically, they use **Wasserstein Distance and Maximum Mean Discrepancy between hardware and sim joint distributions** as a CMA-ES fitness function for fitting unknown sim parameters — a principled, distributional sim-to-real metric that you can replicate in MJX. **Confidence: High** for the methodology, **Medium** for whether the absolute speed numbers transfer outside controlled environments.

### Contact, friction, and MJX-specific pitfalls

**Le Lidec, Jallet, Montaut, Carpentier et al. — *Contact Models in Robotics: a Comparative Analysis*, arXiv 2304.06372** (T-RO under review) is the cleanest cross-simulator benchmark, comparing MuJoCo, Bullet, Drake, RaiSim, PhysX on Solo, Talos, and Allegro hand. They show MuJoCo's regularized PGS soft contact introduces measurable interpenetration and creep at large timesteps and explicitly warn that "approximations or algorithms commonly used in robotics can severely widen the reality gap." For MJX specifically, **Dakhia et al. — *Hard Contacts with Soft Gradients (DiffMJX)*, arXiv 2506.14186, 2025** quantifies inaccurate gradients in MJX from time-discretization of hard contact and proposes adaptive integration via Diffrax with 2–5× compute overhead. **Choi, Hwangbo et al. — *Learning quadrupedal locomotion on deformable terrain*, Science Robotics 2023** (10.1126/scirobotics.ade2256) shows the harder failure mode: standard rigid-contact RL **collapses on sand and mattresses** even with friction randomization, because the underlying contact model is wrong, not just miscalibrated.

For static friction, **Zhong et al., arXiv 2503.01255 (2025)** shows that lack of stiction modeling alone is sufficient to cause manipulation failures even with broad DR. And **Aceituno-Cabezas et al., RA-L 2022** (arXiv 2110.00541) remains the most-cited motion-capture-grounded comparison of impact dynamics across simulators.

### Sensors, latency, and observation drift

**Sensor gaps** are most studied for cameras. **DREDS** (Dai et al., ECCV 2022, arXiv 2208.03792) and **SimSense** add realistic active-stereo IR-pattern noise via Blender raytracing, and 2024-2025 work like *"Close the Sim2Real Gap via Physically-based Structured Light Synthetic Data Simulation"* (arXiv 2407.12449) extends this to gray-code structured light. For depth-based locomotion specifically, **Miki et al., Science Robotics 2022** (arXiv 2201.08117) catalogs in-the-wild depth failure modes — snow, vegetation, water, dust, fog, reflective surfaces — and is the canonical reference on "what real depth fails on." **Chen et al. — *MMDR (Multi-Modal Delay Randomization)*, arXiv 2109.14549** showed that asynchronous proprio-vs-depth latency, not magnitude, breaks vision-conditioned legged RL; per-modality delay randomization is now standard.

For drones, **NeuroBEM** (Bauersfeld et al., RSS 2021, arXiv 2106.08015) is the cleanest aerodynamic-gap paper: a hybrid blade-element + neural-residual model halved force/torque RMSE on a quadrotor up to 65 km/h. **Coursey et al., DX 2024** (DOI 10.4230/OASIcs.DX.2024.16) report a roughly 100% increase in flight-path deviation real-vs-sim on octocopters, and **Ferede et al., arXiv 2311.16948** show end-to-end RL's 1.39 s sim advantage over baseline collapses to 0.17 s in reality — a clean negative-result example that's rare in this literature.

For **observation distribution shift**, the modern view is that "extrinsic-parameter drift" (changing mass, friction, payload, motor strength) is the main thing the policy must adapt to, and most failures are in the *adaptation module*, not the base policy — see **Kumar, Z. Li et al., *Adapting RMA for Bipedal Robots*, IROS 2022, arXiv 2205.15299**, which explicitly notes "extrinsics estimator could be imperfect, which leads to poor performance of the base policy which expects a perfect estimator."

### Surveys to anchor the taxonomy

The most useful single reference is **Aljalbout, Xing, Romero, Akinola, Garrett, Heiden, Gupta, Hermans, Narang, Fox, Scaramuzza, Ramos — *The Reality Gap in Robotics: Challenges, Solutions, and Best Practices*, Annual Review of Control, Robotics, and Autonomous Systems 2026** (arXiv 2510.20808, project: robotics-reality-gap.github.io). It decomposes the gap into atomic causes (dynamics, contact, sensing, actuation, observation, control loop) with metrics and is the natural anchor citation. **Ha, Lee, van de Panne, Xie, Yu, Khadiv — *Learning-based legged locomotion: state of the art*, IJRR 2025** (arXiv 2406.01152) is the legged-specific complement; sections on actuator-bandwidth, torque tracking, state-estimation drift, and contact bifurcations are the most useful.

### The most under-reported failure mode: silent failures

Almost no paper publishes negative real-world rollouts with controlled baselines, which is why **Shi, Zhang, Miki, Lee, Hutter, Coros — *Rethinking Robustness Assessment*, RSS 2024** (arXiv 2405.12424) is so valuable: they show that even SOTA "robust" controllers from Miki et al. 2022 fail under low-magnitude adversarial perturbations on real ANYmal, and adversarial fine-tuning robustifies them. **RAPT, arXiv 2602.01515 (2026, very recent)** frames this as the "deployment paradox" — high in-sim reward, catastrophic real degradation from OOD states — and proposes a 50 Hz online monitor. Treat the "zero-shot sim-to-real" framing in any 2024-2025 paper with skepticism unless reproduction code and failure rates are reported.

---

## 2) Methods and architectures that actually transfer

### Domain randomization, honestly

The Tobin 2017 (arXiv 1703.06907) and Peng 2018 (arXiv 1710.06537) DR papers established the recipe; **OpenAI's ADR (Akkaya et al. 2019, arXiv 1910.07113)** introduced automatic curriculum expansion of randomization ranges and demonstrated Rubik's cube on a Shadow Hand at 60% on easy / 20% on hard scrambles after roughly 13,000 simulated years of experience. The honest framing today is that DR moves the engineering burden from sim-fidelity to reward and DR-range design, and **wider DR makes policies more conservative**. The **DrEureka** paper (Ma et al., RSS 2024, arXiv 2406.01967) automates DR-range selection via LLM-generated reward functions plus a "reward-aware physics prior" — they report 34% better speed and 20% better distance on quadruped locomotion than human-tuned, with 300% more in-hand cube rotations, suggesting **human DR ranges are routinely sub-optimal**. Confidence: Medium for the LLM-grounding claim; the empirical wins are real but compute-heavy.

The most pragmatic counterpoint is **Berkeley Humanoid (Liao et al., arXiv 2407.21781, 2024)**: when hardware is co-designed for low-backlash and accurate URDF, **light DR plus simple PPO** suffices for outdoor walking and single-leg hopping. The lesson — invest in actuator and contact modeling first, DR second.

### Privileged learning and teacher-student is the legged-locomotion recipe

The ETH RSL line is now the dominant template. **Hwangbo 2019 → Lee 2020 → Miki 2022** evolved from "actuator network alone" to "privileged teacher + proprioception-only TCN student" to "GRU + attention encoder fusing proprio and noisy exteroception." **Lee et al., *Learning quadrupedal locomotion over challenging terrain*, Science Robotics 2020** (arXiv 2010.11251) gives the clearest two-stage privileged-distillation pattern: teacher MLP sees terrain heights, friction, contact states, disturbances; student is a Temporal Convolutional Network over proprio history (~50 steps) supervised to reproduce the teacher's latent and actions.

**Kumar, Fu, Pathak, Malik — *RMA: Rapid Motor Adaptation*, RSS 2021** (arXiv 2107.04034) is the simpler and most easily implementable variant: train base policy with privileged extrinsics z, then train an adaptation module φ that maps proprio history to ẑ via supervised regression. They report 73.5% success vs 76.2% with true z and 52.1% no-adaptation across diverse terrains on Unitree A1. **Confidence: High** — multiple independent re-implementations confirm the pattern works.

The 2024-2025 evolution adds explicit state estimators: **DreamWaQ** (Nahrendra, Yu, Myung, ICRA 2023, arXiv 2301.10602) trains a concurrent context-aided estimator network outputting explicit base velocity plus an implicit terrain latent via VAE-style reconstruction, paired with asymmetric actor-critic. **HIM (Long et al., ICLR 2024, arXiv 2312.11460)** drops privileged rollouts and uses contrastive learning between predicted "robot response" and observed history — about an hour of training on a single RTX 4090 produces robust quadruped behavior. For a Spot starting policy, DreamWaQ or HIM are simpler than full two-stage distillation and produce comparable results on flat-to-moderate terrain.

The capstone results are **Cheng et al. — *Extreme Parkour*, ICRA 2024** (arXiv 2309.14341, jumps over 2× body length / 2× body height) and **Hoeller et al. — *ANYmal Parkour*, Science Robotics 2024** (arXiv 2306.14874, hierarchical skill library + scene-aware selector). Both rely on teacher-student distillation from privileged scandots to a depth-camera student.

### Asymmetric actor-critic is now the default, not a technique

Pinto et al. (RSS 2018, arXiv 1710.06542) introduced it, but in 2024-2026 essentially every legged sim-to-real paper — DreamWaQ, HIM, ASAP, ExBody2, OmniH2O, HumanPlus, Berkeley Humanoid, MuJoCo Playground — uses a privileged critic as standard practice. The pattern: critic gets true base velocity, terrain heights, friction, mass, contact forces; actor sees only proprio. Recent theory papers (Lambrechts et al., ICML 2025; Cai et al., arXiv 2412.00985) provide regret bounds. **For your Spot project, asymmetric AC should be the baseline before any teacher-student stage.**

### Real-to-sim, system identification, and digital twins

Three credible directions: differentiable-sim sysid, distributional-fit sysid, and delta-action residuals. **Kovalev et al. — *Achieving Precise Locomotion via Differentiable Sim-Based SysID*, arXiv 2508.04696** uses MJX gradients to fit motor parameters from trajectory data with no torque sensors — directly applicable to your stack. **Miller et al. on Spot (arXiv 2504.17857)** uses CMA-ES against Wasserstein/MMD distances. **He, Gao, Xiao et al. — *ASAP: Aligning Simulation and Real-World Physics*, RSS 2025** (arXiv 2502.01143) is the most influential recent residual-correction approach: train a tracking policy in sim, collect a few dozen real rollouts on Unitree G1, learn a **delta-action residual model** that compensates dynamics mismatch, then fine-tune in sim augmented with the delta model. They report up to 52.7% reduction in tracking error vs SysID/DR baselines and demonstrate Cristiano-Ronaldo-style agile motions previously infeasible. **Confidence: Medium-High** for the technique; real-data requirement is modest, transfer back into the simulator is non-trivial to implement correctly.

For digital-twin construction with photorealistic rendering, the **Gaussian-splatting / Real2Sim2Real** line (RoboGSim arXiv 2411.11839, Robo-GS 2408.14873, RoboSimGS 2510.10637, S2GS 2512.04731) is genuinely novel for *manipulation* but dynamically irrelevant for proprio-driven legged locomotion — appearance fidelity is not your bottleneck. Skip unless adding vision.

Differentiable simulators (Brax, MJX, Dojo, Nimble) are excellent for sysid and parameter fitting but produce **noisy gradients through contacts** for direct policy learning (Suh & Tedrake, ICML 2022, arXiv 2202.00817). **Schwarke et al. — *DiffSim2Real*, arXiv 2411.02189 (CoRL 2024 workshop)** is the first published quadruped sim-to-real using SHAC analytic gradients with a smooth-but-accurate contact model — interesting but not yet a PPO replacement for production legged work. **Recommendation: use MJX gradients for sysid, PPO + parallelism for policy.**

### Massive parallelization is what made modern legged RL practical

**Rudin, Hoeller, Reist, Hutter — *Learning to Walk in Minutes*, CoRL 2022** (arXiv 2109.11978) is the recipe everyone uses: 4096 parallel ANYmal envs on one GPU + a game-inspired terrain curriculum that adapts difficulty per robot. Flat-terrain policy in under 4 minutes; uneven terrain in 20 minutes. Built on Isaac Gym (Makoviychuk et al., NeurIPS 2021, arXiv 2108.10470), now superseded by **Isaac Lab** (Mittal et al., arXiv 2301.04195; NVIDIA 2025 update arXiv 2511.04831).

For your **JAX/Flax + MJX** stack, the direct analog is **Zakka, Tabanpour, Liao, Haiderbhai et al. — *MuJoCo Playground*, RSS 2025 Outstanding Demo** (arXiv 2502.08844, playground.mujoco.org, github.com/google-deepmind/mujoco_playground). It deployed sim-to-real on Berkeley Humanoid, Unitree Go1 + G1, LEAP hand, and Franka **in under eight weeks** and includes a Spot environment with PPO recipe. **This is your highest-leverage starting point.** Pair with **Barkour** (Caluwaerts et al., arXiv 2305.14654) as a reference MJX quadruped baseline.

### Diffusion policies, world models, and real-world fine-tuning

**Diffusion Policy** (Chi et al., RSS 2023, arXiv 2303.04137) is dominant for imitation-learning manipulation but **not yet a strong choice for online RL of legged locomotion** — multi-step denoising is too slow at 50+ Hz. **DiffuseLoco** (Huang et al., CoRL 2024, arXiv 2404.19264) is the only credible legged diffusion paper to date: zero-shot sim-to-real on quadrupeds via offline multi-skill distillation from RL teachers, with delayed-input formulation; bipedal Cassie transfer was poor while quadruped Go1/Cyberdog was good. **Consistency Policy** (Prasad et al., RSS 2024, arXiv 2405.07503) and **FlowPolicy** (arXiv 2412.04987) distill diffusion into 1-step models for ~10× faster inference, which is what makes onboard deployment plausible. **DPPO** (Ren et al., arXiv 2409.00588) shows PPO-style fine-tuning of diffusion policies. Pragmatic verdict: skip diffusion for v1; revisit if you need offline multi-skill on Spot.

**World models** (DreamerV3, Hafner et al., Nature 2025, arXiv 2301.04104; DayDreamer, Wu et al., CoRL 2022, arXiv 2206.14176; TD-MPC2, Hansen et al., ICLR 2024, arXiv 2310.16828) are technically impressive but **have not displaced PPO + sim parallelism for legged locomotion**. DayDreamer learned A1 walking in ~1 hour of real-world experience, but the resulting gait is crude versus PPO-trained sim-to-real policies. Useful only if real-world sample efficiency matters more than peak performance.

For real-world fine-tuning, **SERL** (Luo et al., ICRA 2024, arXiv 2401.16013) and its human-in-the-loop sequel **HIL-SERL** (Luo et al., Science Robotics 2025, arXiv 2410.21845) are the current best manipulation pipelines — near-100% success on Jenga whipping, motherboard assembly, IKEA, timing-belt — but the techniques (RLPD off-policy, async actor/learner, demo bootstrapping) are portable. **Residual policy learning** (Silver et al. arXiv 1812.06298; Johannink et al. ICRA 2019, arXiv 1812.03201) remains a sound default when a hand-designed or analytical base controller exists.

### Architecture choices that empirically transfer

The pattern across nearly every successful 2023-2026 legged paper is striking and **almost boringly consistent**:

- **3-layer MLP actor [512, 256, 128]** with proprio history (5–25 steps), commands, and last action; output as joint position residuals around nominal stance scaled to ~0.25–0.5 rad, with low-level PD at 50–200 Hz.
- **Privileged MLP critic** with true base velocity, terrain heights, friction, mass, contact forces.
- **Action-rate penalty** and **left-right symmetry augmentation** (Abdolhosseini 2019, baked into legged_gym).

GRU-based recurrence helps when exteroception is added or when the proprio observation is too partial (Miki 2022, Cheng Extreme Parkour, Agarwal Egocentric Vision). Transformers win for **humanoids and whole-body control** where long history matters — see **Radosavovic et al., *Real-World Humanoid Locomotion with RL*, Science Robotics 2024** (DOI 10.1126/scirobotics.adi9579, arXiv 2303.03381) and the next-token-prediction sequel (NeurIPS 2024, arXiv 2402.19469). For quadruped joystick locomotion, **MLP with short history is consistently competitive with recurrent or transformer alternatives**, and is lighter on the deployment compute — relevant for Spot's onboard.

The 2024 humanoid wave — **HumanPlus** (Fu et al., CoRL 2024, arXiv 2406.10454), **OmniH2O / H2O** (He et al., CoRL 2024 / IROS 2024, arXiv 2406.08858, 2403.04436), **ExBody / ExBody2** (Cheng et al., RSS 2024 / 2024, arXiv 2402.16796, 2412.13196), **Humanoid-Gym** (Gu et al., arXiv 2404.05695), **Humanoid Parkour Learning** (Zhuang et al., CoRL 2024, arXiv 2406.10759) — converges on the same recipe: privileged teacher with mocap-retargeted reference, asymmetric AC, distill to proprio-only or sparse-input student, deploy with action chunking. **Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation** (Fey et al., MIT, RSS 2025) extends the pattern to manipulation+locomotion.

### A concrete pipeline for your Spot stack

Given that you're on PPO + JAX/Flax + MJX, the strongest plan, in dependency order, is:

1. **Start from MuJoCo Playground's quadruped joystick env**, port Spot's MJCF via Boston Dynamics' RL Researcher Dev Kit (Miller et al. 2025).
2. Train **asymmetric AC PPO with light DR** (mass ±10%, friction 0.4–1.2, motor strength ±15%, latency 5–20 ms, command curriculum from Rapid Locomotion).
3. Use **Rudin 2022 game-inspired terrain curriculum**.
4. Add an **RMA-style 1D-conv adaptation module** or **DreamWaQ-style concurrent estimator** for proprio-only deployment.
5. After zero-shot rollout works, run **Miller 2025's WD/MMD-based CMA-ES sysid** on the largest sim parameters (joint armature, damping, friction, delay — the PACE 4n+1 set).
6. If a residual gap persists, add **ASAP-style delta-action correction** from a few dozen real rollouts.
7. Avoid initially: world models, GAN-based domain adaptation, Gaussian-splat digital twins, diffusion policies. They don't pay off for proprio legged locomotion.

---

## 3) The "one model for all robots" question, honestly

**Short version: the strong claim is not yet supported by evidence. The architectural recipe scales; the trunk learns useful shared representations; but every published generalist still uses embodiment-specific input/output projectors, and none runs the high-frequency torque control legged hardware demands.** Below I trace the evidence.

### What's actually shared in the major generalist VLAs

The table summarizes architectural sharing across the canonical generalist policies. "Embodiment-specific" means different weights or modules per robot at deployment.

| Model | Trunk | State encoder | Action head | Embodiments evaluated |
|---|---|---|---|---|
| RT-1 (Brohan 2022, arXiv 2212.06817) | Shared (EfficientNet+Transformer ~35M) | n/a | Shared (256-bin discretization, 7-DoF EE) | **One** robot |
| RT-2 (arXiv 2307.15818) | Shared (PaLI-X 5B/55B) | n/a | Shared text-token actions | One robot |
| Open X / RT-X (arXiv 2310.08864, ICRA 2024 best paper) | Shared | n/a | Shared, **single-arm only** despite 22-embodiment dataset | 9 single-arm manipulators |
| Octo (arXiv 2405.12213, RSS 2024) | Shared (~93M trans.) | Shared | Diffusion shared at pretrain, **swapped on finetune** | 9 manipulators |
| OpenVLA (arXiv 2406.09246, CoRL 2024) | Shared (Llama-2 7B + SigLIP+DINOv2) | n/a | Shared 7-DoF EE token | 3 single-arm manipulators |
| π₀ (arXiv 2410.24164) | Shared (PaliGemma 3B) + 300M action expert | Shared (padded) | Shared flow-matching, up to ~24-D actions | 7 platforms — all arm/bimanual/mobile-base |
| π₀-FAST / π₀.₅ (arXiv 2501.09747, 2504.16054) | Shared | Shared | Shared FAST tokens / flow | Same arm family + new homes |
| CrossFormer (arXiv 2408.11812, CoRL 2024) | Shared (~130M) | Shared | **Embodiment-specific MLP heads** | 20 embodiments **including Go1 quadruped + Tello drone** |
| GR00T N1/N1.5 (arXiv 2503.14734, NVIDIA 2025) | Shared DiT + frozen Eagle-2 VLM | **Embodiment-specific MLP keyed by ID** | **Embodiment-specific MLP** | Bimanual humanoids + arms |
| RDT-1B (arXiv 2410.07864, ICLR 2025) | Shared DiT 1.2B | Shared | Diffusion over 128-D unified manipulator action space | Bimanual + arms only |
| HPT (arXiv 2409.20537, NeurIPS 2024 spotlight) | **Shared trunk only** | **Stems specific** | **Heads specific** | Many, but trunk is the only generalist part |

Two patterns stand out. First, **CrossFormer is the only single-network generalist that controls a quadruped, a drone, and arms with one set of weights** — and even there, "shared" means the trunk; each morphology gets its own MLP action head. Second, **GR00T's trunk is shared but state and action MLPs are explicitly indexed by embodiment ID**, which is architecturally identical to "separate models that share a backbone." HPT is the most honest about this: it explicitly markets itself as a generalist representation backbone, not a generalist policy.

### CrossFormer — the most relevant to your skeptical question

**Doshi, Walke, Mees, Dasari, Levine — *Scaling Cross-Embodied Learning*, CoRL 2024** (arXiv 2408.11812, crossformer-model.github.io) trained on 900k trajectories across 20 embodiments. The eye-catching result is **73% average success across all embodiments vs 67% for single-robot baselines**. The honest reading is more nuanced. The Go1 quadruped slice is **only 25 minutes of data**, the task is **walking forward**, the evaluation reward is normalized to the RL expert that generated the data, and the quadruped runs at the dataset's joint-position frequency (~20 Hz), **not the ~500 Hz–1 kHz torque control of a deployed locomotion stack**. The drone (Tello) slice is similarly small. The authors themselves are transparent about this. Their core claim — that cross-embodiment co-training does **no negative transfer** and modest positive transfer in low-data regimes — is well-supported. The stronger interpretation that "a single network can do quadruped locomotion plus manipulation plus drone flight" is technically true only at imitation-learning frequencies for behaviors already present in expert data.

### π₀ family — strongest dexterous evidence, narrow morphology range

**Black, Brown, Driess et al. (Physical Intelligence) — π₀, arXiv 2410.24164 (2024)** uses PaliGemma plus a separate ~300M action expert with block-wise causal attention, trained by flow matching on action chunks at 50 Hz. They evaluate on **7 platforms across 68 tasks** — UR5e, bimanual UR5e, Franka, bimanual Trossen (ALOHA-style), bimanual ARX/AgileX, mobile bimanual Trossen, mobile Fibocom. **All are arm-based; no quadruped, drone, or full-body humanoid.** π₀-FAST (arXiv 2501.09747) replaces the flow head with DCT+BPE-tokenized autoregressive output, claims a universal manipulator tokenizer, and trains 5× faster. π₀.₅ (arXiv 2504.16054) demonstrates cleaning entirely new homes — a meaningful generalization claim, though independent replication is sparse. **Confidence: High** for dexterous manipulation results within the arm family, **Low** for any cross-morphology transfer claim — the paper does not make one.

### GR00T and the humanoid foundation-model story

**NVIDIA — GR00T N1, arXiv 2503.14734 (2025), with N1.5 and N1.6 model cards on HuggingFace** is a dual-system VLA (Eagle-2 VLM at 10 Hz + DiT action expert at up to 120 Hz via 16-action chunking). Marketing positions it as "the first open foundation model for generalist humanoid robots." The technical reality is less unified. The published model controls **upper-body bimanual manipulation only**; whole-body humanoid behavior in NVIDIA's deployments uses a **separate decoupled RL controller for the lower body** plus inverse kinematics. This is acknowledged in the GR00T-WholeBodyControl release (Nov 2025) and the SONIC behavior model is **not** the same network as N1.5/N1.6 — combining them is a *system*, not a unified policy.

Two recent independent papers are revealing. **MOTIF (arXiv 2602.13764, 2026)** reports GR00T N1 cross-arm transfer success of 21.25% versus their motif-based 67.5% on a new manipulator — meaningful evidence GR00T zero-shots poorly to unseen morphologies. **Modality-Augmented Fine-Tuning of GR00T (arXiv 2512.01358, Dec 2025)** documents **0% zero-shot transfer from Fourier GR-1 to Unitree G1 humanoid**, rising only to ~48% with 5,000 demonstrations of fine-tuning. The "foundation model for any humanoid" framing does not survive contact with new humanoids without significant retraining. **Confidence: High** that the gap is real; **Medium** that fine-tuning closes it usefully.

### Where authors are most candid about limits

- **RDT-1B's HuggingFace model card** explicitly says: "Due to the embodiment gap, RDT cannot yet generalize to new robot platforms (not seen in the pre-training datasets)." Unusually honest.
- **Octo authors** explicitly title their paper "An Open-Source Generalist Robot **Policy**" but evaluate only on manipulators, and the action head is swapped on fine-tune.
- **CrossFormer authors** call out that quadruped evaluation is single-task forward walking on 25 minutes of data.
- **HPT authors** explicitly market a generalist *trunk*, not a generalist policy.
- **RT-X paper** acknowledges that on the largest in-domain data subsets (RT-1 robot, BridgeData), RT-1-X **underperforms** the in-domain specialist — direct evidence that more cross-embodiment data does not always help.

### The frequency wall

This is the single best argument that "one model for all robots" cannot be true today. Manipulation policies typically run at 5–50 Hz, with VLAs at 6–10 Hz on the language head. Quadruped torque control needs ~500 Hz–1 kHz; humanoid whole-body balance is usually 200–500 Hz. **No published generalist VLA runs natively at these frequencies.** Action chunking buys some headroom (π₀ at 50 Hz, GR00T DiT at 120 Hz), but the high-frequency stabilization layer on every legged robot in the literature is a separate, smaller network or a classical controller. CrossFormer's quadruped works because the data was 20 Hz joint positions, not torques. There is no architectural reason this can't change, but as of April 2026, it hasn't.

### Cross-embodiment locomotion specifically

Outside CrossFormer, the genuine cross-embodiment legged work is locomotion-only and modest in scope. **GROQLoco (arXiv 2505.10973, 2025)** is a generalist quadruped controller across Go1, Go2, Aliengo, Stoch5 trained on offline data — a useful proof-point for "one policy across multiple quadrupeds" but not across morphologies. **MXT (Modularized Cross-embodiment Transformer)** in the Frontiers 2025 imitation learning survey reports +38.6% success on quadrupedal-arm transfer from human demos with separate tokenizers per embodiment. These are real but narrow.

### Verdict on the user's hypothesis

Your prior — that "one model for all robots is not possible" — is partially right and partially too strong. Specifically: **One set of weights end-to-end across radically different morphologies, running at native control frequencies, is not yet possible** as of April 2026. **One trunk that learns useful shared representations with embodiment-specific I/O projectors** is possible and is what every successful generalist actually implements. The marketing language conflates these. The strong claim is unsupported; the weak claim is a real and useful research result. For locomotion specifically, the embodiment-specific layer matters disproportionately because high-frequency dynamics and contact stabilization is exactly where morphology cannot be abstracted away.

---

## Conclusion: what to take home

The most reliable bet for your Spot project is the conservative one: **MuJoCo Playground's PPO recipe with asymmetric AC, RMA- or DreamWaQ-style adaptation, light DR, terrain curriculum, and Miller-2025-style distributional sysid against real Spot rollouts**. Miller et al. is your single highest-priority read; PACE, ASAP, and the Aljalbout reality-gap survey are next. Skip world models, GAN domain adaptation, diffusion policies, and Gaussian-splat digital twins for v1 — they do not pay off for proprio-driven legged locomotion.

The genuinely novel research insight of the last two years is not from the generalist VLAs — it's from **delta-action residual models** (ASAP) and **minimal-parameter dynamics fitting** (PACE), both of which suggest that the legged sim-to-real gap can be closed with **minutes of real data and the right parameterization**, rather than with ever-wider domain randomization or ever-larger foundation models. That's an actionable result.

The cross-embodiment story is genuinely exciting and genuinely oversold, in roughly equal proportion. What is real: shared transformer trunks, modest positive transfer in low-data regimes, scaling laws for representation learning across morphologies. What is not real yet: one set of weights running quadruped torque control, drone aerodynamics, and manipulator end-effector planning at native frequencies. Treat any "foundation model for robots" claim by checking three things: (1) what action heads are shared vs swapped, (2) what control frequencies are run end-to-end, and (3) whether legged dynamics are inside the foundation model or in a separate controller. The third is the tell.
