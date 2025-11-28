# Intentional/Incremental/Inferential Continual Reinforcement Learning - ICRL

This repository provides the core reinforcement learning framework used across all CRoSS benchmark environments.
It contains the training, evaluation, and control infrastructure required to run continual reinforcement learning (CRL) experiments in both kinematic and physically simulated robotic settings.

The framework is centered around the RLAgent, which orchestrates the full CRL loop: interacting with a benchmark environment, managing task transitions, applying the selected learning algorithm, and collecting feedback for continual improvement.
Environments from any CRoSS benchmark plug directly into this agent, enabling a unified training pipeline across all robotic scenarios.
However each tool in this repository can be used independently in other frameworks or contexts as well.

The repository includes:

-   Apptainer definition file icrl.def to build an apptainer image (.sif) file. It's like docker but better ;)
-   Gazebo model (.sdf) files for a two-wheeled robot in two variants
-   Gazebo model (.sdf) files for a robotic arm
-   Python code for a CRL framework we use for our experiments.
    -   RLAgent and RLPGAgent for controlling the experiments
    -   Currently implementions for Q-learning and REINFORCE only
    -   Different exploration strategies
    -   Different Neural Network implementations
    -   Other useful utils such as a replay buffer

This repository serves as the shared backbone for all CRoSS benchmarks, providing a consistent API, reliable tooling, and a robust foundation for developing and evaluating continual reinforcement learning algorithms.

## Creating a .sif file

This may be a rather lengthy process (approx. 10 minutes). It involves loading a base image from docker hub and installing all required software via a script:

```bash
apptainer build icrl.sif icrl.def
```

or

```
singularity build icrl.sif icrl.def
```

depending on what you want to use.

## Related Repositories
* [**CRoSS** - Entry Repository](https://github.com/anon-scientist/continual-robotic-simulation-suite/)
* [CL_Experiments - Utils Repository](https://github.com/anon-scientist/cl_experiment/)
