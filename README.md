# mujoco-animator

Welcome to the mujoco-animator project!

This utility can be used to generate animations of Mujoco models, which in turn can be used for training robots to do bespoke behaviors.

## Getting Started

If you just want to use recorded animations to train your model, install the base package:

```bash
pip install mujoco-animator
```

If you want to start recording animations using the CLI, install with:

```bash
pip install 'mujoco-animator[cli]'
```

Then you can run

```bash
mujoco-animator /path/to/your/robot.mjcf
```
