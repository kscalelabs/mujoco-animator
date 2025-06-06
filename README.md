# mujoco-animator

Welcome to the mujoco-animator project!

This utility can be used to generate animations of Mujoco models, which in turn can be used for training robots to do bespoke behaviors.

## Getting Started

### File Format

If you just want to use recorded animations to train your model, install the base package:

```bash
pip install mujoco-animator
```

You can then load the `MjAnim` file format, which exposes a nice helper method to get per-frame qpos targets:

```python
from mujoco_animator import MjAnim

anim = MjAnim.load("/path/to/file.mjanim")
qpos_sequence = anim.to_numpy(dt, interp="cubic", loop=True)
```

### CLI

If you want to start recording animations using the CLI, install with:

```bash
pip install 'mujoco-animator[cli]'
```

Then you can run

```bash
mujoco-animator /path/to/your/robot.mjcf
```

You may provide a desired output file name with `--output`. The output is dynamic depending on the filetype. Currently supported:
* `.mjanim` -> Condensed Binary
* `.json` -> JSON Dictionary
