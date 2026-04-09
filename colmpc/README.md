# COLMPC: Collision Avoidance for MPC

This repo is addon residuals for Crocoddyl for collision avoidance for trajectory optimisation and model predictive control (MPC).
It is composed of two different constraints:
- The first one is **ResidualDistanceCollision**, defined in depths in this [paper](https://gepettoweb.laas.fr/articles/haffemayer2024.html).
Simply, it is the distance between the closest points of the two objects in the collision pair given in input of the residual.
- The second one is **ResidualModelVelocityAvoidance**, defined in depths in this [paper](https://gepettoweb.laas.fr/articles/haffemayer2025.html).
Not only this residual takes the distance between the closest points of the two objects but their approach speed toward each other as well. However, this second residual only works on ellipsoids and spheres for now.

An in-depth comparison is [here](https://gepettoweb.laas.fr/articles/haffemayer2025.html) and a practical comparison is provided here along 3 different scenarios.

## Dependencies

### For OCP & MPC scripts:

- [Coal](https://github.com/coal-library/coal)  (tag: v3.0.0) for collision computations.
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) (tag: v3.3.0) fast rigid body dynamics.
- [Crocoddyl](https://github.com/loco-3d/crocoddyl) (tag: v2.1.0) framework for the solver.

### For visualization:
- [Meshcat](https://github.com/meshcat-dev/meshcat-python).

### For the examples:

- [MiM Solvers](https://github.com/machines-in-motion/mim_solvers) (tag: v0.0.5) solver for the SQP and Constrained-SQP solver, and [Mim Robot](https://github.com/machines-in-motion/mim_robots/tree/main).

## Installation
### From source
Coal & Pinocchio must be built from sources. Build pinocchio with the flag : WITH_COLLISION_SUPPORT=ON.
> [!NOTE]
> Don't forget to switch to the right commits!

### Using Docker

You can run examples with docker with following command:
``` bash
docker container run -it --rm -p 7000:7000 ghcr.io/agimus-project/colmpc:v0.2.0 python colmpc/examples/main_ocp.py --scene 1
```


### Possible issue
- If you have a problem with ```FakeCollisionGeometry```, it is likely that the linking of Pinocchio with Coal wasn't done properly. Verify that you have the right commits & the right compilation flags.
- The main branch of Coal doesn't compute well the closest points and thus, this repo needs to be built upon the devel branch. If it built but doesn't avoid collision, make sure that you didn't built the main branch.

### For the OCP part:

To see the different scenarios with collision avoidance simply run in the main directory ```python examples/main_ocp.py -s i```, where i is the index of the scenario, going from 1 to 3.


As the code is still in developpement, the code is constantly moving and sometimes, examples do not work. Hence, do not hesitate to contact me at [ahaffemaye@laas.fr](mailto:ahaffemaye@laas.fr).

# Citation
To cite **COLMPC** in your academic research, please use the following bibtex entry:
```bibtex
@inproceedings{haffemayer_model_2024,
	title = {Model predictive control under hard collision avoidance constraints for a robotic arm},
	author = {Haffemayer, Arthur and Jordana, Armand and Fourmy, Médéric and Wojciechowski, Krzysztof and Saurel, Guilhem and Petrík, Vladimír and Lamiraux, Florent and Mansard, Nicolas},
    booktitle={Ubiquitous Robots (UR)}
	year = {2024},
}

@unpublished{haffemayer:hal-04707324,
  TITLE = {{Collision Avoidance in Model Predictive Control using Velocity Damper}},
  AUTHOR = {Haffemayer, Arthur and Jordana, Armand and de Matte{\"i}s, Ludovic and Wojciechowski, Krzysztof and Lamiraux, Florent and Mansard, Nicolas},
  URL = {https://laas.hal.science/hal-04707324},
  NOTE = {working paper or preprint},
  YEAR = {2024},
  MONTH = Sep,
  PDF = {https://laas.hal.science/hal-04707324v1/file/ICRA_2025__1_-11.pdf},
  HAL_ID = {hal-04707324},
  HAL_VERSION = {v1},
}
```
