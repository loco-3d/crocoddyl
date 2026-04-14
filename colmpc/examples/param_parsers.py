# BSD 3-Clause License
#
# Copyright (C) 2024, LAAS-CNRS.
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.

from pathlib import Path

import coal
import numpy as np
import numpy.typing as npt
import pinocchio as pin
import yaml


class ParamParser:
    def __init__(self, path: str | Path, scene: int) -> None:
        self.path = Path(path)
        self.params = None
        self.scene = scene

        with self.path.open() as stream:
            try:
                self.params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.data = self.params[f"scene{self.scene}"]

    @staticmethod
    def _parse_obstacle_shape(shape: str, size: list) -> coal.CollisionGeometry:
        if shape == "box":
            return coal.Box(*size)
        elif shape == "sphere":
            return coal.Sphere(size[0])
        elif shape == "cylinder":
            return coal.Cylinder(size[0], size[1])
        elif shape == "ellipsoid":
            return coal.Ellipsoid(*size)
        else:
            raise ValueError(f"Unknown shape {shape}")

    def add_ellipsoid_on_robot(
        self, rmodel: pin.Model, cmodel: pin.GeometryModel
    ) -> pin.GeometryModel:
        """Add ellipsoid on the robot model

        Args:
            rmodel (pin.Model): Robot model
            cmodel (pin.GeometryModel): Collision model

        Returns:
            cmodel (pin.GeometryModel): Collision model with added ellipsoids
        """
        if "ROBOT_ELLIPSOIDS" in self.data:
            for ellipsoid in self.data["ROBOT_ELLIPSOIDS"]:
                rob_coal = coal.Ellipsoid(
                    *self.data["ROBOT_ELLIPSOIDS"][ellipsoid]["dim"]
                )
                idf_rob = rmodel.getFrameId(
                    self.data["ROBOT_ELLIPSOIDS"][ellipsoid]["parentFrame"]
                )
                print(idf_rob)
                idj_rob = rmodel.frames[idf_rob].parentJoint
                if (
                    "translation" in self.data["ROBOT_ELLIPSOIDS"][ellipsoid]
                    and "orientation" in self.data["ROBOT_ELLIPSOIDS"][ellipsoid]
                ):
                    rot_mat = (
                        pin.Quaternion(
                            *tuple(
                                self.data["ROBOT_ELLIPSOIDS"][ellipsoid]["orientation"]
                            )
                        )
                        .normalized()
                        .toRotationMatrix()
                    )
                    Mrob = pin.SE3(
                        rot_mat,
                        np.array(
                            self.data["ROBOT_ELLIPSOIDS"][ellipsoid]["translation"]
                        ),
                    )
                else:
                    Mrob = rmodel.frames[idf_rob].placement
                rob_geom = pin.GeometryObject(
                    ellipsoid, idj_rob, idf_rob, Mrob, rob_coal
                )
                rob_geom.meshColor = np.array([1, 1, 0, 1])
                cmodel.addGeometryObject(rob_geom)
        return cmodel

    def add_collisions(
        self, rmodel: pin.Model, cmodel: pin.GeometryModel
    ) -> pin.GeometryModel:
        """Add collisions to the robot model

        Args:
            rmodel (pin.Model): Robot model
            cmodel (pin.GeometryModel): Collision model

        Returns:
            cmodel (pin.GeometryModel): Collision model with added collisions
        """
        cmodel = self.add_ellipsoid_on_robot(rmodel, cmodel)
        rng = np.random.default_rng()
        for obs in self.data["OBSTACLES"]:
            obs_coal = self._parse_obstacle_shape(
                self.data["OBSTACLES"][obs]["type"], self.data["OBSTACLES"][obs]["dim"]
            )
            Mobs = pin.SE3(
                pin.Quaternion(*tuple(self.data["OBSTACLES"][obs]["orientation"]))
                .normalized()
                .toRotationMatrix(),
                np.array(self.data["OBSTACLES"][obs]["translation"]),
            )
            obs_id_frame = rmodel.addFrame(pin.Frame(obs, 0, 0, Mobs, pin.OP_FRAME))
            obs_geom = pin.GeometryObject(
                obs, 0, obs_id_frame, rmodel.frames[obs_id_frame].placement, obs_coal
            )
            obs_geom.meshColor = np.concatenate((rng.integers(0, 1, 3), np.ones(1)))
            cmodel.addGeometryObject(obs_geom)

        for col in self.data["collision_pairs"]:
            if cmodel.existGeometryName(col[0]) and cmodel.existGeometryName(col[1]):
                cmodel.addCollisionPair(
                    pin.CollisionPair(
                        cmodel.getGeometryId(col[0]),
                        cmodel.getGeometryId(col[1]),
                    )
                )
            else:
                raise ValueError(
                    f"Collision pair {col} does not exist in the collision model"
                )
        return cmodel

    def get_target_pose(self) -> pin.SE3:
        return pin.SE3(
            pin.Quaternion(
                *tuple(self.data["TARGET_POSE"]["orientation"])
            ).toRotationMatrix(),
            np.array(self.data["TARGET_POSE"]["translation"]),
        )

    def get_initial_config(self) -> npt.NDArray[np.float64]:
        return np.array(self.data["INITIAL_CONFIG"])

    def get_X0(self) -> npt.NDArray[np.float64]:
        return np.concatenate(
            (self.get_initial_config(), np.array(self.data["INITIAL_VELOCITY"]))
        )

    def get_safety_threshold(self) -> float:
        return float(self.data["SAFETY_THRESHOLD"])

    def get_distance_threshold(self) -> float:
        return float(self.data["DISTANCE_THRESHOLD"])

    def get_T(self) -> int:
        return int(self.data["T"])

    def get_dt(self) -> float:
        return float(self.data["dt"])

    def get_di(self) -> float:
        return float(self.data["di"])

    def get_ds(self) -> float:
        return float(self.data["ds"])

    def get_ksi(self) -> float:
        return float(self.data["ksi"])

    def get_W_xREG(self) -> float:
        return float(self.data["WEIGHT_xREG"])

    def get_W_uREG(self) -> float:
        return float(self.data["WEIGHT_uREG"])

    def get_W_gripper_pose(self) -> float:
        return float(self.data["WEIGHT_GRIPPER_POSE"])

    def get_W_gripper_pose_term(self) -> float:
        return float(self.data["WEIGHT_GRIPPER_POSE_TERM"])

    def get_W_obstacle(self) -> float:
        return float(self.data["WEIGHT_OBSTACLE"])


if __name__ == "__main__":
    from wrapper_panda import PandaWrapper

    # Creating the robot
    robot_wrapper = PandaWrapper(capsule=True)
    rmodel, cmodel, vmodel = robot_wrapper()

    path = "scenes.yaml"
    scene = 0
    pp = ParamParser(path, scene)
    cmodel = pp.add_collisions(rmodel, cmodel)
