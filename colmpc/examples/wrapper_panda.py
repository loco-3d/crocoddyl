# BSD 3-Clause License
#
# Copyright (C) 2024, LAAS-CNRS.
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.
from pathlib import Path

import coal
import numpy as np
import pinocchio as pin

# This class is for unwrapping an URDF and converting it to a model. It is also possible
# to add objects in the model, such as a ball at a specific position.

RED = np.array([249, 136, 126, 125]) / 255.0


class PandaWrapper:
    def __init__(
        self,
        auto_col: bool = False,
        capsule: bool = False,
    ) -> None:
        """Create a wrapper for the robot panda.

        Args:
            auto_col (bool, optional): Include the auto collision in the collision
                model. Defaults to False.
            capsule (bool, optional): Transform the spheres and cylinder of the robot
                into capsules. Defaults to False.
        """

        # Importing the model
        pinocchio_model_dir = Path(__file__).parent
        model_path = pinocchio_model_dir / "models"
        self._mesh_dir = model_path / "meshes"
        urdf_filename = "franka2.urdf"
        srdf_filename = "demo.srdf"
        self._urdf_model_path = model_path / "urdf" / urdf_filename
        self._srdf_model_path = model_path / "srdf" / srdf_filename

        # Color of the robot
        self._color = np.array([249, 136, 126, 255]) / 255.0

        # Boolean describing whether the auto-collisions are in the collision model or
        # not
        self._auto_col = auto_col

        # Transforming the robot from cylinders/spheres to capsules
        self._capsule = capsule

    def __call__(self) -> tuple[pin.Model, pin.GeometryModel, pin.GeometryModel]:
        """Create a robot.

        Returns:
            rmodel (pin.Model): Model of the robot
            cmodel (pin.GeometryModel): Collision model of the robot
            vmodel (pin.GeometryModel): Visual model of the robot
        """
        (
            self._rmodel,
            self._cmodel,
            self._vmodel,
        ) = pin.buildModelsFromUrdf(
            self._urdf_model_path, self._mesh_dir, pin.JointModelFreeFlyer()
        )

        q0 = pin.neutral(self._rmodel)

        # Locking the gripper
        jointsToLockIDs = [1, 9, 10]

        geom_models = [self._vmodel, self._cmodel]
        self._model_reduced, geometric_models_reduced = pin.buildReducedModel(
            self._rmodel,
            list_of_geom_models=geom_models,
            list_of_joints_to_lock=jointsToLockIDs,
            reference_configuration=q0,
        )

        self._vmodel_reduced, self._cmodel_reduced = (
            geometric_models_reduced[0],
            geometric_models_reduced[1],
        )

        # Modifying the collision model to transform the spheres/cylinders into capsules
        if self._capsule:
            self.transform_model_into_capsules()

        # Adding the auto-collisions in the collision model if required
        if self._auto_col:
            self._cmodel_reduced.addAllCollisionPairs()
            pin.removeCollisionPairs(
                self._model_reduced, self._cmodel_reduced, self._srdf_model_path
            )

        rdata = self._model_reduced.createData()
        cdata = self._cmodel_reduced.createData()
        q0 = pin.neutral(self._model_reduced)

        # Updating the models
        pin.framesForwardKinematics(self._model_reduced, rdata, q0)
        pin.updateGeometryPlacements(
            self._model_reduced, rdata, self._cmodel_reduced, cdata, q0
        )

        return (
            self._model_reduced,
            self._cmodel_reduced,
            self._vmodel_reduced,
        )

    def transform_model_into_capsules(self) -> None:
        """
        Modifying the collision model to transform the spheres/cylinders into capsules
        which makes it easier to have a fully constrained robot.
        """
        collision_model_reduced_copy = self._cmodel_reduced.copy()
        list_names_capsules = []

        # Going through all the goemetry objects in the collision model
        for geom_object in collision_model_reduced_copy.geometryObjects:
            if isinstance(geom_object.geometry, coal.Cylinder):
                # Sometimes for one joint there are two cylinders,
                # which need to be defined by two capsules for the same link.
                # Hence the name convention here.
                if (geom_object.name[:-4] + "capsule_0") in list_names_capsules:
                    name = geom_object.name[:-4] + "capsule_1"
                else:
                    name = geom_object.name[:-4] + "capsule_0"
                list_names_capsules.append(name)
                placement = geom_object.placement
                parentJoint = geom_object.parentJoint
                parentFrame = geom_object.parentFrame
                geometry = geom_object.geometry
                geom = pin.GeometryObject(
                    name,
                    parentFrame,
                    parentJoint,
                    coal.Capsule(geometry.radius, geometry.halfLength),
                    placement,
                )
                geom.meshColor = RED
                self._cmodel_reduced.addGeometryObject(geom)
                self._cmodel_reduced.removeGeometryObject(geom_object.name)
            elif (
                isinstance(geom_object.geometry, coal.Sphere)
                and "link" in geom_object.name
            ):
                self._cmodel_reduced.removeGeometryObject(geom_object.name)
