# 2-Clause BSD License

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from pathlib import Path

import coal
import numpy as np
import pinocchio as pin

# This class is for unwrapping an URDF and converting it to a model. It is also possible
# to add objects in the model,
# such as a ball at a specific position.


class PandaWrapper:
    def __init__(
        self,
        auto_col=False,
        capsule=False,
        obstacles=None,
    ):
        """
        Initialize the wrapper with a scaling number of the target and the name of the
        robot wanted to get unwrapped.
        """

        # Importing the model
        model_path = Path(__file__).parent.parent / "examples" / "models"
        self._urdf_model_path = model_path / "urdf" / "franka2.urdf"
        self._mesh_dir = model_path
        self._srdf_model_path = model_path / "srdf" / "demo.srdf"

        # Color of the robot
        self._color = np.array([249, 136, 126, 255]) / 255

        # Boolean describing whether the auto-collisions are in the collision model or
        # not
        self._auto_col = auto_col

        # Transforming the robot from cylinders/spheres to capsules
        self._capsule = capsule

        # Dict describing the obstacles andd their positions in the world's frame
        self._obstacles = obstacles

    def __call__(self):
        """Create a robot.

        Returns
        -------
        _rmodel
            Model of the robot
        _gmodel
            Geometrical model of the robot
        _gmodel
            Visual model of the robot


        """
        (
            self._rmodel,
            self._collision_model,
            self._visual_model,
        ) = pin.buildModelsFromUrdf(
            self._urdf_model_path, self._mesh_dir, pin.JointModelFreeFlyer()
        )

        q0 = pin.neutral(self._rmodel)

        jointsToLockIDs = [1, 9, 10]

        geom_models = [self._visual_model, self._collision_model]
        self._model_reduced, geometric_models_reduced = pin.buildReducedModel(
            self._rmodel,
            list_of_geom_models=geom_models,
            list_of_joints_to_lock=jointsToLockIDs,
            reference_configuration=q0,
        )

        self._visual_model_reduced, self._collision_model_reduced = (
            geometric_models_reduced[0],
            geometric_models_reduced[1],
        )

        # Modifying the collision model to add the capsules
        rdata = self._model_reduced.createData()
        cdata = self._collision_model_reduced.createData()
        q0 = pin.neutral(self._model_reduced)

        # Updating the models
        pin.framesForwardKinematics(self._model_reduced, rdata, q0)
        pin.updateGeometryPlacements(
            self._model_reduced, rdata, self._collision_model_reduced, cdata, q0
        )

        # Adding the capsules to the collision model

        collision_model_reduced_copy = self._collision_model_reduced.copy()

        # # Replacing the cylinders by capsules
        if self._capsule:
            list_names_capsules = []
            for i, geometry_object in enumerate(
                collision_model_reduced_copy.geometryObjects
            ):
                if isinstance(geometry_object.geometry, coal.Sphere):
                    self._collision_model_reduced.removeGeometryObject(
                        geometry_object.name
                    )
                # Only selecting the cylinders
                if isinstance(geometry_object.geometry, coal.Cylinder):
                    if (geometry_object.name[:-4] + "capsule") in list_names_capsules:
                        capsule = pin.GeometryObject(
                            geometry_object.name[:-4] + "capsule" + "1",
                            geometry_object.parentJoint,
                            geometry_object.parentFrame,
                            coal.Capsule(
                                geometry_object.geometry.radius,
                                geometry_object.geometry.halfLength,
                            ),
                            geometry_object.placement,
                        )
                        capsule.meshPath = "CAPSULE"
                        self._collision_model_reduced.addGeometryObject(capsule)
                        self._collision_model_reduced.removeGeometryObject(
                            geometry_object.name
                        )
                        list_names_capsules.append(
                            geometry_object.name[:-4] + "capsule" + "1"
                        )
                    else:
                        capsule = pin.GeometryObject(
                            geometry_object.name[:-4] + "capsule",
                            geometry_object.parentJoint,
                            geometry_object.parentFrame,
                            coal.Capsule(
                                geometry_object.geometry.radius,
                                geometry_object.geometry.halfLength,
                            ),
                            geometry_object.placement,
                        )
                        capsule.meshPath = "CAPSULE"
                        self._collision_model_reduced.addGeometryObject(capsule)
                        self._collision_model_reduced.removeGeometryObject(
                            geometry_object.name
                        )
                        list_names_capsules.append(
                            geometry_object.name[:-4] + "capsule"
                        )

            # Removing the geometry objects that aren't Capsule / Box and disabling the
            # collisions for the finger and the camera
            for geometry_object in self._collision_model_reduced.geometryObjects:
                # Disabling the collisions for the fingers
                if (
                    "finger" in geometry_object.name
                    or "camera" in geometry_object.name
                    or "support" in geometry_object.name
                ):
                    geometry_object.disableCollision = True
                # Getting rid of the cylinders in cmodel
                if isinstance(geometry_object.geometry, coal.Cylinder):
                    self._collision_model_reduced.removeGeometryObject(
                        geometry_object.name
                    )

            if self._obstacles is not None:
                for name, obstacle_info in self._obstacles.items():
                    shape = obstacle_info[0]
                    pose = obstacle_info[1]
                    obstacle = pin.GeometryObject(name, 0, 0, shape, pose)
                    self._collision_model_reduced.addGeometryObject(obstacle)

        if self._auto_col:
            self._collision_model_reduced.addCollisionPair(
                pin.CollisionPair(
                    self._collision_model_reduced.getGeometryId("panda2_leftfinger_0"),
                    self._collision_model_reduced.getGeometryId(
                        "panda2_link6_capsule22"
                    ),
                )
            )
            self._collision_model_reduced.addCollisionPair(
                pin.CollisionPair(
                    self._collision_model_reduced.getGeometryId("panda2_leftfinger_0"),
                    self._collision_model_reduced.getGeometryId(
                        "panda2_link6_capsule22"
                    ),
                )
            )

        pin.removeCollisionPairs(
            self._model_reduced, self._collision_model_reduced, self._srdf_model_path
        )

        return (
            self._model_reduced,
            self._collision_model_reduced,
            self._visual_model_reduced,
        )


def compute_distance_between_shapes(
    rmodel: pin.Model, cmodel: pin.Model, shape1_id: int, shape2_id: int, q: np.ndarray
) -> float:
    """Computes the distance between shapes given the configuration vector.

    Args:
        rmodel (pin.Model): model of the robot
        cmodel (pin.Model): collision model of the robot
        shape1_id (int): id of the shape 1 in the collision model
        shape2_id (int): id of the shape 2 in the collision model
        q (np.ndarray): configuration array of the robot

    Returns:
        float: distance between the two given shapes
    """
    rdata = rmodel.createData()
    cdata = cmodel.createData()
    pin.forwardKinematics(rmodel, rdata, q)
    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    req = coal.DistanceRequest()
    res = coal.DistanceResult()

    distance = coal.distance(
        cmodel.geometryObjects[shape1_id].geometry,
        cdata.oMg[shape1_id],
        cmodel.geometryObjects[shape2_id].geometry,
        cdata.oMg[shape2_id],
        req,
        res,
    )

    return distance
