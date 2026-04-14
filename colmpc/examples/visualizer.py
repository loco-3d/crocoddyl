# BSD 3-Clause License
#
# Copyright (C) 2024, LAAS-CNRS.
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.


import meshcat
import numpy as np
import numpy.typing as npt
import pinocchio as pin
from pinocchio import visualize

RED = np.array([249, 136, 126, 125]) / 255.0
RED_FULL = np.array([249, 136, 126, 255]) / 255.0

GREEN = np.array([170, 236, 149, 125]) / 255.0
GREEN_FULL = np.array([170, 236, 149, 255]) / 255.0

BLUE = np.array([144, 169, 183, 125]) / 255.0
BLUE_FULL = np.array([144, 169, 183, 255]) / 255.0

YELLOW = np.array([1.0, 1.0, 0.0, 0.5])
YELLOW_FULL = np.array([1.0, 1.0, 0.0, 1.0])

BLACK = np.array([0.0, 0.0, 0.0, 0.5])
BLACK_FULL = np.array([0.0, 0.0, 0.0, 1.0])


def create_viewer(
    rmodel: pin.Model, cmodel: pin.GeometryModel, vmodel: pin.GeometryModel
) -> visualize.MeshcatVisualizer:
    """Create a Meshcat visualizer.

    Args:
        rmodel (pin.Model): Robot model.
        cmodel (pin.GeometryModel): Collision model.
        vmodel (pin.GeometryModel): Visual model.

    Returns:
        visualize.MeshcatVisualizer: The Meshcat visualizer.
    """
    viz = visualize.MeshcatVisualizer(
        model=rmodel,
        collision_model=cmodel,
        visual_model=vmodel,
    )
    viz.initViewer(open=True)
    viz.clean()
    viz.loadViewerModel("pinocchio")

    viz.displayCollisions(True)
    return viz


def add_sphere_to_viewer(
    viz: visualize.MeshcatVisualizer,
    sphere_name: str,
    radius: float,
    position: list | npt.NDArray[np.float64],
    color: int,
) -> None:
    """
    Adds a sphere to the Meshcat visualizer.

    Parameters:
    viz (MeshcatVisualizer): The Meshcat visualizer.
    sphere_name (str): The name of the sphere.
    radius (float): The radius of the sphere.
    position (list or np.array): The position of the sphere as [x, y, z].
    color (int): The color of the sphere as ?.
    """
    try:
        # Check if the sphere already exists
        viz.viewer[sphere_name]
        viz.viewer[sphere_name].delete()
    except KeyError:
        # Sphere does not exist
        pass
    sphere_geom = meshcat.geometry.Sphere(radius)
    sphere_material = meshcat.geometry.MeshLambertMaterial(color=color)

    viz.viewer[sphere_name].set_object(sphere_geom, sphere_material)
    viz.viewer[sphere_name].set_transform(
        meshcat.transformations.translation_matrix(position)
    )


def add_cube_to_viewer(
    viz: visualize.MeshcatVisualizer,
    cube_name: str,
    dim: list | npt.NDArray[np.float64],
    position: list | npt.NDArray[np.float64],
    color: int,
) -> None:
    """
    Adds a sphere to the Meshcat visualizer.

    Parameters:
    viz (MeshcatVisualizer): The Meshcat visualizer.
    cube_name (str): The name of the sphere.
    radius (float): The radius of the sphere.
    position (list or np.array): The position of the sphere as [x, y, z].
    color (int): The color of the sphere as ?.
    """
    try:
        # Check if the sphere already exists
        viz.viewer[cube_name]
        viz.viewer[cube_name].delete()
    except KeyError:
        # Sphere does not exist
        pass
    sphere_geom = meshcat.geometry.Box(dim)
    sphere_material = meshcat.geometry.MeshBasicMaterial(color=color)

    viz.viewer[cube_name].set_object(sphere_geom, sphere_material)
    viz.viewer[cube_name].set_transform(
        meshcat.transformations.translation_matrix(position)
    )
