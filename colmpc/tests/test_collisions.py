import unittest

import coal
import numpy as np
import pinocchio as pin


class TestCollisions(unittest.TestCase):
    """This class is made to test the collisions between primitives pairs such as
    sphere-sphere. The collisions shapes are from coal."""

    def test_sphere_sphere_not_in_collision(self):
        """
        Testing the sphere-sphere pair, going from the distance between each shape to
        making sure the closest points are well computed.
        """

        r1 = 0.4
        r2 = 0.5

        rmodel = pin.Model()
        cmodel = pin.GeometryModel()
        geometries = [
            coal.Sphere(r1),
            coal.Sphere(r2),
        ]
        # With pinocchio3, a new way of constructing a geometry object is available and
        # the old one will be deprecated.
        for i, geom in enumerate(geometries):
            placement = pin.SE3(np.eye(3), np.array([i, 0, 0]))
            try:
                geom_obj = pin.GeometryObject(f"obj{i}", 0, 0, placement, geom)
            except:  # noqa 422 TODO
                geom_obj = pin.GeometryObject(f"obj{i}", 0, 0, geom, placement)
            cmodel.addGeometryObject(geom_obj)

        rdata = rmodel.createData()
        cdata = cmodel.createData()

        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)
        # Creating the shapes for the collision detection.
        shape1_id = cmodel.getGeometryId("obj0")

        # Coloring the sphere
        shape1 = cmodel.geometryObjects[shape1_id]
        self.assertIsInstance(shape1.geometry, coal.Sphere)

        # Getting its pose in the world reference
        shape1_placement = cdata.oMg[shape1_id]

        # Doing the same for the second shape.
        shape2_id = cmodel.getGeometryId("obj1")

        # Coloring the sphere
        shape2 = cmodel.geometryObjects[shape2_id]
        self.assertIsInstance(shape2.geometry, coal.Sphere)

        # Getting its pose in the world reference
        shape2_placement = cdata.oMg[shape2_id]

        req = coal.DistanceRequest()
        res = coal.DistanceResult()

        # Testing the distance calculus
        distance_hpp = coal.distance(
            shape1.geometry,
            coal.Transform3s(shape1_placement.rotation, shape1_placement.translation),
            shape2.geometry,
            coal.Transform3s(shape2_placement.rotation, shape2_placement.translation),
            req,
            res,
        )

        distance_ana = np.linalg.norm(
            shape1_placement.translation - shape2_placement.translation
        ) - (r1 + r2)
        self.assertAlmostEqual(distance_ana, distance_hpp)

        # Testing the computation of closest points
        cp1 = res.getNearestPoint1()
        cp2 = res.getNearestPoint2()

        self.assertAlmostEqual(np.linalg.norm(cp1 - cp2), distance_ana)

    def test_sphere_sphere_in_collision(self):
        """
        Testing the sphere-sphere pair, going from the distance between each shape to
        making sure the closest points are well computed.
        """

        r1 = 0.7
        r2 = 0.5

        rmodel = pin.Model()
        cmodel = pin.GeometryModel()
        geometries = [
            coal.Sphere(r1),
            coal.Sphere(r2),
        ]

        # With pinocchio3, a new way of constructing a geometry object is available and
        # the old one will be deprecated.
        for i, geom in enumerate(geometries):
            placement = pin.SE3(np.eye(3), np.array([i, 0, 0]))
            try:
                geom_obj = pin.GeometryObject(f"obj{i}", 0, 0, placement, geom)
            except:  # noqa 422 TODO
                geom_obj = pin.GeometryObject(f"obj{i}", 0, 0, geom, placement)

            cmodel.addGeometryObject(geom_obj)

        rdata = rmodel.createData()
        cdata = cmodel.createData()

        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)
        # Creating the shapes for the collision detection.
        shape1_id = cmodel.getGeometryId("obj0")

        # Coloring the sphere
        shape1 = cmodel.geometryObjects[shape1_id]
        self.assertIsInstance(shape1.geometry, coal.Sphere)

        # Getting its pose in the world reference
        shape1_placement = cdata.oMg[shape1_id]

        # Doing the same for the second shape.
        shape2_id = cmodel.getGeometryId("obj1")

        # Coloring the sphere
        shape2 = cmodel.geometryObjects[shape2_id]
        self.assertIsInstance(shape2.geometry, coal.Sphere)

        # Getting its pose in the world reference
        shape2_placement = cdata.oMg[shape2_id]

        req = coal.DistanceRequest()
        res = coal.DistanceResult()

        # Testing the distance calculus
        distance_hpp = coal.distance(
            shape1.geometry,
            coal.Transform3s(shape1_placement.rotation, shape1_placement.translation),
            shape2.geometry,
            coal.Transform3s(shape2_placement.rotation, shape2_placement.translation),
            req,
            res,
        )

        distance_ana = np.linalg.norm(
            shape1_placement.translation - shape2_placement.translation
        ) - (r1 + r2)
        self.assertAlmostEqual(distance_ana, distance_hpp)

        # Testing the computation of closest points
        cp1 = res.getNearestPoint1()
        cp2 = res.getNearestPoint2()

        distance_cp = -1 * np.linalg.norm(cp1 - cp2)

        # - distance because interpenetration
        self.assertAlmostEqual(distance_cp, distance_ana)

    def test_sphere_capsule_not_in_collision(self):
        """
        Testing the sphere-sphere pair, going from the distance between each shape to
        making sure the closest points are well computed.
        """

        r1 = 0.7
        r2 = 0.5
        l2 = 0.7

        rmodel = pin.Model()
        cmodel = pin.GeometryModel()
        geometries = [
            coal.Sphere(r1),
            coal.Capsule(r2, l2),
        ]
        placement0 = pin.SE3(pin.utils.rotate("y", np.pi), np.array([0, 0, 2]))
        try:
            geom_obj0 = pin.GeometryObject("obj0", 0, 0, placement0, geometries[0])
        except:  # noqa 422 TODO
            geom_obj0 = pin.GeometryObject("obj0", 0, 0, geometries[0], placement0)

        cmodel.addGeometryObject(geom_obj0)

        placement1 = pin.SE3(pin.utils.rotate("y", np.pi), np.array([0, 0, 0]))
        try:
            geom_obj1 = pin.GeometryObject("obj1", 0, 0, placement1, geometries[1])
        except:  # noqa 422 TODO
            geom_obj1 = pin.GeometryObject("obj1", 0, 0, geometries[1], placement1)

        cmodel.addGeometryObject(geom_obj1)

        rdata = rmodel.createData()
        cdata = cmodel.createData()

        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)
        # Creating the shapes for the collision detection.
        shape1_id = cmodel.getGeometryId("obj0")

        # Coloring the sphere
        shape1 = cmodel.geometryObjects[shape1_id]
        self.assertIsInstance(shape1.geometry, coal.Sphere)

        # Getting its pose in the world reference
        shape1_placement = cdata.oMg[shape1_id]

        # Doing the same for the second shape.
        shape2_id = cmodel.getGeometryId("obj1")

        # Coloring the sphere
        shape2 = cmodel.geometryObjects[shape2_id]
        self.assertIsInstance(shape2.geometry, coal.Capsule)

        # Getting its pose in the world reference
        shape2_placement = cdata.oMg[shape2_id]

        req = coal.DistanceRequest()
        res = coal.DistanceResult()

        # Testing the distance calculus
        distance_hpp = coal.distance(
            shape1.geometry,
            coal.Transform3s(shape1_placement.rotation, shape1_placement.translation),
            shape2.geometry,
            coal.Transform3s(shape2_placement.rotation, shape2_placement.translation),
            req,
            res,
        )

        distance_ana = self.distance_sphere_capsule(
            shape1, shape1_placement, shape2, shape2_placement
        )
        self.assertAlmostEqual(distance_ana, distance_hpp)

        # Testing the computation of closest points
        cp1 = res.getNearestPoint1()
        cp2 = res.getNearestPoint2()

        distance_cp = np.linalg.norm(cp1 - cp2)

        # - distance because interpenetration
        self.assertAlmostEqual(distance_cp, distance_ana, places=3)

    def test_sphere_capsule_in_collision(self):
        """
        Testing the sphere-sphere pair, going from the distance between each shape to
        making sure the closest points are well computed.
        """

        r1 = 0.7
        r2 = 0.5
        l2 = 0.7

        rmodel = pin.Model()
        cmodel = pin.GeometryModel()
        geometries = [
            coal.Sphere(r1),
            coal.Capsule(r2, l2),
        ]
        placement0 = pin.SE3(pin.utils.rotate("y", np.pi), np.array([1, 0, 0]))
        try:
            geom_obj0 = pin.GeometryObject("obj0", 0, 0, placement0, geometries[0])
        except:  # noqa 422 TODO
            geom_obj0 = pin.GeometryObject("obj0", 0, 0, geometries[0], placement0)

        cmodel.addGeometryObject(geom_obj0)

        placement1 = pin.SE3(pin.utils.rotate("y", np.pi), np.array([0, 0, 0]))
        try:
            geom_obj1 = pin.GeometryObject("obj1", 0, 0, placement1, geometries[1])
        except:  # noqa 422 TODO
            geom_obj1 = pin.GeometryObject("obj1", 0, 0, geometries[1], placement1)

        cmodel.addGeometryObject(geom_obj1)

        rdata = rmodel.createData()
        cdata = cmodel.createData()

        pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata)
        # Creating the shapes for the collision detection.
        shape1_id = cmodel.getGeometryId("obj0")

        # Coloring the sphere
        shape1 = cmodel.geometryObjects[shape1_id]
        self.assertIsInstance(shape1.geometry, coal.Sphere)

        # Getting its pose in the world reference
        shape1_placement = cdata.oMg[shape1_id]

        # Doing the same for the second shape.
        shape2_id = cmodel.getGeometryId("obj1")

        # Coloring the sphere
        shape2 = cmodel.geometryObjects[shape2_id]
        self.assertIsInstance(shape2.geometry, coal.Capsule)

        # Getting its pose in the world reference
        shape2_placement = cdata.oMg[shape2_id]

        req = coal.DistanceRequest()
        res = coal.DistanceResult()

        # Testing the distance calculus
        distance_hpp = coal.distance(
            shape1.geometry,
            coal.Transform3s(shape1_placement.rotation, shape1_placement.translation),
            shape2.geometry,
            coal.Transform3s(shape2_placement.rotation, shape2_placement.translation),
            req,
            res,
        )

        distance_ana = self.distance_sphere_capsule(
            shape1, shape1_placement, shape2, shape2_placement
        )
        self.assertAlmostEqual(distance_ana, distance_hpp)

        # Testing the computation of closest points
        cp1 = res.getNearestPoint1()
        cp2 = res.getNearestPoint2()

        distance_cp = -1 * np.linalg.norm(cp1 - cp2)

        # - distance because interpenetration
        self.assertAlmostEqual(distance_cp, distance_ana)

    def distance_sphere_capsule(
        self, sphere, sphere_placement, capsule, capsule_placement
    ):
        """Computes the signed distance between a sphere & a capsule.

        Args:
            sphere (pin.GeometryObject): Geometry object of pinocchio, stored in the
                geometry model.
            capsule (pin.GeometryObject): Geometry object of pinocchio, stored in the
                geometry model.

        Returns:
            disntance (float): Signed distance between the closest points of a
                capsule-sphere pair.
        """
        r1 = sphere.geometry.radius
        r2 = capsule.geometry.radius
        A, B = self.get_A_B_from_center_capsule(capsule, capsule_placement)

        # Position of the center of the sphere
        C = sphere_placement.translation

        AB = B - A
        AC = C - A

        # Project AC onto AB, but deferring divide by Dot(AB, AB)
        t = np.dot(AC, AB)
        if t <= 0.0:
            # C projects outside the [A, B] interval, on the A side; clamp to A
            t = 0.0
            closest_point = A
        else:
            denom = np.dot(AB, AB)  # Always nonnegative since denom = ||AB||^2
            if t >= denom:
                # C projects outside the [A, B] interval, on the B side; clamp to B
                t = 1.0
                closest_point = B
            else:
                # C projects inside the [A, B] interval; must do deferred divide now
                t = t / denom
                closest_point = A + t * AB

        # Calculate distance between C and the closest point on the segment
        distance = np.linalg.norm(C - closest_point) - (r1 + r2)

        return distance

    def get_A_B_from_center_capsule(self, capsule, capsule_placement):
        """
        Computes the points A & B of a capsule. The point A & B are the limits of the
        segment defining the capsule.

        Args:
            capsule (pin.GeometryObject): Geometry object of pinocchio, stored in the
                geometry model.
        """

        A = pin.SE3.Identity()
        A.translation = np.array([0, 0, -capsule.geometry.halfLength])
        B = pin.SE3.Identity()
        B.translation = np.array([0, 0, +capsule.geometry.halfLength])

        A = A * capsule_placement
        B *= capsule_placement
        return (A.translation, B.translation)


if __name__ == "__main__":
    unittest.main()
