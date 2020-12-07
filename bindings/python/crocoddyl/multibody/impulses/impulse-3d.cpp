///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"

namespace crocoddyl {
namespace python {

void exposeImpulse3D() {
  bp::class_<ImpulseModel3D, bp::bases<ImpulseModelAbstract> >(
      "ImpulseModel3D",
      "Rigid 3D impulse model.\n\n"
      "It defines a rigid 3D impulse models (point impulse) based on acceleration-based holonomic constraints.\n"
      "The calc and calcDiff functions compute the impulse Jacobian and drift (holonomic constraint) or\n"
      "the derivatives of the holonomic constraint, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, int>(bp::args("self", "state", "frame"),
                                                       "Initialize the 3D impulse model.\n\n"
                                                       ":param state: state of the multibody system\n"
                                                       ":param frame: reference frame id"))
      .def("calc", &ImpulseModel3D::calc, bp::args("self", "data", "x"),
           "Compute the 3D impulse Jacobian and drift.\n\n"
           "The rigid impulse model throught acceleration-base holonomic constraint\n"
           "of the impulse frame placement.\n"
           ":param data: impulse data\n"
           ":param x: state vector")
      .def("calcDiff", &ImpulseModel3D::calcDiff, bp::args("self", "data", "x"),
           "Compute the derivatives of the 3D impulse holonomic constraint.\n\n"
           "The rigid impulse model throught acceleration-base holonomic constraint\n"
           "of the impulse frame placement.\n"
           "It assumes that calc has been run first.\n"
           ":param data: cost data\n"
           ":param x: state vector\n")
      .def("updateForce", &ImpulseModel3D::updateForce, bp::args("self", "data", "force"),
           "Convert the force into a stack of spatial forces.\n\n"
           ":param data: cost data\n"
           ":param force: force vector (dimension 3)")
      .def("createData", &ImpulseModel3D::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the 3D impulse data.\n\n"
           "Each impulse model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return impulse data.")
      .add_property("frame",
                    bp::make_function(&ImpulseModel3D::get_frame),
                    "reference frame id");

  bp::register_ptr_to_python<boost::shared_ptr<ImpulseData3D> >();

  bp::class_<ImpulseData3D, bp::bases<ImpulseDataAbstract> >(
      "ImpulseData3D", "Data for 3D impulse.\n\n",
      bp::init<ImpulseModel3D*, pinocchio::Data*>(
          bp::args("self", "model", "data"),
          "Create 3D impulse data.\n\n"
          ":param model: 3D impulse model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("fXj", bp::make_getter(&ImpulseData3D::fXj, bp::return_value_policy<bp::return_by_value>()),
                    "action matrix from impulse to local frames")
      .add_property("fJf", bp::make_getter(&ImpulseData3D::fJf, bp::return_internal_reference<>()),
                    "local Jacobian of the impulse frame")
      .add_property("v_partial_dq", bp::make_getter(&ImpulseData3D::v_partial_dq, bp::return_internal_reference<>()),
                    "Jacobian of the spatial body velocity")
      .add_property("v_partial_dv", bp::make_getter(&ImpulseData3D::v_partial_dv, bp::return_internal_reference<>()),
                    "Jacobian of the spatial body velocity");
}

}  // namespace python
}  // namespace crocoddyl
