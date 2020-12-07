///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"

namespace crocoddyl {
namespace python {

void exposeImpulse6D() {
  bp::class_<ImpulseModel6D, bp::bases<ImpulseModelAbstract> >(
      "ImpulseModel6D",
      "Rigid 6D impulse model.\n\n"
      "It defines a rigid 6D impulse models based on acceleration-based holonomic constraints.\n"
      "The calc and calcDiff functions compute the impulse Jacobian and drift (holonomic constraint) or\n"
      "the derivatives of the holonomic constraint, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, int>(bp::args("self", "state", "frame"),
                                                       "Initialize the impulse model.\n\n"
                                                       ":param state: state of the multibody system\n"
                                                       ":param frame: reference frame id"))
      .def("calc", &ImpulseModel6D::calc, bp::args("self", "data", "x"),
           "Compute the 6D impulse Jacobian and drift.\n\n"
           "The rigid impulse model throught acceleration-base holonomic constraint\n"
           "of the impulse frame placement.\n"
           ":param data: impulse data\n"
           ":param x: state vector")
      .def("calcDiff", &ImpulseModel6D::calcDiff, bp::args("self", "data", "x"),
           "Compute the derivatives of the 6D impulse holonomic constraint.\n\n"
           "The rigid impulse model throught acceleration-base holonomic constraint\n"
           "of the impulse frame placement.\n"
           "It assumes that calc has been run first.\n"
           ":param data: cost data\n"
           ":param x: state vector\n")
      .def("updateForce", &ImpulseModel6D::updateForce, bp::args("self", "data", "force"),
           "Convert the force into a stack of spatial forces.\n\n"
           ":param data: cost data\n"
           ":param lambda: force vector (dimension 6)")
      .def("createData", &ImpulseModel6D::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the 6D impulse data.\n\n"
           "Each impulse model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return impulse data.")
      .add_property("frame",
                    bp::make_function(&ImpulseModel6D::get_frame),
                    "reference frame id");

  bp::register_ptr_to_python<boost::shared_ptr<ImpulseData6D> >();

  bp::class_<ImpulseData6D, bp::bases<ImpulseDataAbstract> >(
      "ImpulseData6D", "Data for 6D impulse.\n\n",
      bp::init<ImpulseModel6D*, pinocchio::Data*>(
          bp::args("self", "model", "data"),
          "Create 6D impulse data.\n\n"
          ":param model: 6D impulse model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("fXj", bp::make_getter(&ImpulseData6D::fXj, bp::return_value_policy<bp::return_by_value>()),
                    "action matrix from impulse to local frames")
      .add_property("fJf", bp::make_getter(&ImpulseData6D::fJf, bp::return_internal_reference<>()),
                    "local Jacobian of the impulse frame")
      .add_property("v_partial_dq", bp::make_getter(&ImpulseData6D::v_partial_dq, bp::return_internal_reference<>()),
                    "Jacobian of the spatial body velocity")
      .add_property("v_partial_dv", bp::make_getter(&ImpulseData6D::v_partial_dv, bp::return_internal_reference<>()),
                    "Jacobian of the spatial body velocity");
}

}  // namespace python
}  // namespace crocoddyl
