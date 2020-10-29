///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/constraints/frame-velocity-equality.hpp"

namespace crocoddyl {
namespace python {

void exposeConstraintFrameVelocityEquality() {
  bp::class_<ConstraintModelFrameVelocityEquality, bp::bases<ConstraintModelAbstract> >(
      "ConstraintModelFrameVelocityEquality",
      "This equality constraint function imposes a reference velocity of a given frame, i.e. v - vref = 0, with v "
      "and vref as the current and reference "
      "frame velocity, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, FrameMotion, std::size_t>(
          bp::args("self", "state", "vref", "nu"),
          "Initialize the frame velocity equality constraint model.\n\n"
          ":param state: state of the multibody system\n"
          ":param vref: reference frame velocity\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameMotion>(
          bp::args("self", "state", "vref"),
          "Initialize the frame velocity equality constraint model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param Mref: reference frame velocity"))
      .def<void (ConstraintModelFrameVelocityEquality::*)(const boost::shared_ptr<ConstraintDataAbstract>&,
                                                          const Eigen::Ref<const Eigen::VectorXd>&,
                                                          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ConstraintModelFrameVelocityEquality::calc, bp::args("self", "data", "x", "u"),
          "Compute the residual of the frame velocity constraint.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ConstraintModelFrameVelocityEquality::*)(const boost::shared_ptr<ConstraintDataAbstract>&,
                                                          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ConstraintModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ConstraintModelFrameVelocityEquality::*)(const boost::shared_ptr<ConstraintDataAbstract>&,
                                                          const Eigen::Ref<const Eigen::VectorXd>&,
                                                          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ConstraintModelFrameVelocityEquality::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the frame velocity constraint.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ConstraintModelFrameVelocityEquality::*)(const boost::shared_ptr<ConstraintDataAbstract>&,
                                                          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ConstraintModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ConstraintModelFrameVelocityEquality::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(), bp::args("self", "data"),
           "Create the frame velocity constraint data.\n\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference", &ConstraintModelFrameVelocityEquality::get_reference<FrameMotion>,
                    &ConstraintModelFrameVelocityEquality::set_reference<FrameMotion>, "reference frame velocity");

  bp::register_ptr_to_python<boost::shared_ptr<ConstraintDataFrameVelocityEquality> >();

  bp::class_<ConstraintDataFrameVelocityEquality, bp::bases<ConstraintDataAbstract> >(
      "ConstraintDataFrameVelocityEquality", "Data for frame velocity constraint.\n\n",
      bp::init<ConstraintModelFrameVelocityEquality*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame velocity constraint data.\n\n"
          ":param model: frame velocity cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("h", bp::make_getter(&ConstraintDataFrameVelocityEquality::h, bp::return_internal_reference<>()),
                    "constraint residual");
}

}  // namespace python
}  // namespace crocoddyl
