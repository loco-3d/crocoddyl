///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/constraints/frame-placement-equality.hpp"

namespace crocoddyl {
namespace python {

void exposeConstraintFramePlacementEquality() {
  bp::class_<ConstraintModelFramePlacementEquality, bp::bases<ConstraintModelAbstract> >(
      "ConstraintModelFramePlacementEquality",
      "This equality constraint function imposes a reference placement of a given frame, i.e. p - pref = 0, with p "
      "and pref as the current and reference "
      "frame placements, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, FramePlacement, std::size_t>(
          bp::args("self", "state", "Mref", "nu"),
          "Initialize the frame placement equality constraint model.\n\n"
          ":param state: state of the multibody system\n"
          ":param Mref: reference frame placement\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FramePlacement>(
          bp::args("self", "state", "Mref"),
          "Initialize the frame placement equality constraint model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param Mref: reference frame placement"))
      .def<void (ConstraintModelFramePlacementEquality::*)(const boost::shared_ptr<ConstraintDataAbstract>&,
                                                           const Eigen::Ref<const Eigen::VectorXd>&,
                                                           const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ConstraintModelFramePlacementEquality::calc, bp::args("self", "data", "x", "u"),
          "Compute the residual of the frame placement constraint.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ConstraintModelFramePlacementEquality::*)(const boost::shared_ptr<ConstraintDataAbstract>&,
                                                           const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ConstraintModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ConstraintModelFramePlacementEquality::*)(const boost::shared_ptr<ConstraintDataAbstract>&,
                                                           const Eigen::Ref<const Eigen::VectorXd>&,
                                                           const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ConstraintModelFramePlacementEquality::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the frame placement constraint.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ConstraintModelFramePlacementEquality::*)(const boost::shared_ptr<ConstraintDataAbstract>&,
                                                           const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ConstraintModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ConstraintModelFramePlacementEquality::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(), bp::args("self", "data"),
           "Create the frame placement constraint data.\n\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference", &ConstraintModelFramePlacementEquality::get_reference<FramePlacement>,
                    &ConstraintModelFramePlacementEquality::set_reference<FramePlacement>,
                    "reference frame placement");

  bp::register_ptr_to_python<boost::shared_ptr<ConstraintDataFramePlacementEquality> >();

  bp::class_<ConstraintDataFramePlacementEquality, bp::bases<ConstraintDataAbstract> >(
      "ConstraintDataFramePlacementEquality", "Data for frame placement constraint.\n\n",
      bp::init<ConstraintModelFramePlacementEquality*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame placement constraint data.\n\n"
          ":param model: frame placement cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("h", bp::make_getter(&ConstraintDataFramePlacementEquality::h, bp::return_internal_reference<>()),
                    "constraint residual")
      .add_property(
          "rMf",
          bp::make_getter(&ConstraintDataFramePlacementEquality::rMf, bp::return_value_policy<bp::return_by_value>()),
          "error frame placement of the frame")
      .add_property("J", bp::make_getter(&ConstraintDataFramePlacementEquality::J, bp::return_internal_reference<>()),
                    "Jacobian at the error point")
      .add_property("rJf",
                    bp::make_getter(&ConstraintDataFramePlacementEquality::rJf, bp::return_internal_reference<>()),
                    "error Jacobian of the frame")
      .add_property("fJf",
                    bp::make_getter(&ConstraintDataFramePlacementEquality::fJf, bp::return_internal_reference<>()),
                    "local Jacobian of the frame");
}

}  // namespace python
}  // namespace crocoddyl
