///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/frame-placement.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualFramePlacement() {
  bp::register_ptr_to_python<std::shared_ptr<ResidualModelFramePlacement> >();

  bp::class_<ResidualModelFramePlacement, bp::bases<ResidualModelAbstract> >(
      "ResidualModelFramePlacement",
      "This residual function defines the tracking of theframe placement "
      "residual as r = p - pref, with p and pref "
      "as\n"
      "the current and reference frame placements, respectively.",
      bp::init<std::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
               pinocchio::SE3, std::size_t>(
          bp::args("self", "state", "id", "pref", "nu"),
          "Initialize the frame placement residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param pref: reference frame placement\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<std::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
                    pinocchio::SE3>(
          bp::args("self", "state", "id", "pref"),
          "Initialize the frame placement residual model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param pref: reference frame placement"))
      .def<void (ResidualModelFramePlacement::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelFramePlacement::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the frame placement residual.\n\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelFramePlacement::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelFramePlacement::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelFramePlacement::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the frame placement residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelFramePlacement::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &ResidualModelFramePlacement::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame placement residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for the frame placement residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("id", &ResidualModelFramePlacement::get_id,
                    &ResidualModelFramePlacement::set_id, "reference frame id")
      .add_property("reference",
                    bp::make_function(
                        &ResidualModelFramePlacement::get_reference,
                        bp::return_value_policy<bp::copy_const_reference>()),
                    &ResidualModelFramePlacement::set_reference,
                    "reference frame placement")
      .def(CopyableVisitor<ResidualModelFramePlacement>());

  bp::register_ptr_to_python<std::shared_ptr<ResidualDataFramePlacement> >();

  bp::class_<ResidualDataFramePlacement, bp::bases<ResidualDataAbstract> >(
      "ResidualDataFramePlacement", "Data for frame placement residual.\n\n",
      bp::init<ResidualModelFramePlacement*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create frame placement residual data.\n\n"
          ":param model: frame placement residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataFramePlacement::pinocchio,
                                    bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property(
          "rMf",
          bp::make_getter(&ResidualDataFramePlacement::rMf,
                          bp::return_value_policy<bp::return_by_value>()),
          "error frame placement of the frame")
      .add_property("rJf",
                    bp::make_getter(&ResidualDataFramePlacement::rJf,
                                    bp::return_internal_reference<>()),
                    "error Jacobian of the frame")
      .add_property("fJf",
                    bp::make_getter(&ResidualDataFramePlacement::fJf,
                                    bp::return_internal_reference<>()),
                    "local Jacobian of the frame")
      .def(CopyableVisitor<ResidualDataFramePlacement>());
}

}  // namespace python
}  // namespace crocoddyl
