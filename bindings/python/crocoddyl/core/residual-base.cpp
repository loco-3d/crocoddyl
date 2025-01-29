///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/residual-base.hpp"

#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/printable.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualAbstract() {
  bp::register_ptr_to_python<std::shared_ptr<ResidualModelAbstract> >();

  bp::class_<ResidualModelAbstract_wrap, boost::noncopyable>(
      "ResidualModelAbstract",
      "Abstract class for residual models.\n\n"
      "A residual model defines a vector function r(x,u) in R^nr, where nr "
      "describes its dimension in the\n"
      "the Euclidean space. For each residual model, we need to provide ways "
      "of computing the residual\n"
      "vector and its Jacobians. These computations are mainly carried out "
      "inside calc() and calcDiff(),\n"
      "respectively.",
      bp::init<std::shared_ptr<StateAbstract>, std::size_t, std::size_t,
               bp::optional<bool, bool, bool> >(
          bp::args("self", "state", "nr", "nu", "q_dependent", "v_dependent",
                   "u_dependent"),
          "Initialize the residual model.\n\n"
          ":param state: state description,\n"
          ":param nr: dimension of the residual vector\n"
          ":param nu: dimension of control vector (default state.nv)\n"
          ":param q_dependent: define if the residual function depends on q "
          "(default true)\n"
          ":param v_dependent: define if the residual function depends on v "
          "(default true)\n"
          ":param u_dependent: define if the residual function depends on u "
          "(default true)"))
      .def(bp::init<std::shared_ptr<StateAbstract>, std::size_t,
                    bp::optional<bool, bool, bool> >(
          bp::args("self", "state", "nr", "q_dependent", "v_dependent",
                   "u_dependent"),
          "Initialize the cost model.\n\n"
          ":param state: state description\n"
          ":param nr: dimension of the residual vector\n"
          ":param q_dependent: define if the residual function depends on q "
          "(default true)\n"
          ":param v_dependent: define if the residual function depends on v "
          "(default true)\n"
          ":param u_dependent: define if the residual function depends on u "
          "(default true)"))
      .def("calc", pure_virtual(&ResidualModelAbstract_wrap::calc),
           bp::args("self", "data", "x", "u"),
           "Compute the residual vector.\n\n"
           ":param data: residual data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: control input (dim. nu)")
      .def<void (ResidualModelAbstract::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"),
          "Compute the residual vector for nodes that depends only on the "
          "state.\n\n"
          "It updates the residual vector based on the state only.\n"
          "This function is used in the terminal nodes of an optimal control "
          "problem.\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)")
      .def("calcDiff", pure_virtual(&ResidualModelAbstract_wrap::calcDiff),
           bp::args("self", "data", "x", "u"),
           "Compute the Jacobians of the residual function.\n\n"
           ":param data: residual data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: control input (dim. nu)")
      .def<void (ResidualModelAbstract::*)(
          const std::shared_ptr<ResidualDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff,
          bp::args("self", "data", "x"),
          "Compute the Jacobian of the residual functions with respect to the "
          "state only.\n\n"
          "It updates the Jacobian of the residual function based on the state "
          "only.\n"
          "This function is used in the terminal nodes of an optimal control "
          "problem.\n"
          ":param data: residual data\n"
          ":param x: state point (dim. state.nx)")
      .def("createData", &ResidualModelAbstract_wrap::createData,
           &ResidualModelAbstract_wrap::default_createData, bp::args("self"),
           "Create the residual data.\n\n"
           "Each residual model might has its own data that needs to be "
           "allocated.")
      .def("calcCostDiff", &ResidualModelAbstract_wrap::calcCostDiff,
           &ResidualModelAbstract_wrap::default_calcCostDiff,
           bp::args("self", "cdata", "rdata", "adata", "update_u"),
           "Compute the derivative of the cost function.\n\n"
           "This function assumes that the derivatives of the activation and "
           "residual are\n"
           "computed via calcDiff functions.\n"
           ":param cdata: cost data\n"
           ":param rdata: residual data\n"
           ":param adata: activation data\n"
           ":param update_u: update the derivative of the cost function w.r.t. "
           "to the control if True.")
      .def("calcCostDiff",
           &ResidualModelAbstract_wrap::default_calcCostDiff_noupdate_u)
      .add_property(
          "state",
          bp::make_function(&ResidualModelAbstract_wrap::get_state,
                            bp::return_value_policy<bp::return_by_value>()),
          "state")
      .add_property("nr",
                    bp::make_function(&ResidualModelAbstract_wrap::get_nr),
                    "dimension of residual vector")
      .add_property("nu",
                    bp::make_function(&ResidualModelAbstract_wrap::get_nu),
                    "dimension of control vector")
      .add_property(
          "q_dependent",
          bp::make_function(&ResidualModelAbstract_wrap::get_q_dependent),
          "flag that indicates if the residual function depends on q")
      .add_property(
          "v_dependent",
          bp::make_function(&ResidualModelAbstract_wrap::get_v_dependent),
          "flag that indicates if the residual function depends on v")
      .add_property(
          "u_dependent",
          bp::make_function(&ResidualModelAbstract_wrap::get_u_dependent),
          "flag that indicates if the residual function depends on u")
      .def(CopyableVisitor<ResidualModelAbstract_wrap>())
      .def(PrintableVisitor<ResidualModelAbstract>());

  bp::register_ptr_to_python<std::shared_ptr<ResidualDataAbstract> >();

  bp::class_<ResidualDataAbstract>(
      "ResidualDataAbstract",
      "Abstract class for residual data.\n\n"
      "In crocoddyl, a residual data contains all the required information for "
      "processing an\n"
      "user-defined residual models. The residual data typically is allocated "
      "once and containts\n"
      "the residual vector and its Jacobians.",
      bp::init<ResidualModelAbstract*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create common data shared between residual models.\n\n"
          ":param model: residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("shared",
                    bp::make_getter(&ResidualDataAbstract::shared,
                                    bp::return_internal_reference<>()),
                    "shared data")
      .add_property("r",
                    bp::make_getter(&ResidualDataAbstract::r,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ResidualDataAbstract::r),
                    "residual vector")
      .add_property("Rx",
                    bp::make_getter(&ResidualDataAbstract::Rx,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ResidualDataAbstract::Rx),
                    "Jacobian of the residual")
      .add_property("Ru",
                    bp::make_getter(&ResidualDataAbstract::Ru,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ResidualDataAbstract::Ru),
                    "Jacobian of the residual")
      .add_property("Arr_Rx",
                    bp::make_getter(&ResidualDataAbstract::Arr_Rx,
                                    bp::return_internal_reference<>()),
                    "Intermediate product of Arr (2nd deriv of Activation) "
                    "with Rx (deriv of residue)")
      .add_property("Arr_Ru",
                    bp::make_getter(&ResidualDataAbstract::Arr_Ru,
                                    bp::return_internal_reference<>()),
                    "Intermediate product of Arr (2nd deriv of Activation) "
                    "with Ru (deriv of residue)")
      .def(CopyableVisitor<ResidualDataAbstract>());
}

}  // namespace python
}  // namespace crocoddyl
