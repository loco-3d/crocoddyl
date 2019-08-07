///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_
#define PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_

#include "crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class CostModelAbstract_wrap : public CostModelAbstract, public bp::wrapper<CostModelAbstract> {
 public:
  CostModelAbstract_wrap(StateMultibody& state, ActivationModelAbstract& activation, int nu,
                         bool with_residuals = true)
      : CostModelAbstract(state, activation, nu, with_residuals) {}

  CostModelAbstract_wrap(StateMultibody& state, ActivationModelAbstract& activation, bool with_residuals = true)
      : CostModelAbstract(state, activation, with_residuals) {}

  CostModelAbstract_wrap(StateMultibody& state, int nr, int nu, bool with_residuals = true)
      : CostModelAbstract(state, nr, nu, with_residuals), bp::wrapper<CostModelAbstract>() {}

  CostModelAbstract_wrap(StateMultibody& state, int nr, bool with_residuals = true)
      : CostModelAbstract(state, nr, with_residuals), bp::wrapper<CostModelAbstract>() {}

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true) {
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u, recalc);
  }
};

void exposeCostMultibody() {
  bp::class_<CostModelAbstract_wrap, boost::noncopyable>(
      "CostModelAbstract",
      "Abstract multibody cost model using Pinocchio.\n\n"
      "It defines a template of cost model whose residual and derivatives can be retrieved from\n"
      "Pinocchio data, through the calc and calcDiff functions, respectively.",
      bp::init<StateMultibody&, ActivationModelAbstract&, int, bp::optional<bool> >(
          bp::args(" self", " state", " activation", " nu=model.nv", " withResiduals=True"),
          "Initialize the cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: Activation model\n"
          ":param nu: dimension of control vector\n"
          ":param withResiduals: true if the cost function has residuals")[bp::with_custodian_and_ward<1, 3>()])
      .def(bp::init<StateMultibody&, ActivationModelAbstract&, bp::optional<bool> >(
          bp::args(" self", " state", " activation", " withResiduals=True"),
          "Initialize the cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: Activation model\n"
          ":param withResiduals: true if the cost function has residuals")[bp::with_custodian_and_ward<1, 3>()])
      .def(bp::init<StateMultibody&, int, int, bp::optional<bool> >(
          bp::args(" self", " state", " nr", " nu=model.nv", " withResiduals=True"),
          "Initialize the cost model.\n\n"
          "For this case the default activation model is quadratic, i.e. crocoddyl.ActivationModelQuad(nr).\n"
          ":param state: state of the multibody system\n"
          ":param nr: dimension of cost vector\n"
          ":param nu: dimension of control vector\n"
          ":param withResiduals: true if the cost function has residuals")[bp::with_custodian_and_ward<1, 2>()])
      .def(bp::init<StateMultibody&, int, bp::optional<bool> >(
          bp::args(" self", " state", " nr", " withResiduals=True"),
          "Initialize the cost model.\n\n"
          "For this case the default activation model is quadratic, i.e. crocoddyl.ActivationModelQuad(nr).\n"
          ":param state: state of the multibody system\n"
          ":param nr: dimension of cost vector\n"
          ":param withResiduals: true if the cost function has residuals")[bp::with_custodian_and_ward<1, 2>()])
      .def("calc", pure_virtual(&CostModelAbstract_wrap::calc), bp::args(" self", " data", " x", " u"),
           "Compute the cost value and its residuals.\n\n"
           ":param data: cost data\n"
           ":param x: state vector\n"
           ":param u: control input")
      .def("calcDiff", pure_virtual(&CostModelAbstract_wrap::calcDiff),
           bp::args(" self", " data", " x", " u", " recalc=True"),
           "Compute the derivatives of the cost function and its residuals.\n\n"
           ":param data: cost data\n"
           ":param x: state vector\n"
           ":param u: control input\n"
           ":param recalc: If true, it updates the cost value.")
      .def("createData", &CostModelAbstract_wrap::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return cost data.")
      .add_property("State", bp::make_function(&CostModelAbstract_wrap::get_state, bp::return_internal_reference<>()),
                    "state of the multibody system")
      .add_property("activation",
                    bp::make_function(&CostModelAbstract_wrap::get_activation, bp::return_internal_reference<>()),
                    "activation model")
      .add_property("nu",
                    bp::make_function(&CostModelAbstract_wrap::get_nu, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of control vector");

  bp::class_<CostDataAbstract, boost::shared_ptr<CostDataAbstract>, boost::noncopyable>(
      "CostDataAbstract", "Abstract class for cost datas.\n\n",
      bp::init<CostModelAbstract*, pinocchio::Data*>(
          bp::args(" self", " model", " data"),
          "Create common data shared between cost models.\n\n"
          ":param model: cost model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 3>()])
      .add_property("pinocchio",
                    bp::make_function(&CostDataAbstract::get_pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("activation",
                    bp::make_getter(&CostDataAbstract::activation, bp::return_value_policy<bp::return_by_value>()),
                    "terminal data")
      .add_property("cost", bp::make_getter(&CostDataAbstract::cost, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::cost), "cost value")
      .add_property("Lx", bp::make_getter(&CostDataAbstract::Lx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Lx), "Jacobian of the cost")
      .add_property("Lu", bp::make_getter(&CostDataAbstract::Lu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Lu), "Jacobian of the cost")
      .add_property("Lxx", bp::make_getter(&CostDataAbstract::Lxx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Lxx), "Hessian of the cost")
      .add_property("Lxu", bp::make_getter(&CostDataAbstract::Lxu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Lxu), "Hessian of the cost")
      .add_property("Luu", bp::make_getter(&CostDataAbstract::Luu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Luu), "Hessian of the cost")
      .add_property("costResiduals",
                    bp::make_getter(&CostDataAbstract::r, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::r), "cost residual")
      .add_property("Rx", bp::make_getter(&CostDataAbstract::Rx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Rx), "Jacobian of the cost residual")
      .add_property("Ru", bp::make_getter(&CostDataAbstract::Ru, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Ru), "Jacobian of the cost residual");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_