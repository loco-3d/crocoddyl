///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class CostModelAbstract_wrap : public CostModelAbstract, public bp::wrapper<CostModelAbstract> {
 public:
  CostModelAbstract_wrap(boost::shared_ptr<StateMultibody> state,
                         boost::shared_ptr<ActivationModelAbstract> activation, int nu, bool with_residuals = true)
      : CostModelAbstract(state, activation, nu, with_residuals) {}

  CostModelAbstract_wrap(boost::shared_ptr<StateMultibody> state,
                         boost::shared_ptr<ActivationModelAbstract> activation, bool with_residuals = true)
      : CostModelAbstract(state, activation, with_residuals) {}

  CostModelAbstract_wrap(boost::shared_ptr<StateMultibody> state, int nr, int nu, bool with_residuals = true)
      : CostModelAbstract(state, nr, nu, with_residuals), bp::wrapper<CostModelAbstract>() {}

  CostModelAbstract_wrap(boost::shared_ptr<StateMultibody> state, int nr, bool with_residuals = true)
      : CostModelAbstract(state, nr, with_residuals), bp::wrapper<CostModelAbstract>() {}

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(), "x has wrong dimension");
    assert_pretty((static_cast<std::size_t>(u.size()) == nu_ || nu_ == 0), "u has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true) {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(), "x has wrong dimension");
    assert_pretty((static_cast<std::size_t>(u.size()) == nu_ || nu_ == 0), "u has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u, recalc);
  }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(CostModel_calc_wraps, CostModelAbstract::calc_wrap, 2, 3)

void exposeCostMultibody() {
  bp::register_ptr_to_python<boost::shared_ptr<CostModelAbstract> >();

  bp::class_<CostModelAbstract_wrap, boost::noncopyable>(
      "CostModelAbstract",
      "Abstract multibody cost model using Pinocchio.\n\n"
      "It defines a template of cost model whose residual and derivatives can be retrieved from\n"
      "Pinocchio data, through the calc and calcDiff functions, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, int,
               bp::optional<bool> >(bp::args("self", "state", "activation", "nu", "withResiduals"),
                                    "Initialize the cost model.\n\n"
                                    ":param state: state of the multibody system\n"
                                    ":param activation: Activation model\n"
                                    ":param nu: dimension of control vector (default model.nv)\n"
                                    ":param withResiduals: true if the cost function has residuals (default True)"))
      .def(
          bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, bp::optional<bool> >(
              bp::args("self", "state", "activation", "withResiduals"),
              "Initialize the cost model.\n\n"
              ":param state: state of the multibody system\n"
              ":param activation: Activation model\n"
              ":param withResiduals: true if the cost function has residuals (default True)"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, int, int, bp::optional<bool> >(
          bp::args("self", "state", "nr", "nu", "withResiduals"),
          "Initialize the cost model.\n\n"
          "For this case the default activation model is quadratic, i.e. crocoddyl.ActivationModelQuad(nr).\n"
          ":param state: state of the multibody system\n"
          ":param nr: dimension of cost vector\n"
          ":param nu: dimension of control vector (default model.nv)\n"
          ":param withResiduals: true if the cost function has residuals (default True)"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, int, bp::optional<bool> >(
          bp::args("self", "state", "nr", "withResiduals"),
          "Initialize the cost model.\n\n"
          "For this case the default activation model is quadratic, i.e. crocoddyl.ActivationModelQuad(nr).\n"
          ":param state: state of the multibody system\n"
          ":param nr: dimension of cost vector\n"
          ":param withResiduals: true if the cost function has residuals (default True)"))
      .def("calc", pure_virtual(&CostModelAbstract_wrap::calc), bp::args("self", "data", "x", "u"),
           "Compute the cost value and its residuals.\n\n"
           ":param data: cost data\n"
           ":param x: state vector\n"
           ":param u: control input")
      .def("calcDiff", pure_virtual(&CostModelAbstract_wrap::calcDiff), bp::args("self", "data", "x", "u", "recalc"),
           "Compute the derivatives of the cost function and its residuals.\n\n"
           ":param data: cost data\n"
           ":param x: state vector\n"
           ":param u: control input\n"
           ":param recalc: If true, it updates the cost value.")
      .def("createData", &CostModelAbstract_wrap::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property(
          "state",
          bp::make_function(&CostModelAbstract_wrap::get_state, bp::return_value_policy<bp::return_by_value>()),
          "state of the multibody system")
      .add_property(
          "activation",
          bp::make_function(&CostModelAbstract_wrap::get_activation, bp::return_value_policy<bp::return_by_value>()),
          "activation model")
      .add_property("nu",
                    bp::make_function(&CostModelAbstract_wrap::get_nu, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of control vector");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataAbstract> >();

  bp::class_<CostDataAbstract, boost::noncopyable>(
      "CostDataAbstract", "Abstract class for cost data.\n\n",
      bp::init<CostModelAbstract*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create common data shared between cost models.\n\n"
          ":param model: cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("shared", bp::make_getter(&CostDataAbstract::shared, bp::return_internal_reference<>()),
                    "shared data")
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
      .add_property("r", bp::make_getter(&CostDataAbstract::r, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::r), "cost residual")
      .add_property("Rx", bp::make_getter(&CostDataAbstract::Rx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Rx), "Jacobian of the cost residual")
      .add_property("Ru", bp::make_getter(&CostDataAbstract::Ru, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Ru), "Jacobian of the cost residual");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_
