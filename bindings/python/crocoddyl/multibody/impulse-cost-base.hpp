///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSE_COST_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSE_COST_BASE_HPP_

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/impulse-cost-base.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class ImpulseCostModelAbstract_wrap : public ImpulseCostModelAbstract, public bp::wrapper<ImpulseCostModelAbstract> {
 public:
  ImpulseCostModelAbstract_wrap(boost::shared_ptr<StateMultibody> state,
                                boost::shared_ptr<ActivationModelAbstract> activation, bool with_residuals = true)
      : ImpulseCostModelAbstract(state, activation, with_residuals) {}

  ImpulseCostModelAbstract_wrap(boost::shared_ptr<StateMultibody> state, int nr, bool with_residuals = true)
      : ImpulseCostModelAbstract(state, nr, with_residuals), bp::wrapper<ImpulseCostModelAbstract>() {}

  void calc(const boost::shared_ptr<ImpulseCostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x) {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(), "x has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x);
  }

  void calcDiff(const boost::shared_ptr<ImpulseCostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const bool& recalc = true) {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(), "x has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, recalc);
  }
};

// BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ImpulseCostModel_calc_wraps, ImpulseCostModelAbstract::calc_wrap, 1, 2)

void exposeImpulseCostMultibody() {
  bp::register_ptr_to_python<boost::shared_ptr<ImpulseCostModelAbstract> >();

  bp::class_<ImpulseCostModelAbstract_wrap, boost::noncopyable>(
      "ImpulseCostModelAbstract",
      "Abstract impulse cost model using Pinocchio.\n\n"
      "It defines a template of impulse costs whose residual and derivatives can be retrieved from\n"
      "Pinocchio data, through the calc and calcDiff functions, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, bp::optional<bool> >(
          bp::args("self", "state", "activation", "withResiduals"),
          "Initialize the impulse cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: Activation model\n"
          ":param withResiduals: true if the impulse cost function has residuals (default True)"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, int, bp::optional<bool> >(
          bp::args("self", "state", "nr", "withResiduals"),
          "Initialize the impulse cost model.\n\n"
          "For this case the default activation model is quadratic, i.e. crocoddyl.ActivationModelQuad(nr).\n"
          ":param state: state of the multibody system\n"
          ":param nr: dimension of impulse cost vector\n"
          ":param withResiduals: true if the impulse cost function has residuals (default True)"))
      .def("calc", pure_virtual(&ImpulseCostModelAbstract_wrap::calc), bp::args("self", "data", "x"),
           "Compute the impulse cost value and its residuals.\n\n"
           ":param data: impulse cost data\n"
           ":param x: state vector")
      .def("calcDiff", pure_virtual(&ImpulseCostModelAbstract_wrap::calcDiff), bp::args("self", "data", "x", "recalc"),
           "Compute the derivatives of the impulse cost function and its residuals.\n\n"
           ":param data: impulse cost data\n"
           ":param x: state vector\n"
           ":param recalc: If true, it updates the cost value.")
      .def("createData", &ImpulseCostModelAbstract_wrap::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the impulse cost data.\n\n"
           "Each impulse cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined impulse cost.\n"
           ":param data: shared data\n"
           ":return impulse cost data.")
      .add_property(
          "state",
          bp::make_function(&ImpulseCostModelAbstract_wrap::get_state, bp::return_value_policy<bp::return_by_value>()),
          "state of the multibody system")
      .add_property("activation",
                    bp::make_function(&ImpulseCostModelAbstract_wrap::get_activation,
                                      bp::return_value_policy<bp::return_by_value>()),
                    "activation model");

  bp::register_ptr_to_python<boost::shared_ptr<ImpulseCostDataAbstract> >();

  bp::class_<ImpulseCostDataAbstract, boost::noncopyable>(
      "ImpulseCostDataAbstract", "Abstract class for impulse cost data.\n\n",
      bp::init<ImpulseCostModelAbstract*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create common data shared between impulse cost models.\n\n"
          ":param model: impulse cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("shared", bp::make_getter(&ImpulseCostDataAbstract::shared, bp::return_internal_reference<>()),
                    "shared data")
      .add_property(
          "activation",
          bp::make_getter(&ImpulseCostDataAbstract::activation, bp::return_value_policy<bp::return_by_value>()),
          "terminal data")
      .add_property("cost",
                    bp::make_getter(&ImpulseCostDataAbstract::cost, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ImpulseCostDataAbstract::cost), "impulse cost value")
      .add_property("Lx",
                    bp::make_getter(&ImpulseCostDataAbstract::Lx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ImpulseCostDataAbstract::Lx), "Jacobian of the impulse cost")
      .add_property("Lxx",
                    bp::make_getter(&ImpulseCostDataAbstract::Lxx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ImpulseCostDataAbstract::Lxx), "Hessian of the impulse cost")
      .add_property("r", bp::make_getter(&ImpulseCostDataAbstract::r, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ImpulseCostDataAbstract::r), "impulse cost residual")
      .add_property("Rx",
                    bp::make_getter(&ImpulseCostDataAbstract::Rx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ImpulseCostDataAbstract::Rx), "Jacobian of the impulse cost residual");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_
