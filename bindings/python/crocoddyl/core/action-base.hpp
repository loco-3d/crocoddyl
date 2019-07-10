///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_
#define PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_

#include <crocoddyl/core/action-base.hpp>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class ActionModelAbstract_wrap : public ActionModelAbstract, public bp::wrapper<ActionModelAbstract> {
 public:
  using ActionModelAbstract::ncost_;
  using ActionModelAbstract::ndx_;
  using ActionModelAbstract::nu_;
  using ActionModelAbstract::nx_;
  using ActionModelAbstract::unone_;

  ActionModelAbstract_wrap(StateAbstract* const state, int nu, int ncost = 0)
      : ActionModelAbstract(state, nu, ncost), bp::wrapper<ActionModelAbstract>() {}

  void calc(std::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }
  void calc_wrap(std::shared_ptr<ActionDataAbstract>& data, const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    calc(data, x, u);
  }

  void calcDiff(std::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true) {
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x,
                          (Eigen::VectorXd)u, recalc);
  }

  std::shared_ptr<ActionDataAbstract> createData() { return std::make_shared<ActionDataAbstract>(this); }
};


void exposeActionAbstract() {
  bp::class_<ActionModelAbstract_wrap, boost::noncopyable>(
      "ActionModelAbstract",
      R"(Abstract class for action models.

        In crocoddyl, an action model combines dynamics and cost data. Each node, in our optimal 
        control problem, is described through an action model. Every time that we want to describe
        a problem, we need to provide ways of computing the dynamics, cost functions and their
        derivatives. These computations are mainly carry on inside calc() and calcDiff(),
        respectively.)",
      bp::init<StateAbstract*, int, bp::optional<int>>(bp::args(" self", " state", " nu", " ncost"),
                                                       R"(Initialize the action model.

:param state: state description,
:param nu: dimension of control vector,
:param ncost: dimension of the cost-residual vector)"))
      .def("calc", pure_virtual(&ActionModelAbstract_wrap::calc_wrap), bp::args(" self", " data", " x", " u"),
           R"(Compute the next state and cost value.

It describes the time-discrete evolution of our dynamical system
in which we obtain the next state. Additionally it computes the
cost value associated to this discrete state and control pair.
:param data: action data
:param x: time-discrete state vector
:param u: time-discrete control input
:returns the next state and cost value)")
      .def("calcDiff", pure_virtual(&ActionModelAbstract_wrap::calcDiff),
           bp::args(" self", " data", " x", " u", " recalc=True"),
           R"(Compute the derivatives of the dynamics and cost functions.

It computes the partial derivatives of the dynamical system and the
cost function. If recalc == True, it first updates the state evolution
and cost value. This function builds a quadratic approximation of the
action model (i.e. linear dynamics and quadratic cost).
:param data: action data
:param x: time-discrete state vector
:param u: time-discrete control input
:param recalc: If true, it updates the state evolution and the cost value.
:returns the next state and cost value)")
      .def("createData", &ActionModelAbstract_wrap::createData, bp::args(" self"),
           R"(Create the action data.

Each action model (AM) has its own data that needs to be allocated.
This function returns the allocated data for a predefined AM. Note that
you need to defined the ActionDataType inside your AM.
:return AM data.)")
      .add_property("nx", &ActionModelAbstract_wrap::nx_, "dimension of state configuration vector")
      .add_property("ndx", &ActionModelAbstract_wrap::ndx_, "dimension of state tangent vector")
      .add_property("nu", &ActionModelAbstract_wrap::nu_, "dimension of control vector")
      .add_property("ncost", &ActionModelAbstract_wrap::ncost_, "dimension of cost-residual vector")
      .add_property("unone",
                    bp::make_getter(&ActionModelAbstract_wrap::unone_, bp::return_value_policy<bp::return_by_value>()),
                    "default control vector")
      .def("State", &ActionModelAbstract_wrap::get_state, bp::return_value_policy<bp::reference_existing_object>());

  bp::register_ptr_to_python<std::shared_ptr<ActionDataAbstract>>();

  bp::class_<ActionDataAbstract, std::shared_ptr<ActionDataAbstract>, boost::noncopyable>(
      "ActionDataAbstract",
      R"(Abstract class for action datas.

        In crocoddyl, an action data contains all the required information for processing an
        user-defined action model. The action data typically is allocated onces by running
        model.createData() and contains the first- and second- order derivatives of the dynamics
        and cost function, respectively.)",
      bp::init<ActionModelAbstract*>(bp::args(" self", " model"),
                                     R"(Create common data shared between AMs.

The action data uses the model in order to first process it.
:param model: action model)"))
      .add_property("cost",
                    bp::make_getter(&ActionDataAbstract::cost, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::cost), "cost value")
      .add_property("xnext",
                    bp::make_getter(&ActionDataAbstract::xnext, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::xnext), "next state")
      .add_property("Fx",
                    bp::make_getter(&ActionDataAbstract::Fx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Fx), "Jacobian of the dynamics")
      .add_property("Fu",
                    bp::make_getter(&ActionDataAbstract::Fu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Fu), "Jacobian of the dynamics")
      .add_property("Lx",
                    bp::make_getter(&ActionDataAbstract::Lx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Lx), "Jacobian of the cost")
      .add_property("Lu",
                    bp::make_getter(&ActionDataAbstract::Lu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Lu), "Jacobian of the cost")
      .add_property("Lxx",
                    bp::make_getter(&ActionDataAbstract::Lxx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Lxx), "Hessian of the cost")
      .add_property("Lxu",
                    bp::make_getter(&ActionDataAbstract::Lxu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Lxu), "Hessian of the cost")
      .add_property("Luu",
                    bp::make_getter(&ActionDataAbstract::Luu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Luu), "Hessian of the cost")
      .add_property("costResiduals",
                    bp::make_getter(&ActionDataAbstract::r, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::r))
      .add_property("Rx",
                    bp::make_getter(&ActionDataAbstract::Rx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Rx))
      .add_property("Ru",
                    bp::make_getter(&ActionDataAbstract::Ru, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Ru));
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_