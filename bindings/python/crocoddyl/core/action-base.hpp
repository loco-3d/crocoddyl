///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_PYTHON_CORE_ACTION_BASE_HPP_
#define CROCODDYL_PYTHON_CORE_ACTION_BASE_HPP_

#include <crocoddyl/core/action-base.hpp>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class ActionModelAbstract_wrap : public ActionModelAbstract,
                                 public bp::wrapper<ActionModelAbstract> {
 public:
  using ActionModelAbstract::nx_;
  using ActionModelAbstract::ndx_;
  using ActionModelAbstract::nu_;
  using ActionModelAbstract::ncost_;
  using ActionModelAbstract::unone_;

  ActionModelAbstract_wrap(StateAbstract *const state,
                           int nu, int ncost=0) : ActionModelAbstract(state, nu, ncost),
      bp::wrapper<ActionModelAbstract>() {}

  void calc(std::shared_ptr<ActionDataAbstract>& data,
            const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
    return bp::call<void>(this->get_override("calc").ptr(),
                          boost::ref(data),
                          (Eigen::VectorXd) x,
                          (Eigen::VectorXd) u);
  }

  void calcDiff(std::shared_ptr<ActionDataAbstract>& data,
                 const Eigen::Ref<const Eigen::VectorXd>& x,
                 const Eigen::Ref<const Eigen::VectorXd>& u,
                 const bool& recalc=true) {
    return bp::call<void>(this->get_override("calcDiff").ptr(),
                          boost::ref(data),
                          (Eigen::VectorXd) x,
                          (Eigen::VectorXd) u,
                          recalc);
  }

  std::shared_ptr<ActionDataAbstract> createData() {
    return std::make_shared<ActionDataAbstract>(this);
  }
};

struct ActionDataAbstract_wrap : public ActionDataAbstract,
                                 public bp::wrapper<ActionDataAbstract> {
  ActionDataAbstract_wrap(ActionModelAbstract* model) : ActionDataAbstract(model),
      bp::wrapper<ActionDataAbstract>() {}
};


void exposeActionAbstract() {
  bp::class_<ActionModelAbstract_wrap, boost::noncopyable>(
      "ActionModelAbstract",
      R"(Abstract class for action models.

        In crocoddyl, an action model combines dynamics and cost data. Each node, in our optimal 
        control problem, is described through an action model. Every time that we want describe
        a problem, we need to provide ways of computing the dynamics, cost functions and their
        derivatives. These computations are mainly carry on inside calc() and calcDiff(),
        respectively.)",
      bp::init<StateAbstract*, int, bp::optional<int>>(bp::args(" self", " state", " nu", " ncost"),
                         R"(Initialize the action model.

:param state: state description,
:param nu: dimension of control vector,
:param ncost: dimension of the cost-residual vector)"))
      .def("calc", pure_virtual(&ActionModelAbstract_wrap::calc), bp::args(" self", " data", " x", " u"),
           R"(Compute the next state and cost value.

First, it describes the time-discrete evolution of our dynamical system
in which we obtain the next discrete state. Additionally it computes
the cost value associated to this discrete state and control pair.
:param model: action model
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
action model (i.e. dynamical system and cost function).
:param model: action model
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
      .add_property("unone", bp::make_getter(&ActionModelAbstract_wrap::unone_,
                             bp::return_value_policy<bp::return_by_value>()),
                             "default control vector");
      // .add_property("state", &ActionModelAbstract::get_state, bp::return_value_policy<bp::manage_new_object>());

  boost::python::register_ptr_to_python<std::shared_ptr<ActionDataAbstract>>();

  bp::class_<ActionDataAbstract_wrap, boost::noncopyable>(
      "ActionDataAbstract",
      R"(Abstract class for action datas.

        In crocoddyl, an action data contains all the required information for processing an
        user-defined action model. The action data typically is allocated onces by running
        model.createData() and contains the first- and second- order derivatives of the dynamics
        and cost function, respectively.)",
      bp::init<ActionModelAbstract*>(bp::args(" self", " model"),
                         R"(Create common data shared between AMs.

In crocoddyl, an action data might use an externally defined cost data.
If so, you need to pass your own cost data using costData. Otherwise
it will be allocated here.
:param model: action model)"))
      .add_property("cost", bp::make_getter(&ActionDataAbstract_wrap::cost, bp::return_value_policy<bp::return_by_value>()),
                             bp::make_setter(&ActionDataAbstract_wrap::cost))
      .add_property("xnext", bp::make_getter(&ActionDataAbstract_wrap::xnext, bp::return_value_policy<bp::return_by_value>()),
                             bp::make_setter(&ActionDataAbstract_wrap::xnext))
      .add_property("Fx", bp::make_getter(&ActionDataAbstract_wrap::Fx, bp::return_value_policy<bp::return_by_value>()),
                          bp::make_setter(&ActionDataAbstract_wrap::Fx))
      .add_property("Fu", bp::make_getter(&ActionDataAbstract_wrap::Fu, bp::return_value_policy<bp::return_by_value>()),
                          bp::make_setter(&ActionDataAbstract_wrap::Fu))
      .add_property("Lx", bp::make_getter(&ActionDataAbstract_wrap::Lx, bp::return_value_policy<bp::return_by_value>()),
                          bp::make_setter(&ActionDataAbstract_wrap::Lx))
      .add_property("Lu", bp::make_getter(&ActionDataAbstract_wrap::Lu, bp::return_value_policy<bp::return_by_value>()),
                          bp::make_setter(&ActionDataAbstract_wrap::Lu))
      .add_property("Lxx", bp::make_getter(&ActionDataAbstract_wrap::Lxx, bp::return_value_policy<bp::return_by_value>()),
                           bp::make_setter(&ActionDataAbstract_wrap::Lxx))
      .add_property("Lxu", bp::make_getter(&ActionDataAbstract_wrap::Lxu, bp::return_value_policy<bp::return_by_value>()),
                           bp::make_setter(&ActionDataAbstract_wrap::Lxu))
      .add_property("Luu", bp::make_getter(&ActionDataAbstract_wrap::Luu, bp::return_value_policy<bp::return_by_value>()),
                           bp::make_setter(&ActionDataAbstract_wrap::Luu))
      .add_property("costResiduals", bp::make_getter(&ActionDataAbstract_wrap::r, bp::return_value_policy<bp::return_by_value>()),
                                     bp::make_setter(&ActionDataAbstract_wrap::r))
      .add_property("Rx", bp::make_getter(&ActionDataAbstract_wrap::Rx, bp::return_value_policy<bp::return_by_value>()),
                          bp::make_setter(&ActionDataAbstract_wrap::Rx))
      .add_property("Ru", bp::make_getter(&ActionDataAbstract_wrap::Ru, bp::return_value_policy<bp::return_by_value>()),
                          bp::make_setter(&ActionDataAbstract_wrap::Ru));
}

}  // namespace python
}  // namespace crocoddyl

#endif  // CROCODDYL_PYTHON_CORE_ACTION_BASE_HPP_