///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
#define PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_

#include "crocoddyl/core/diff-action-base.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class DifferentialActionModelAbstract_wrap : public DifferentialActionModelAbstract,
                                             public bp::wrapper<DifferentialActionModelAbstract> {
 public:
  using DifferentialActionModelAbstract::ncost_;
  using DifferentialActionModelAbstract::ndx_;
  using DifferentialActionModelAbstract::nout_;
  using DifferentialActionModelAbstract::nq_;
  using DifferentialActionModelAbstract::nu_;
  using DifferentialActionModelAbstract::nv_;
  using DifferentialActionModelAbstract::nx_;
  using DifferentialActionModelAbstract::unone_;

  DifferentialActionModelAbstract_wrap(StateAbstract* const state, int nu, int ncost = 0)
      : DifferentialActionModelAbstract(state, nu, ncost), bp::wrapper<DifferentialActionModelAbstract>() {}

  void calc(std::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) override {
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  void calcDiff(std::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true) override {
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u, recalc);
  }

  std::shared_ptr<DifferentialActionDataAbstract> createData() override {
    return std::make_shared<DifferentialActionDataAbstract>(this);
  }
};

void exposeDifferentialActionAbstract() {
  bp::class_<DifferentialActionModelAbstract_wrap, boost::noncopyable>(
      "DifferentialActionModelAbstract",
      R"(Abstract class for the differential action model.

        In crocoddyl, a differential action model combines dynamics and cost data described in
        continuous time. Each node, in our optimal control problem, is described through an
        action model. Every time that we want describe a problem, we need to provide ways of
        computing the dynamics, cost functions and their derivatives. These computations are
        mainly carry on inside calc() and calcDiff(), respectively.)",
      bp::init<StateAbstract*, int, bp::optional<int>>(bp::args(" self", " state", " nu", " ncost"),
                                                       R"(Initialize the differential action model.

:param state: state
:param nu: dimension of control vector
:param ncost: dimension of cost vector)")[bp::with_custodian_and_ward<1, 2>()])
      .def("calc", pure_virtual(&DifferentialActionModelAbstract_wrap::calc), bp::args(" self", " data", " x", " u"),
           R"(Compute the state evolution and cost value.

First, it describes the time-continuous evolution of our dynamical system
in which along predefined integrated action self we might obtain the
next discrete state. Indeed it computes the time derivatives of the
state from a predefined dynamical system. Additionally it computes the
cost value associated to this state and control pair.
:param data: differential action data
:param x: state vector
:param u: control input)")
      .def("calcDiff", pure_virtual(&DifferentialActionModelAbstract_wrap::calcDiff),
           bp::args(" self", " data", " x", " u", " recalc=True"),
           R"(Compute the derivatives of the dynamics and cost functions.

It computes the partial derivatives of the dynamical system and the cost
function. If recalc == True, it first updates the state evolution and
cost value. This function builds a quadratic approximation of the
time-continuous action model (i.e. dynamical system and cost function).
:param data: differential action data
:param x: state vector
:param u: control input
:param recalc: If true, it updates the state evolution and the cost value.)")
      .def("createData", &DifferentialActionModelAbstract_wrap::createData, bp::args(" self"),
           R"(Create the differential action data.

Each differential action model has its own data that needs to be
allocated. This function returns the allocated data for a predefined
DAM. Note that you need to defined the DifferentialActionDataType inside
your DAM.
:return DAM data.)")
      .add_property("nq", &DifferentialActionModelAbstract_wrap::nq_, "dimension of configuration vector")
      .add_property("nv", &DifferentialActionModelAbstract_wrap::nv_, "dimension of velocity vector")
      .add_property("nu", &DifferentialActionModelAbstract_wrap::nu_, "dimension of control vector")
      .add_property("nx", &DifferentialActionModelAbstract_wrap::nx_, "dimension of state configuration vector")
      .add_property("ndx", &DifferentialActionModelAbstract_wrap::ndx_, "dimension of state tangent vector")
      .add_property("nout", &DifferentialActionModelAbstract_wrap::nout_, "dimension of evolution vector")
      .add_property("ncost", &DifferentialActionModelAbstract_wrap::ncost_, "dimension of cost-residual vector")
      .add_property("unone",
                    bp::make_getter(&DifferentialActionModelAbstract_wrap::unone_,
                                    bp::return_value_policy<bp::return_by_value>()),
                    "default control vector")
      .add_property("State",
                    bp::make_function(&DifferentialActionModelAbstract_wrap::get_state,
                                      bp::return_value_policy<bp::reference_existing_object>()),
                    "state");

  bp::class_<DifferentialActionDataAbstract, std::shared_ptr<DifferentialActionDataAbstract>, boost::noncopyable>(
      "DifferentialActionDataAbstract",
      R"(Abstract class for action datas.

        In crocoddyl, an action data contains all the required information for processing an
        user-defined action model. The action data typically is allocated onces by running
        model.createData() and contains the first- and second- order derivatives of the dynamics
        and cost function, respectively.)",
      bp::init<DifferentialActionModelAbstract*>(bp::args(" self", " model"),
                                                 R"(Create common data shared between DAMs.

The differentail action data uses the model in order to first process it.
:param model: differential action model)"))
      .add_property(
          "cost",
          bp::make_getter(&DifferentialActionDataAbstract::cost, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::cost), "cost value")
      .add_property(
          "xout",
          bp::make_getter(&DifferentialActionDataAbstract::xout, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::xout), "evolution state")
      .add_property(
          "Fx", bp::make_getter(&DifferentialActionDataAbstract::Fx, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Fx), "Jacobian of the dynamics")
      .add_property(
          "Fu", bp::make_getter(&DifferentialActionDataAbstract::Fu, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Fu), "Jacobian of the dynamics")
      .add_property(
          "Lx", bp::make_getter(&DifferentialActionDataAbstract::Lx, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Lx), "Jacobian of the cost")
      .add_property(
          "Lu", bp::make_getter(&DifferentialActionDataAbstract::Lu, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Lu), "Jacobian of the cost")
      .add_property(
          "Lxx", bp::make_getter(&DifferentialActionDataAbstract::Lxx, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Lxx), "Hessian of the cost")
      .add_property(
          "Lxu", bp::make_getter(&DifferentialActionDataAbstract::Lxu, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Lxu), "Hessian of the cost")
      .add_property(
          "Luu", bp::make_getter(&DifferentialActionDataAbstract::Luu, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Luu), "Hessian of the cost")
      .add_property(
          "costResiduals",
          bp::make_getter(&DifferentialActionDataAbstract::r, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::r))
      .add_property(
          "Rx", bp::make_getter(&DifferentialActionDataAbstract::Rx, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Rx))
      .add_property(
          "Ru", bp::make_getter(&DifferentialActionDataAbstract::Ru, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Ru));
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_