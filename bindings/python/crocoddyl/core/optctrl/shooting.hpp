///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_
#define PYTHON_CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_

#include <crocoddyl/core/optctrl/shooting.hpp>
#include <python/crocoddyl/utils.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class ShootingProblem_wrap : public ShootingProblem, public bp::wrapper<ShootingProblem> {
 public:
  using ShootingProblem::T_;

  ShootingProblem_wrap(const Eigen::VectorXd& x0, const bp::list& running_models,
                       ActionModelAbstract* terminal_model) : ShootingProblem(), bp::wrapper<ShootingProblem>() {
    x0_ = x0;
    T_ = len(running_models);
    running_models_ = python_list_to_std_vector<ActionModelAbstract*>(running_models);
    terminal_model_ = terminal_model;
    allocateData();
  }

  double calc_wrap(const bp::list& xs, const bp::list& us) {
    const std::vector<Eigen::VectorXd>& xs_vec = python_list_to_std_vector<Eigen::VectorXd>(xs);
    const std::vector<Eigen::VectorXd>& us_vec = python_list_to_std_vector<Eigen::VectorXd>(us);
    return calc(xs_vec, us_vec);
  }

  double calcDiff_wrap(const bp::list& xs, const bp::list& us) {
    const std::vector<Eigen::VectorXd>& xs_vec = python_list_to_std_vector<Eigen::VectorXd>(xs);
    const std::vector<Eigen::VectorXd>& us_vec = python_list_to_std_vector<Eigen::VectorXd>(us);
    return calcDiff(xs_vec, us_vec);
  }

  bp::list rollout_wrap(const bp::list& us) {
    std::vector<Eigen::VectorXd> xs_vec;
    const std::vector<Eigen::VectorXd>& us_vec = python_list_to_std_vector<Eigen::VectorXd>(us);
    rollout(us_vec, xs_vec);
    return std_vector_to_python_list(xs_vec);
  }

  bp::list get_runningModels_wrap() { return std_vector_to_python_list(get_runningModels()); }

  bp::list get_runningDatas_wrap() { return std_vector_to_python_list(get_runningDatas()); }

  std::shared_ptr<ActionDataAbstract> get_terminalData_wrap() { return get_terminalData(); }
};


void exposeShootingProblem() {
  bp::class_<ShootingProblem_wrap, boost::noncopyable>(
      "ShootingProblem",
      R"(Declare a shooting problem.

        A shooting problem declares the initial state, a set of running action models and a
        terminal action model. It has two main methods - calc, calcDiff and rollout. The
        first computes the set of next states and cost values per each action model. calcDiff
        updates the derivatives of all action models. The last rollouts the stacks of actions
        models.)",
      bp::init<Eigen::VectorXd, bp::list, ActionModelAbstract*>(
          bp::args(" self", " initialState", " runningModels", " terminalModel"),
          R"(Initialize the shooting problem.

:param initialState: initial state
:param runningModels: running action models
:param terminalModel: terminal action model)"))
      .def("calc", &ShootingProblem_wrap::calc_wrap, bp::args(" self", " xs", " us"),
           R"(Compute the cost and the next states.

First, it computes the next state and cost for each action model
along a state and control trajectory.
:param xs: time-discrete state trajectory
:param us: time-discrete control sequence
:returns the total cost value)")
      .def("calcDiff", &ShootingProblem_wrap::calcDiff_wrap, bp::args(" self", " xs", " us"),
           R"(Compute the cost-and-dynamics derivatives.

These quantities are computed along a given pair of trajectories xs
(states) and us (controls).
:param xs: time-discrete state trajectory
:param us: time-discrete control sequence)")
      .def("rollout", &ShootingProblem_wrap::rollout_wrap, bp::args(" self", " us"),
           R"(Integrate the dynamics given a control sequence.

Rollout the dynamics give a sequence of control commands
:param us: time-discrete control sequence)")
      .add_property("T", &ShootingProblem_wrap::T_, "number of nodes")
      .add_property("runningModels", bp::make_function(&ShootingProblem_wrap::get_runningModels_wrap), "running models")
      .add_property("terminalModel", bp::make_function(&ShootingProblem_wrap::get_terminalModel, bp::return_internal_reference<>()), "terminal model")
      .def_readonly("runningDatas", &ShootingProblem_wrap::get_runningDatas_wrap, "running datas")
      .def_readonly("terminalData", &ShootingProblem_wrap::get_terminalData_wrap, "terminal data");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_