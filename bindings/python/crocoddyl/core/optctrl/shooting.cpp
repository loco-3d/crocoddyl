///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <memory>
#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/vector-converter.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"

namespace crocoddyl {
namespace python {

void exposeShootingProblem() {
  // Register custom converters between std::vector and Python list
  typedef boost::shared_ptr<ActionModelAbstract> ActionModelPtr;
  typedef boost::shared_ptr<ActionDataAbstract> ActionDataPtr;
  StdVectorPythonVisitor<ActionModelPtr, std::allocator<ActionModelPtr>, true>::expose("StdVec_ActionModel");
  StdVectorPythonVisitor<ActionDataPtr, std::allocator<ActionDataPtr>, true>::expose("StdVec_ActionData");

  bp::register_ptr_to_python<boost::shared_ptr<ShootingProblem> >();

  bp::class_<ShootingProblem, boost::noncopyable>(
      "ShootingProblem",
      "Declare a shooting problem.\n\n"
      "A shooting problem declares the initial state, a set of running action models and a\n"
      "terminal action model. It has three main methods - calc, calcDiff and rollout. The\n"
      "first computes the set of next states and cost values per each action model. calcDiff\n"
      "updates the derivatives of all action models. The last rollouts the stacks of actions\n"
      "models.",
      bp::init<Eigen::VectorXd, std::vector<boost::shared_ptr<ActionModelAbstract> >,
               boost::shared_ptr<ActionModelAbstract> >(bp::args("self", "x0", "runningModels", "terminalModel"),
                                                        "Initialize the shooting problem and allocate its data.\n\n"
                                                        ":param x0: initial state\n"
                                                        ":param runningModels: running action models (size T)\n"
                                                        ":param terminalModel: terminal action model"))
      .def(bp::init<Eigen::VectorXd, std::vector<boost::shared_ptr<ActionModelAbstract> >,
                    boost::shared_ptr<ActionModelAbstract>, std::vector<boost::shared_ptr<ActionDataAbstract> >,
                    boost::shared_ptr<ActionDataAbstract> >(
          bp::args("self", "x0", "runningModels", "terminalModel", "runningDatas", "terminalData"),
          "Initialize the shooting problem (models and datas).\n\n"
          ":param x0: initial state\n"
          ":param runningModels: running action models (size T)\n"
          ":param terminalModel: terminal action model\n"
          ":param runningDatas: running action datas  (size T)\n"
          ":param terminalData: terminal action data"))
      .def("calc", &ShootingProblem::calc, bp::args("self", "xs", "us"),
           "Compute the cost and the next states.\n\n"
           "For each node k, and along the state xs and control us trajectories, it computes the next state x_{k+1}\n"
           " and cost l_k.\n"
           ":param xs: time-discrete state trajectory (size T+1)\n"
           ":param us: time-discrete control sequence (size T)\n"
           ":returns the total cost value")
      .def("calcDiff", &ShootingProblem::calcDiff, bp::args("self", "xs", "us"),
           "Compute the derivatives of the cost and dynamics.\n\n"
           "For each node k, and along the state x_s and control u_s trajectories, it computes the derivatives of\n"
           "the cost (lx, lu, lxx, lxu, luu) and dynamics (fx, fu).\n"
           ":param xs: time-discrete state trajectory (size T+1)\n"
           ":param us: time-discrete control sequence (size T)\n"
           ":returns the total cost value")
      .def("rollout", &ShootingProblem::rollout_us, bp::args("self", "us"),
           "Integrate the dynamics given a control sequence.\n\n"
           "Rollout the dynamics give a sequence of control commands\n"
           ":param us: time-discrete control sequence (size T)")
      .def("quasiStatic", &ShootingProblem::quasiStatic_xs, bp::args("self", "xs"),
           "Compute the quasi static commands given a state trajectory.\n\n"
           "Generally speaking, it uses Newton-Raphson method for computing the quasi static commands.\n"
           ":param xs: time-discrete state trajectory (size T)")
      .def<void (ShootingProblem::*)(boost::shared_ptr<ActionModelAbstract>, boost::shared_ptr<ActionDataAbstract>)>(
          "circularAppend", &ShootingProblem::circularAppend, bp::args("self", "model", "data"),
          "Circular append the model and data onto the end running node.\n\n"
          "Once we update the end running node, the first running mode is removed as in a circular buffer.\n"
          ":param model: new model\n"
          ":param data: new data")
      .def<void (ShootingProblem::*)(boost::shared_ptr<ActionModelAbstract>)>(
          "circularAppend", &ShootingProblem::circularAppend, bp::args("self", "model"),
          "Circular append the model and data onto the end running node.\n\n"
          "Once we update the end running node, the first running mode is removed as in a circular buffer.\n"
          "Note that this method allocates new data for the end running node.\n"
          ":param model: new model")
      .def("updateNode", &ShootingProblem::updateNode, bp::args("self", "i", "model", "data"),
           "Update the model and data for a specific node.\n\n"
           ":param i: index of the node (0 <= i <= T + 1)\n"
           ":param model: new model\n"
           ":param data: new data")
      .def("updateModel", &ShootingProblem::updateModel, bp::args("self", "i", "model"),
           "Update a model and allocated new data for a specific node.\n\n"
           ":param i: index of the node (0 <= i <= T + 1)\n"
           ":param model: new model")
      .add_property("T", bp::make_function(&ShootingProblem::get_T, bp::return_value_policy<bp::return_by_value>()),
                    "number of running nodes")
      .add_property("x0", bp::make_function(&ShootingProblem::get_x0, bp::return_internal_reference<>()),
                    &ShootingProblem::set_x0, "initial state")
      .add_property(
          "runningModels",
          bp::make_function(&ShootingProblem::get_runningModels, bp::return_value_policy<bp::return_by_value>()),
          &ShootingProblem::set_runningModels, "running models")
      .add_property(
          "terminalModel",
          bp::make_function(&ShootingProblem::get_terminalModel, bp::return_value_policy<bp::return_by_value>()),
          &ShootingProblem::set_terminalModel, "terminal model")
      .add_property(
          "runningDatas",
          bp::make_function(&ShootingProblem::get_runningDatas, bp::return_value_policy<bp::return_by_value>()),
          "running datas")
      .add_property(
          "terminalData",
          bp::make_function(&ShootingProblem::get_terminalData, bp::return_value_policy<bp::return_by_value>()),
          "terminal data")
      .add_property("nx", bp::make_function(&ShootingProblem::get_nx, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of state tuple")
      .add_property("ndx",
                    bp::make_function(&ShootingProblem::get_ndx, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of the tangent space of the state manifold")
      .add_property("nu_max",
                    bp::make_function(&ShootingProblem::get_nu_max, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of the maximun control vector");
}

}  // namespace python
}  // namespace crocoddyl
