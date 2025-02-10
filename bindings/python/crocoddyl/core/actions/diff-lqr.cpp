///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actions/diff-lqr.hpp"

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/diff-action-base.hpp"
#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

BOOST_PYTHON_FUNCTION_OVERLOADS(DifferentialActionModelLQR_Random_wrap,
                                DifferentialActionModelLQR::Random, 2, 4)

void exposeDifferentialActionLQR() {
// TODO: Remove once the deprecated update call has been removed in a future
// release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  boost::python::register_ptr_to_python<
      std::shared_ptr<DifferentialActionModelLQR> >();

  bp::class_<DifferentialActionModelLQR,
             bp::bases<DifferentialActionModelAbstract> >(
      "DifferentialActionModelLQR",
      "Differential action model for linear dynamics and quadratic cost.\n\n"
      "This class implements a linear dynamics, quadratic costs, and linear "
      "constraints (i.e. LQR action). Since the DAM is a second order system, "
      "and the integrated action models are implemented as being second order "
      "integrators. This class implements a second order linear system given "
      "by\n"
      "  x = [q, v]\n"
      "  dv = Fq q + Fv v + Fu u + f0\n"
      "where Fq, Fv, Fu and f are randomly chosen constant terms. On the other "
      "hand, the cost function is given by\n"
      "  l(x,u) = 1/2 [x,u].T [Q N; N.T R] [x,u] + [q,r].T [x,u],\n"
      "and the linear equality and inequality constraints has the form:\n"
      "  g(x,u) = G [x,u] + g<=0\n"
      "  h(x,u) = H [x,u] + h.",
      bp::init<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
               Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>(
          bp::args("self", "Aq", "Av", "B", "Q", "R", "N"),
          "Initialize the differential LQR action model.\n\n"
          ":param Aq: position matrix\n"
          ":param Av: velocity matrix\n"
          ":param B: input matrix\n"
          ":param Q: state weight matrix\n"
          ":param R: input weight matrix\n"
          ":param N: state-input weight matrix"))
      .def(bp::init<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
                    Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
                    Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>(
          bp::args("self", "Aq", "Av", "B", "Q", "R", "N", "f", "q", "r"),
          "Initialize the differential LQR action model.\n\n"
          ":param Aq: position matrix\n"
          ":param Av: velocity matrix\n"
          ":param B: input matrix\n"
          ":param Q: state weight matrix\n"
          ":param R: input weight matrix\n"
          ":param N: state-input weight matrix\n"
          ":param f: dynamics drift\n"
          ":param q: state weight vector\n"
          ":param r: input weight vector"))
      .def(bp::init<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
                    Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd,
                    Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd,
                    Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd,
                    Eigen::VectorXd>(
          bp::args("self", "Aq", "Av", "B", "Q", "R", "N", "G", "H", "f", "q",
                   "r", "g", "h"),
          "Initialize the differential LQR action model.\n\n"
          ":param Aq: position matrix\n"
          ":param Av: velocity matrix\n"
          ":param B: input matrix\n"
          ":param Q: state weight matrix\n"
          ":param R: input weight matrix\n"
          ":param N: state-input weight matrix\n"
          ":param G: state-input inequality constraint matrix\n"
          ":param H: state-input equality constraint matrix\n"
          ":param f: dynamics drift\n"
          ":param q: state weight vector\n"
          ":param r: input weight vector\n"
          ":param g: state-input inequality constraint bias\n"
          ":param h: state-input equality constraint bias"))
      .def(bp::init<std::size_t, std::size_t, bp::optional<bool> >(
          bp::args("self", "nq", "nu", "driftFree"),
          "Initialize the differential LQR action model.\n\n"
          ":param nx: dimension of the state vector\n"
          ":param nu: dimension of the control vector\n"
          ":param driftFree: enable/disable the bias term of the linear "
          "dynamics (default True)"))
      .def<void (DifferentialActionModelLQR::*)(
          const std::shared_ptr<DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &DifferentialActionModelLQR::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the next state and cost value.\n\n"
          "It describes the time-continuous evolution of the LQR system. "
          "Additionally it\n"
          "computes the cost value associated to this discrete state and "
          "control pair.\n"
          ":param data: action data\n"
          ":param x: time-continuous state vector\n"
          ":param u: time-continuous control input")
      .def<void (DifferentialActionModelLQR::*)(
          const std::shared_ptr<DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &DifferentialActionModelAbstract::calc,
          bp::args("self", "data", "x"))
      .def<void (DifferentialActionModelLQR::*)(
          const std::shared_ptr<DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &DifferentialActionModelLQR::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the differential LQR dynamics and cost "
          "functions.\n\n"
          "It computes the partial derivatives of the differential LQR system "
          "and the\n"
          "cost function. It assumes that calc has been run first.\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: action data\n"
          ":param x: time-continuous state vector\n"
          ":param u: time-continuous control input\n")
      .def<void (DifferentialActionModelLQR::*)(
          const std::shared_ptr<DifferentialActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &DifferentialActionModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &DifferentialActionModelLQR::createData,
           bp::args("self"), "Create the differential LQR action data.")
      .def("Random", &DifferentialActionModelLQR::Random,
           DifferentialActionModelLQR_Random_wrap(
               bp::args("nq", "nu", "ng", "nh"),
               "Create a random LQR model.\n\n"
               ":param: nq: position dimension\n"
               ":param nu: control dimension\n"
               ":param ng: inequality constraint dimension (default 0)\n"
               ":param nh: equality constraint dimension (default 0)"))
      .staticmethod("Random")
      .def("setLQR", &DifferentialActionModelLQR::set_LQR,
           bp::args("self", "Aq", "Av", "B", "Q", "R", "N", "f", "q", "r"),
           "Modify the LQR action model.\n\n"
           ":param Aq: position matrix\n"
           ":param Av: velocity matrix\n"
           ":param B: input matrix\n"
           ":param Q: state weight matrix\n"
           ":param R: input weight matrix\n"
           ":param N: state-input weight matrix\n"
           ":param f: dynamics drift\n"
           ":param q: state weight vector\n"
           ":param r: input weight vector")
      .add_property("Aq",
                    bp::make_function(&DifferentialActionModelLQR::get_Aq,
                                      bp::return_internal_reference<>()),
                    "position matrix")
      .add_property("Av",
                    bp::make_function(&DifferentialActionModelLQR::get_Av,
                                      bp::return_internal_reference<>()),
                    "velocity matrix")
      .add_property("B",
                    bp::make_function(&DifferentialActionModelLQR::get_B,
                                      bp::return_internal_reference<>()),
                    "input matrix")
      .add_property("f",
                    bp::make_function(&DifferentialActionModelLQR::get_f,
                                      bp::return_internal_reference<>()),
                    "dynamics drift")
      .add_property("Q",
                    bp::make_function(&DifferentialActionModelLQR::get_Q,
                                      bp::return_internal_reference<>()),
                    "state weight matrix")
      .add_property("R",
                    bp::make_function(&DifferentialActionModelLQR::get_R,
                                      bp::return_internal_reference<>()),
                    "input weight matrix")
      .add_property("N",
                    bp::make_function(&DifferentialActionModelLQR::get_N,
                                      bp::return_internal_reference<>()),
                    "state-input weight matrix")
      .add_property("G",
                    bp::make_function(&DifferentialActionModelLQR::get_G,
                                      bp::return_internal_reference<>()),
                    "state-input inequality constraint matrix")
      .add_property("H",
                    bp::make_function(&DifferentialActionModelLQR::get_H,
                                      bp::return_internal_reference<>()),
                    "state-input equality constraint matrix")
      .add_property("q",
                    bp::make_function(&DifferentialActionModelLQR::get_q,
                                      bp::return_internal_reference<>()),
                    "state weight vector")
      .add_property("r",
                    bp::make_function(&DifferentialActionModelLQR::get_r,
                                      bp::return_internal_reference<>()),
                    "input weight vector")
      .add_property("g",
                    bp::make_function(&DifferentialActionModelLQR::get_g,
                                      bp::return_internal_reference<>()),
                    "state-input inequality constraint bias")
      .add_property("h",
                    bp::make_function(&DifferentialActionModelLQR::get_h,
                                      bp::return_internal_reference<>()),
                    "state-input equality constraint bias")
      // deprecated function
      .add_property(
          "Fq",
          bp::make_function(&DifferentialActionModelLQR::get_Aq,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use Aq.")),
          bp::make_function(&DifferentialActionModelLQR::set_Fq,
                            deprecated<>("Deprecated. Use set_LQR.")),
          "position matrix")
      .add_property(
          "Fv",
          bp::make_function(&DifferentialActionModelLQR::get_Av,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use Av.")),
          bp::make_function(&DifferentialActionModelLQR::set_Fv,
                            deprecated<>("Deprecated. Use set_LQR.")),
          "position matrix")
      .add_property(
          "Fu",
          bp::make_function(&DifferentialActionModelLQR::get_B,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use B.")),
          bp::make_function(&DifferentialActionModelLQR::set_Fu,
                            deprecated<>("Deprecated. Use set_LQR.")),
          "input matrix")
      .add_property(
          "f0",
          bp::make_function(&DifferentialActionModelLQR::get_f,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use f.")),
          bp::make_function(&DifferentialActionModelLQR::set_f0,
                            deprecated<>("Deprecated. Use set_LQR.")),
          "dynamics drift")
      .add_property(
          "lx",
          bp::make_function(&DifferentialActionModelLQR::get_q,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use q.")),
          bp::make_function(&DifferentialActionModelLQR::set_lx,
                            deprecated<>("Deprecated. Use set_LQR.")),
          "state weight vector")
      .add_property(
          "lu",
          bp::make_function(&DifferentialActionModelLQR::get_r,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use r.")),
          bp::make_function(&DifferentialActionModelLQR::set_lu,
                            deprecated<>("Deprecated. Use set_LQR.")),
          "input weight vector")
      .add_property(
          "Lxx",
          bp::make_function(&DifferentialActionModelLQR::get_Q,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use Q.")),
          bp::make_function(&DifferentialActionModelLQR::set_Lxx,
                            deprecated<>("Deprecated. Use set_LQR.")),
          "state weight matrix")
      .add_property(
          "Lxu",
          bp::make_function(&DifferentialActionModelLQR::get_N,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use N.")),
          bp::make_function(&DifferentialActionModelLQR::set_Lxu,
                            deprecated<>("Deprecated. Use set_LQR.")),
          "state-input weight matrix")
      .add_property(
          "Luu",
          bp::make_function(&DifferentialActionModelLQR::get_R,
                            deprecated<bp::return_internal_reference<> >(
                                "Deprecated. Use R.")),
          bp::make_function(&DifferentialActionModelLQR::set_Luu,
                            deprecated<>("Deprecated. Use set_LQR.")),
          "input weight matrix")
      .def(CopyableVisitor<DifferentialActionModelLQR>());

  boost::python::register_ptr_to_python<
      std::shared_ptr<DifferentialActionDataLQR> >();

  bp::class_<DifferentialActionDataLQR,
             bp::bases<DifferentialActionDataAbstract> >(
      "DifferentialActionDataLQR",
      "Action data for the differential LQR system.",
      bp::init<DifferentialActionModelLQR*>(
          bp::args("self", "model"),
          "Create differential LQR data.\n\n"
          ":param model: differential LQR action model"))
      .def(CopyableVisitor<DifferentialActionDataLQR>());

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl
