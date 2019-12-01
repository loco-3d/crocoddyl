///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_CONTROL_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_CONTROL_HPP_

#include "crocoddyl/multibody/costs/control.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeCostControl() {
  bp::class_<CostModelControl, bp::bases<CostModelAbstract> >(
      "CostModelControl",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, Eigen::VectorXd>(
          bp::args(" self", " state", " activation", " uref"),
          "Initialize the control cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param uref: reference control"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract> >(
          bp::args(" self", " state", " activation"),
          "Initialize the control cost model.\n\n"
          "For this case the default uref is the zeros state, i.e. np.zero(nu), where nu is equals to activation.nr.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, int>(
          bp::args(" self", " state", " activation", " nu"),
          "Initialize the control cost model.\n\n"
          "For this case the default uref is the zeros state, i.e. np.zero(nu).\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Eigen::VectorXd>(
          bp::args(" self", " state", " uref"),
          "Initialize the control cost model.\n\n"
          "For this case the default activation model is quadratic, i.e. crocoddyl.ActivationModelQuad(uref.size()).\n"
          ":param state: state of the multibody system\n"
          ":param uref: reference control"))
      .def(bp::init<boost::shared_ptr<StateMultibody> >(
          bp::args(" self", " state"),
          "Initialize the control cost model.\n\n"
          "For this case the default uref is the zeros vector, i.e. np.zero(model.nv), and\n"
          "activation is quadratic, i.e. crocoddyl.ActivationModelQuad(model.nv), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, int>(
          bp::args(" self", " state", " nu"),
          "Initialize the control cost model.\n\n"
          "For this case the default uref is the zeros vector and the default activation\n"
          "model is quadratic, i.e. crocoddyl.ActivationModelQuad(nu)\n"
          ":param state: state of the multibody system\n"
          ":param nu: dimension of control vector"))
      .def("calc", &CostModelControl::calc_wrap,
           CostModel_calc_wraps(bp::args(" self", " data", " x", " u=None"),
                                "Compute the control cost.\n\n"
                                ":param data: cost data\n"
                                ":param x: time-discrete state vector\n"
                                ":param u: time-discrete control input"))
      .def<void (CostModelControl::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                      const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &CostModelControl::calcDiff_wrap, bp::args(" self", " data", " x", " u=None", " recalc=True"),
          "Compute the derivatives of the control cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n"
          ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (CostModelControl::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                      const Eigen::VectorXd&)>("calcDiff", &CostModelControl::calcDiff_wrap,
                                                               bp::args(" self", " data", " x", " u"))
      .def<void (CostModelControl::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &CostModelControl::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (CostModelControl::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                      const bool&)>("calcDiff", &CostModelControl::calcDiff_wrap,
                                                    bp::args(" self", " data", " x", " recalc"))
      .add_property("uref",
                    bp::make_function(&CostModelControl::get_uref, bp::return_value_policy<bp::return_by_value>()),
                    &CostModelControl::set_uref, "reference control");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_CONTROL_HPP_
