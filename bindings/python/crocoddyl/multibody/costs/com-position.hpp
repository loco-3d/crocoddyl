///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_COM_POSITION_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_COM_POSITION_HPP_

#include "crocoddyl/multibody/costs/com-position.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeCostCoMPosition() {
  bp::class_<CostModelCoMPosition, bp::bases<CostModelAbstract> >(
      "CostModelCoMPosition",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, Eigen::Vector3d, int>(
          bp::args(" self", " state", " activation", " cref", " nu"),
          "Initialize the CoM position cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param cref: reference CoM position\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, Eigen::Vector3d>(
          bp::args(" self", " state", " activation", " cref"),
          "Initialize the CoM position cost model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param cref: reference CoM position"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Eigen::Vector3d, int>(
          bp::args(" self", " state", " cref", " nu"),
          "Initialize the CoM position cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6).\n"
          ":param state: state of the multibody system\n"
          ":param cref: reference CoM position\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Eigen::Vector3d>(
          bp::args(" self", " state", " cref"),
          "Initialize the CoM position cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(3), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param cref: reference CoM position"))
      .def("calc", &CostModelCoMPosition::calc_wrap,
           CostModel_calc_wraps(bp::args(" self", " data", " x", " u=None"),
                                "Compute the CoM position cost.\n\n"
                                ":param data: cost data\n"
                                ":param x: time-discrete state vector\n"
                                ":param u: time-discrete control input"))
      .def<void (CostModelCoMPosition::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                          const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &CostModelCoMPosition::calcDiff_wrap,
          bp::args(" self", " data", " x", " u=None", " recalc=True"),
          "Compute the derivatives of the CoM position cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n"
          ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (CostModelCoMPosition::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                          const Eigen::VectorXd&)>("calcDiff", &CostModelCoMPosition::calcDiff_wrap,
                                                                   bp::args(" self", " data", " x", " u"))
      .def<void (CostModelCoMPosition::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &CostModelCoMPosition::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (CostModelCoMPosition::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                          const bool&)>("calcDiff", &CostModelCoMPosition::calcDiff_wrap,
                                                        bp::args(" self", " data", " x", " recalc"))
      .def("createData", &CostModelCoMPosition::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the CoM position cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("cref",
                    bp::make_function(&CostModelCoMPosition::get_cref, bp::return_value_policy<bp::return_by_value>()),
                    &CostModelCoMPosition::set_cref, "reference CoM position");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_COM_POSITION_HPP_
