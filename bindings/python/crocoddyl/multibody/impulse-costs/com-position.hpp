///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSE_COSTS_COM_POSITION_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSE_COSTS_COM_POSITION_HPP_

#include "crocoddyl/multibody/impulse-costs/com-position.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeImpulseCostCoM() {
  bp::class_<ImpulseCostModelCoM, bp::bases<ImpulseCostModelAbstract> >(
      "ImpulseCostModelCoM", bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract> >(
                                 bp::args("self", "state", "activation"),
                                 "Initialize the CoM position cost model.\n\n"
                                 "For this case the default nu is equals to model.nv.\n"
                                 ":param state: state of the multibody system\n"
                                 ":param activation: activation model"))
      .def(bp::init<boost::shared_ptr<StateMultibody> >(
          bp::args("self", "state"),
          "Initialize the CoM position cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(3), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system"))
      .def("calc", &ImpulseCostModelCoM::calc_wrap, bp::args("self", "data", "x"),
           "Compute the CoM position cost.\n\n"
           ":param data: cost data\n"
           ":param x: time-discrete state vector")
      .def<void (ImpulseCostModelCoM::*)(const boost::shared_ptr<ImpulseCostDataAbstract>&, const Eigen::VectorXd&,
                                         const bool&)>(
          "calcDiff", &ImpulseCostModelCoM::calcDiff_wrap, bp::args("self", "data", "x", "recalc"),
          "Compute the derivatives of the CoM position cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param recalc: If true, it updates the state evolution and the cost value (default True).")
      .def<void (ImpulseCostModelCoM::*)(const boost::shared_ptr<ImpulseCostDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &ImpulseCostModelCoM::calcDiff_wrap, bp::args("self", "data", "x"))
      .def("createData", &ImpulseCostModelCoM::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the CoM position cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_COM_POSITION_HPP_
