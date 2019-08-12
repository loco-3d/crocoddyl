///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_

#include "crocoddyl/multibody/costs/frame-placement.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeCostFramePlacement() {
  bp::class_<CostModelFramePlacement, bp::bases<CostModelAbstract> >(
      "CostModelFramePlacement", bp::init<StateMultibody&, ActivationModelAbstract&, FramePlacement, int>(
                                     bp::args(" self", " state", " activation", " Mref", " nu"),
                                     "Initialize the frame placement cost model.\n\n"
                                     ":param state: state of the multibody system\n"
                                     ":param activation: activation model\n"
                                     ":param Mref: reference frame placement\n"
                                     ":param nu: dimension of control vector")[bp::with_custodian_and_ward<1, 3>()])
      .def(bp::init<StateMultibody&, ActivationModelAbstract&, FramePlacement>(
          bp::args(" self", " state", " activation", " Fref"),
          "Initialize the frame placement cost model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param Mref: reference frame placement")[bp::with_custodian_and_ward<1, 3>()])
      .def(bp::init<StateMultibody&, FramePlacement, int>(
          bp::args(" self", " state", " Mref", " nu"),
          "Initialize the control cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6).\n"
          ":param state: state of the multibody system\n"
          ":param Mref: reference frame placement\n"
          ":param nu: dimension of control vector")[bp::with_custodian_and_ward<1, 2>()])
      .def(bp::init<StateMultibody&, FramePlacement>(
          bp::args(" self", " state", " Mref"),
          "Initialize the control cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param Mref: reference frame placement")[bp::with_custodian_and_ward<1, 2>()])
      .def<void (CostModelFramePlacement::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                             const Eigen::VectorXd&)>("calc", &CostModelFramePlacement::calc_wrap,
                                                                      bp::args(" self", " data", " x", " u=None"),
                                                                      "Compute the frame placement cost.\n\n"
                                                                      ":param data: cost data\n"
                                                                      ":param x: time-discrete state vector\n"
                                                                      ":param u: time-discrete control input")
      .def<void (CostModelFramePlacement::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&)>(
          "calc", &CostModelFramePlacement::calc_wrap, bp::args(" self", " data", " x"))
      .def<void (CostModelFramePlacement::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                             const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &CostModelFramePlacement::calcDiff_wrap,
          bp::args(" self", " data", " x", " u=None", " recalc=True"),
          "Compute the derivatives of the frame placement cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n"
          ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (CostModelFramePlacement::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                             const Eigen::VectorXd&)>(
          "calcDiff", &CostModelFramePlacement::calcDiff_wrap, bp::args(" self", " data", " x", " u"))
      .def<void (CostModelFramePlacement::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &CostModelFramePlacement::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (CostModelFramePlacement::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                             const bool&)>("calcDiff", &CostModelFramePlacement::calcDiff_wrap,
                                                           bp::args(" self", " data", " x", " recalc"))
      .def("createData", &CostModelFramePlacement::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the frame placement cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return cost data.")
      .add_property("Mref", bp::make_function(&CostModelFramePlacement::get_Mref, bp::return_internal_reference<>()),
                    "reference frame placement");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_
