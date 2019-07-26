///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_
#define PYTHON_CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_

#include "crocoddyl/multibody/costs/frame-placement.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeCostFramePlacement() {
  bp::class_<FramePlacement, boost::noncopyable>(
      "FramePlacement",
      "Frame placement describe using Pinocchio.\n\n"
      "It defines a frame placement (SE(3) point) for a given frame ID",
      bp::init<int, pinocchio::SE3>(bp::args(" self", " frame", " oMf"),
                                    "Initialize the cost model.\n\n"
                                    ":param frame: frame ID\n"
                                    ":param oMf: Frame placement w.r.t. the origin"))
      .def_readwrite("frame", &FramePlacement::frame, "frame ID")
      .add_property("oMf",
                    bp::make_getter(&FramePlacement::oMf, bp::return_value_policy<bp::reference_existing_object>()),
                    "frame placement");

  bp::class_<CostModelFramePlacement, bp::bases<CostModelAbstract> >(
      "CostModelFramePlacement", bp::init<pinocchio::Model*, ActivationModelAbstract*, FramePlacement, int>(
                                     bp::args(" self", " model", " activation", " Fref", " nu"),
                                     "Initialize the frame placement cost model.\n\n"
                                     ":param model: Pinocchio model of the multibody system\n"
                                     ":param activation: activation model\n"
                                     ":param Fref: reference frame placement\n"
                                     ":param nu: dimension of control vector")[bp::with_custodian_and_ward<1, 3>()])
      // .def(bp::init<pinocchio::Model*, ActivationModelAbstract*>(
      //     bp::args(" self", " model", " activation"),
      //     "Initialize the control cost model.\n\n"
      //     "For this case the default uref is the zeros state, i.e. np.zero(nu), where nu is equals to
      //     activation.nr.\n"
      //     ":param model: Pinocchio model of the multibody system\n"
      //     ":param activation: activation model")[bp::with_custodian_and_ward<1, 3>()])
      // .def(bp::init<pinocchio::Model*, ActivationModelAbstract*, int>(
      //     bp::args(" self", " model", " activation", " nu"),
      //     "Initialize the control cost model.\n\n"
      //     "For this case the default uref is the zeros state, i.e. np.zero(nu).\n"
      //     ":param model: Pinocchio model of the multibody system\n"
      //     ":param activation: activation model\n"
      //     ":param nu: dimension of control vector")[bp::with_custodian_and_ward<1, 3>()])
      // .def(bp::init<pinocchio::Model*, Eigen::VectorXd>(
      //     bp::args(" self", " model", " uref"),
      //     "Initialize the control cost model.\n\n"
      //     "For this case the default activation model is quadratic, i.e.
      //     crocoddyl.ActivationModelQuad(uref.size()).\n"
      //     ":param model: Pinocchio model of the multibody system\n"
      //     ":param uref: reference control")[bp::with_custodian_and_ward<1, 2>()])
      // .def(bp::init<pinocchio::Model*>(
      //     bp::args(" self", " model"),
      //     "Initialize the control cost model.\n\n"
      //     "For this case the default uref is the zeros vector, i.e. np.zero(model.nv), and\n"
      //     "activation is quadratic, i.e. crocoddyl.ActivationModelQuad(model.nv), and nu is equals to model.nv.\n"
      //     ":param model: Pinocchio model of the multibody system")[bp::with_custodian_and_ward<1, 2>()])
      // .def(bp::init<pinocchio::Model*, int>(
      //     bp::args(" self", " model", " nu"),
      //     "Initialize the control cost model.\n\n"
      //     "For this case the default uref is the zeros vector and the default activation\n"
      //     "model is quadratic, i.e. crocoddyl.ActivationModelQuad(nu)\n"
      //     ":param model: Pinocchio model of the multibody system\n"
      //     ":param nu: dimension of control vector")[bp::with_custodian_and_ward<1, 2>()])
      .def<void (CostModelFramePlacement::*)(boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                             const Eigen::VectorXd&)>("calc", &CostModelFramePlacement::calc_wrap,
                                                                      bp::args(" self", " data", " x", " u=None"),
                                                                      "Compute the frame placement cost.\n\n"
                                                                      ":param data: cost data\n"
                                                                      ":param x: time-discrete state vector\n"
                                                                      ":param u: time-discrete control input")
      .def<void (CostModelFramePlacement::*)(boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&)>(
          "calc", &CostModelFramePlacement::calc_wrap, bp::args(" self", " data", " x"))
      .def<void (CostModelFramePlacement::*)(boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                             const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &CostModelFramePlacement::calcDiff_wrap,
          bp::args(" self", " data", " x", " u=None", " recalc=True"),
          "Compute the derivatives of the frame placement cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n"
          ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (CostModelFramePlacement::*)(boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                             const Eigen::VectorXd&)>(
          "calcDiff", &CostModelFramePlacement::calcDiff_wrap, bp::args(" self", " data", " x", " u"))
      .def<void (CostModelFramePlacement::*)(boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &CostModelFramePlacement::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (CostModelFramePlacement::*)(boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
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

#endif  // PYTHON_CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_