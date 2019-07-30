///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_
#define PYTHON_CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_

#include "crocoddyl/multibody/costs/frame-translation.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeCostFrameTranslation() {
  bp::class_<FrameTranslation, boost::noncopyable>(
      "FrameTranslation",
      "Frame translation describe using Pinocchio.\n\n"
      "It defines a frame translation (3D vector) for a given frame ID",
      bp::init<int, Eigen::Vector3d>(bp::args(" self", " frame", " oxf"),
                                     "Initialize the cost model.\n\n"
                                     ":param frame: frame ID\n"
                                     ":param oxf: Frame translation w.r.t. the origin"))
      .def_readwrite("frame", &FrameTranslation::frame, "frame ID")
      .add_property("oxf", bp::make_getter(&FrameTranslation::oxf, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&FrameTranslation::oxf), "frame translation");

  bp::class_<CostModelFrameTranslation, bp::bases<CostModelAbstract> >(
      "CostModelFrameTranslation", bp::init<pinocchio::Model*, ActivationModelAbstract*, FrameTranslation, int>(
                                       bp::args(" self", " model", " activation", " xref", " nu"),
                                       "Initialize the frame translation cost model.\n\n"
                                       ":param model: Pinocchio model of the multibody system\n"
                                       ":param activation: activation model\n"
                                       ":param xref: reference frame translation\n"
                                       ":param nu: dimension of control vector")[bp::with_custodian_and_ward<1, 3>()])
      .def(bp::init<pinocchio::Model*, ActivationModelAbstract*, FrameTranslation>(
          bp::args(" self", " model", " activation", " xref"),
          "Initialize the frame translation cost model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param model: Pinocchio model of the multibody system\n"
          ":param activation: activation model\n"
          ":param xref: reference frame translation")[bp::with_custodian_and_ward<1, 3>()])
      .def(bp::init<pinocchio::Model*, FrameTranslation, int>(
          bp::args(" self", " model", " xref", " nu"),
          "Initialize the control cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(3).\n"
          ":param model: Pinocchio model of the multibody system\n"
          ":param xref: reference frame translation\n"
          ":param nu: dimension of control vector")[bp::with_custodian_and_ward<1, 2>()])
      .def(bp::init<pinocchio::Model*, FrameTranslation>(
          bp::args(" self", " model", " xref"),
          "Initialize the control cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(3), and nu is equals to model.nv.\n"
          ":param model: Pinocchio model of the multibody system\n"
          ":param xref: reference frame translation")[bp::with_custodian_and_ward<1, 2>()])
      .def<void (CostModelFrameTranslation::*)(boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                               const Eigen::VectorXd&)>("calc", &CostModelFrameTranslation::calc_wrap,
                                                                        bp::args(" self", " data", " x", " u=None"),
                                                                        "Compute the frame translation cost.\n\n"
                                                                        ":param data: cost data\n"
                                                                        ":param x: time-discrete state vector\n"
                                                                        ":param u: time-discrete control input")
      .def<void (CostModelFrameTranslation::*)(boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&)>(
          "calc", &CostModelFrameTranslation::calc_wrap, bp::args(" self", " data", " x"))
      .def<void (CostModelFrameTranslation::*)(boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                               const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &CostModelFrameTranslation::calcDiff_wrap,
          bp::args(" self", " data", " x", " u=None", " recalc=True"),
          "Compute the derivatives of the frame translation cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n"
          ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (CostModelFrameTranslation::*)(boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                               const Eigen::VectorXd&)>(
          "calcDiff", &CostModelFrameTranslation::calcDiff_wrap, bp::args(" self", " data", " x", " u"))
      .def<void (CostModelFrameTranslation::*)(boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &CostModelFrameTranslation::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (CostModelFrameTranslation::*)(boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                               const bool&)>("calcDiff", &CostModelFrameTranslation::calcDiff_wrap,
                                                             bp::args(" self", " data", " x", " recalc"))
      .def("createData", &CostModelFrameTranslation::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the frame translation cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return cost data.")
      .add_property("xref", bp::make_function(&CostModelFrameTranslation::get_xref, bp::return_internal_reference<>()),
                    "reference frame translation");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_