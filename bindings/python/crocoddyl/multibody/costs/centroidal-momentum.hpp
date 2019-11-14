///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_

#include "crocoddyl/multibody/costs/centroidal-momentum.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeCostCentroidalMomentum() {
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  bp::class_<CostModelCentroidalMomentum, bp::bases<CostModelAbstract> >(
      "CostModelCentroidalMomentum",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, Vector6d, int>(
          bp::args(" self", " state", " activation", " ref", " nu"),
          "Initialize the centroidal momentum cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param ref: reference centroidal momentum\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, Vector6d>(
          bp::args(" self", " state", " activation", " ref"),
          "Initialize the centroidal momentum cost model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param ref: reference centroidal momentum"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Vector6d, int>(
          bp::args(" self", " state", " ref", " nu"),
          "Initialize the centroidal momentum cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6).\n"
          ":param state: state of the multibody system\n"
          ":param ref: reference centroidal momentum\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Vector6d>(
          bp::args(" self", " state", " ref"),
          "Initialize the centroidal momentum cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(3), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param ref: reference centroidal momentum"))
      .def("calc", &CostModelCentroidalMomentum::calc_wrap,
           CostModel_calc_wraps(bp::args(" self", " data", " x", " u=None"),
                                "Compute the centroidal momentum cost.\n\n"
                                ":param data: cost data\n"
                                ":param x: time-discrete state vector\n"
                                ":param u: time-discrete control input"))
      .def<void (CostModelCentroidalMomentum::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                                 const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &CostModelCentroidalMomentum::calcDiff_wrap,
          bp::args(" self", " data", " x", " u=None", " recalc=True"),
          "Compute the derivatives of the centroidal momentum cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n"
          ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (CostModelCentroidalMomentum::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                                 const Eigen::VectorXd&)>(
          "calcDiff", &CostModelCentroidalMomentum::calcDiff_wrap, bp::args(" self", " data", " x", " u"))
      .def<void (CostModelCentroidalMomentum::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &CostModelCentroidalMomentum::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (CostModelCentroidalMomentum::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                                 const bool&)>("calcDiff", &CostModelCentroidalMomentum::calcDiff_wrap,
                                                               bp::args(" self", " data", " x", " recalc"))
      .def("createData", &CostModelCentroidalMomentum::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the centroidal momentum cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return cost data.")
      .add_property(
          "ref",
          bp::make_function(&CostModelCentroidalMomentum::get_ref, bp::return_value_policy<bp::return_by_value>()),
          "reference centroidal momentum");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_
