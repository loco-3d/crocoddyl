///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/costs/centroidal-momentum.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostCentroidalMomentum() {
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  bp::class_<CostModelCentroidalMomentum, bp::bases<CostModelAbstract> >(
      "CostModelCentroidalMomentum",
      "This cost function defines a residual vector as r = h - href, with h and href as the current and reference "
      "centroidal momenta, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, Vector6d, int>(
          bp::args("self", "state", "activation", "href", "nu"),
          "Initialize the centroidal momentum cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param href: reference centroidal momentum\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, Vector6d>(
          bp::args("self", "state", "activation", "href"),
          "Initialize the centroidal momentum cost model.\n\n"
          "The default nu is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param href: reference centroidal momentum"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Vector6d, int>(
          bp::args("self", "state", "href", "nu"),
          "Initialize the centroidal momentum cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          ":param state: state of the multibody system\n"
          ":param href: reference centroidal momentum\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Vector6d>(
          bp::args("self", "state", "href"),
          "Initialize the centroidal momentum cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param href: reference centroidal momentum"))
      .def<void (CostModelCentroidalMomentum::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelCentroidalMomentum::calc, bp::args("self", "data", "x", "u"),
          "Compute the centroidal momentum cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelCentroidalMomentum::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (CostModelCentroidalMomentum::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelCentroidalMomentum::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the centroidal momentum cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelCentroidalMomentum::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelCentroidalMomentum::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the centroidal momentum cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference", &CostModelCentroidalMomentum::get_reference<MathBaseTpl<double>::Vector6s>,
                    &CostModelCentroidalMomentum::set_reference<MathBaseTpl<double>::Vector6s>,
                    "reference centroidal momentum")
      .add_property("href",
                    bp::make_function(&CostModelCentroidalMomentum::get_reference<MathBaseTpl<double>::Vector6s>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelCentroidalMomentum::set_reference<MathBaseTpl<double>::Vector6s>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference centroidal momentum");

  bp::class_<CostDataCentroidalMomentum, bp::bases<CostDataAbstract> >(
      "CostDataCentroidalMomentum", "Data for centroidal momentum cost.\n\n",
      bp::init<CostModelCentroidalMomentum*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create centroidal momentum cost data.\n\n"
          ":param model: centroidal momentum cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("dhd_dq", bp::make_getter(&CostDataCentroidalMomentum::dhd_dq, bp::return_internal_reference<>()),
                    "Jacobian of the centroidal momentum")
      .add_property("dhd_dv", bp::make_getter(&CostDataCentroidalMomentum::dhd_dv, bp::return_internal_reference<>()),
                    "Jacobian of the centroidal momentum");
}

}  // namespace python
}  // namespace crocoddyl
