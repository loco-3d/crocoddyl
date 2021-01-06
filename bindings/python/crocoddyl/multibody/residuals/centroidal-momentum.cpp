///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/residuals/centroidal-momentum.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualCentroidalMomentum() {
  typedef Eigen::Matrix<double, 6, 1> Vector6d;

  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelCentroidalMomentum> >();

  bp::class_<ResidualModelCentroidalMomentum, bp::bases<ResidualModelAbstract> >(
      "ResidualModelCentroidalMomentum",
      "This residual function defines the centroidal momentum tracking as r = h - href, with h and href as the "
      "current and reference "
      "centroidal momenta, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, Vector6d, std::size_t>(
          bp::args("self", "state", "href", "nu"),
          "Initialize the centroidal momentum residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param href: reference centroidal momentum\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Vector6d>(
          bp::args("self", "state", "href"),
          "Initialize the centroidal momentum residual model.\n\n"
          "The default nu is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param href: reference centroidal momentum"))
      .def<void (ResidualModelCentroidalMomentum::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                     const Eigen::Ref<const Eigen::VectorXd>&,
                                                     const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelCentroidalMomentum::calc, bp::args("self", "data", "x", "u"),
          "Compute the centroidal momentum residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelCentroidalMomentum::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                     const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelCentroidalMomentum::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                     const Eigen::Ref<const Eigen::VectorXd>&,
                                                     const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelCentroidalMomentum::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the centroidal momentum residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelCentroidalMomentum::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                     const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelCentroidalMomentum::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the centroidal momentum residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property(
          "reference",
          bp::make_function(&ResidualModelCentroidalMomentum::get_reference, bp::return_internal_reference<>()),
          &ResidualModelCentroidalMomentum::set_reference, "reference centroidal momentum");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataCentroidalMomentum> >();

  bp::class_<ResidualDataCentroidalMomentum, bp::bases<ResidualDataAbstract> >(
      "ResidualDataCentroidalMomentum", "Data for centroidal momentum residual.\n\n",
      bp::init<ResidualModelCentroidalMomentum*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create centroidal momentum residual data.\n\n"
          ":param model: centroidal momentum residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("dhd_dq",
                    bp::make_getter(&ResidualDataCentroidalMomentum::dhd_dq, bp::return_internal_reference<>()),
                    "Jacobian of the centroidal momentum")
      .add_property("dhd_dv",
                    bp::make_getter(&ResidualDataCentroidalMomentum::dhd_dv, bp::return_internal_reference<>()),
                    "Jacobian of the centroidal momentum");
}

}  // namespace python
}  // namespace crocoddyl
