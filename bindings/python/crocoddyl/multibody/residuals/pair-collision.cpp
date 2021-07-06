///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, LAAS-CNRS, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifdef PINOCCHIO_WITH_HPP_FCL

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/residuals/pair-collision.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualPairCollision() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelPairCollision> >();

  bp::class_<ResidualModelPairCollision, bp::bases<ResidualModelAbstract> >(
      "ResidualModelPairCollision",
      bp::init<boost::shared_ptr<StateMultibody>, std::size_t, boost::shared_ptr<pinocchio::GeometryModel>,
               pinocchio::PairIndex, pinocchio::JointIndex>(
          bp::args("self", "state", "nu", "geom_model", "pair_id", "joint_id"),
          "Initialize the pair collision residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param nu: dimension of control vector\n"
          ":param geom_model: geometric model of the multibody system\n"
          ":param pair_id: id of the pair of colliding objects\n"
          ":param joint_id: used to calculate the Jacobian at the joint"))
      .def<void (ResidualModelPairCollision::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelPairCollision::calc, bp::args("self", "data", "x", "u"),
          "Compute the pair collision residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (ResidualModelPairCollision::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelPairCollision::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelPairCollision::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the pair collision residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (ResidualModelPairCollision::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelPairCollision::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the pair collision residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined residual.\n"
           ":param data: shared data\n"
           ":return residual data.");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataPairCollision> >();

  bp::class_<ResidualDataPairCollision, bp::bases<ResidualDataAbstract> >(
      "ResidualDataPairCollision", "Data for pair collision residual.\n\n",
      bp::init<ResidualModelPairCollision*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create pair collision residual data.\n\n"
          ":param model: pair collision residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataPairCollision::pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("geometry",
                    bp::make_getter(&ResidualDataPairCollision::geometry, bp::return_internal_reference<>()),
                    "pinocchio geometry data");
}

}  // namespace python
}  // namespace crocoddyl
#endif
