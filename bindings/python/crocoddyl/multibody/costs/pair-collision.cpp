///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/costs/pair-collisions.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"
#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/multibody/geometry.hpp>

namespace crocoddyl {
namespace python {

void exposeCostPairCollisions() {
  bp::class_<CostModelPairCollisions, bp::bases<CostModelAbstract> >(
      "CostModelPairCollisions",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, int, pinocchio::GeometryModel&, pinocchio::GeometryData&, pinocchio::PairIndex, pinocchio::JointIndex>(
          bp::args("self", "state", "activation", "nu"),
          "Initialize the frame placement cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param nu: dimension of control vector\n"
          ":param geom_model: Pinocchio geometric model\n"
          ":param geom_data: Pinocchio geometric data\n"
          ":param pair_id: Collision pair id in the geometric model\n"
          ":param joint_id: Joint id in the Pinocchio model (for Jac calculation)"))
      .def<void (CostModelPairCollisions::*)(const boost::shared_ptr<CostDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelPairCollisions::calc, bp::args("self", "data", "x", "u"),
          "Compute the frame placement cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelPairCollisions::*)(const boost::shared_ptr<CostDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (CostModelPairCollisions::*)(const boost::shared_ptr<CostDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelPairCollisions::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the frame placement cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelPairCollisions::*)(const boost::shared_ptr<CostDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelPairCollisions::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the frame placement cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataPairCollisions> >();

  bp::class_<CostDataPairCollisions, bp::bases<CostDataAbstract> >(
      "CostDataPairCollisions", "Data for pair of collisions cost.\n\n",
      bp::init<CostModelPairCollisions*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create pair of collisions cost data.\n\n"
          ":param model: pair of collisions cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("J", bp::make_getter(&CostDataPairCollisions::J, bp::return_internal_reference<>()),
                    "Jacobian at the error point");
}

}  // namespace python
}  // namespace crocoddyl
