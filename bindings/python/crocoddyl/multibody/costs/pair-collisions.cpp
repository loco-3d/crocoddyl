///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/pair-collisions.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostPairCollisions() {  // TODO: Remove once the deprecated update call has been removed in a future
                                   // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelPairCollisions> >();

  bp::class_<CostModelPairCollisions, bp::bases<CostModelResidual> >(
      "CostModelPairCollisions",
      bp::init<boost::shared_ptr<StateMultibody>,
			   double,
			   std::size_t,
			   boost::shared_ptr<pinocchio::GeometryModel>,
			   pinocchio::PairIndex, 
			   pinocchio::JointIndex>(
          bp::args("self", "state", "threshold", "nu","geom_model","pair_id","joint_id"),
          "Initialize the pair collision residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param threshold: threshold for norm2-barrier\n"
          ":param nu: dimension of control vector\n"
          ":param geom_model: geometric model of the multibody system\n"
          ":param pair_id: id of the pair of colliding objects\n"
          ":param joint_id: used to calculate the Jacobian at the joint"));

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl
