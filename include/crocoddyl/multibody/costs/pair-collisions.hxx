///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/pair-collisions.hpp"
#include "crocoddyl/core/activations/norm2-barrier.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelPairCollisionsTpl<Scalar>::CostModelPairCollisionsTpl(boost::shared_ptr<StateMultibody> state,
                                                               const Scalar& threshold,
                                                               const std::size_t& nu,
                                                               boost::shared_ptr<GeometryModel> geom_model,
                                                               const pinocchio::PairIndex& pair_id, 
                                                               const pinocchio::JointIndex& joint_id)
    : Base(state,
           boost::make_shared<ActivationModelNorm2Barrier>(3,threshold), 
           boost::make_shared<ResidualModelPairCollisions>(state, nu,geom_model,pair_id,joint_id)) {
  std::cerr << "Deprecated CostModelPairCollisions: Use ResidualModelPairCollisions with CostModelResidual" << std::endl;
}

template <typename Scalar>
CostModelPairCollisionsTpl<Scalar>::~CostModelPairCollisionsTpl() {}

} // namespace crocoddyl
