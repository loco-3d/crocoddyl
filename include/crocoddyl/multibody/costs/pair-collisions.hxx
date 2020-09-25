///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS, Airbus
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/pair-collisions.hpp"
#include <hpp/fcl/data_types.h>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/geometry.hpp>

#include <iostream>
#include <string>
#include <sstream>

namespace crocoddyl {

template <typename Scalar>
CostModelPairCollisionsTpl<Scalar>::CostModelPairCollisionsTpl(
                                                         boost::shared_ptr<StateMultibody> state,
                                                         boost::shared_ptr<ActivationModelAbstract> activation,
                                                         const std::size_t& nu,
                                                         boost::shared_ptr<GeometryModel> geom_model,
                                                         const pinocchio::PairIndex& pair_id,
                                                         const pinocchio::JointIndex& joint_id)
    : Base(state, activation, nu), geom_model_(geom_model), pair_id_(pair_id), joint_id_(joint_id), p1_{}
{}

template <typename Scalar>
CostModelPairCollisionsTpl<Scalar>::~CostModelPairCollisionsTpl() {}

template <typename Scalar>
void CostModelPairCollisionsTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                              const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  Data* d  = static_cast<Data*>(data.get());

  const const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const pinocchio::ModelTpl<Scalar>& pin_model = *state_->get_pinocchio().get();

  // This function calls forwardKinematics on the whole body
  // and computes the distances for each and every pair
  // We need to find a way to only recompute what is needed
  pinocchio::computeDistances(pin_model, *d->pinocchio, *geom_model_.get(), d->geom_data, q);

  const auto& distance_result = d->geom_data.distanceResults[pair_id_];

  p1_ = distance_result.nearest_points[0]; 
  auto p2 = distance_result.nearest_points[1];
        
  //calculate residual
  data->r = p1_ - p2;

  activation_->calc(d->activation, d->r);
  d->cost = d->activation->a_value;
}

template <typename Scalar>
void CostModelPairCollisionsTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                               const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  calc(data, x, u); // The results of calc are needed here

  Data* d  = static_cast<Data*>(data.get());

  const const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const pinocchio::ModelTpl<Scalar>& pin_model = *state_->get_pinocchio().get();
 
  pinocchio::computeJointJacobians(pin_model, *d->pinocchio, q); // Needed by getJointJacobian
  pinocchio::updateFramePlacements(pin_model, *d->pinocchio);

  // --- Calculate the jacobian on p1 ---
        
  // Calculate the Jacobian at the joint
  const pinocchio::SE3Tpl<Scalar>& oMi = d->pinocchio->oMi[joint_id_];

  // Calculate the vector from the joint jointId to the collision p1, expressed in world frame
  auto p1_local_world = p1_ - oMi.translation();
  typename std::remove_reference_t<decltype(*d->pinocchio)>::Matrix6x J(6, pin_model.nv); 
  J.setZero();
  pinocchio::getJointJacobian(pin_model, *d->pinocchio, joint_id_, pinocchio::LOCAL_WORLD_ALIGNED, J);
  
  // Calculate the Jacobian at p1
  d->J = J.topRows(3) + pinocchio::skew(p1_local_world).transpose() * (J.bottomRows(3)); 
  
  // --- Compute the cost derivatives ---
  activation_->calcDiff(d->activation, d->r);

  // a, b for horizontal stacking
  // a,
  // b for vertical stacking
  d->Rx << d->J, MatrixXs::Zero(activation_->get_nr(), state_->get_nv());
  d->Lx << d->J.transpose() * d->activation->Ar,
          MatrixXs::Zero(state_->get_nv(), 1);
  d->Lxx << d->J.transpose() * d->activation->Arr * d->J, MatrixXs::Zero(state_->get_nv(), state_->get_nv()),
          MatrixXs::Zero(state_->get_nv(), state_->get_ndx());
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelPairCollisionsTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}


template <typename Scalar>
const pinocchio::GeometryModel& CostModelPairCollisionsTpl<Scalar>::get_geomModel() const {
  return *geom_model_.get();
}
  
}  // namespace crocoddyl


