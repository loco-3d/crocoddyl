///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
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
                                                         const pinocchio::GeometryModel& geom_model,
                                                         pinocchio::GeometryData& geom_data,
                                                         const pinocchio::PairIndex pair_id,
                                                         const pinocchio::JointIndex joint_id)
    : Base(state, activation, nu), geom_model_(geom_model), geom_data_(geom_data), pair_id_(pair_id), joint_id_(joint_id), p1_{}
{}

template <typename Scalar>
CostModelPairCollisionsTpl<Scalar>::~CostModelPairCollisionsTpl() {}

template <typename Scalar>
void CostModelPairCollisionsTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                           const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  auto& d  = *static_cast<Data*>(data.get());

  const auto q = x.head(state_->get_nq());
  const auto& pin_model = *(state_->get_pinocchio());
  auto& pin_data  = *(d.pinocchio);

  // This function calls forwardKinematics on the whole body
  // and computes the distances for each and every pair
  // We need to find a way to only recompute what is needed
  pinocchio::computeDistances(pin_model, pin_data, geom_model_, geom_data_, q);
  const auto& distance_result = geom_data_.distanceResults[pair_id_];

  p1_ = distance_result.nearest_points[0]; 
  auto p2 = distance_result.nearest_points[1];
        
  //calculate residual
  d.r = p1_ - p2;
       
  auto& activation = *static_cast<Activation*>(activation_.get()); 
  activation.calc(d.activation, d.r);
  d.cost = d.activation->a_value;
}

template <typename Scalar>
void CostModelPairCollisionsTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                               const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  calc(data, x, u); // The results of calc are needed here

  auto& d = *static_cast<Data*>(data.get());

  const auto q = x.head(state_->get_nq());
  const auto& pin_model = *(state_->get_pinocchio());
  auto& pin_data  = *(d.pinocchio);
 
  pinocchio::computeJointJacobians(pin_model, pin_data, q); // Needed by getJointJacobian
  pinocchio::updateFramePlacements(pin_model, pin_data);

  // --- Calculate the jacobian on p1 ---
        
  // Calculate the Jacobian at the joint
  const auto& oMi = pin_data.oMi[joint_id_];
        
  // Calculate the vector from the joint jointId to the collision p1, expressed in world frame
  auto p1_local_world = p1_ - oMi.translation();
  typename std::remove_reference_t<decltype(pin_data)>::Matrix6x J(6, pin_model.nv); 
  J.setZero();
  pinocchio::getJointJacobian(pin_model, pin_data, joint_id_, pinocchio::LOCAL_WORLD_ALIGNED, J);
  
  // Calculate the Jacobian at p1
  d.J = J.topRows(3) + pinocchio::skew(p1_local_world).transpose() * (J.bottomRows(3)); 
  
  // --- Compute the cost derivatives ---
  auto& activation = *static_cast<Activation*>(activation_.get()); 
  activation.calcDiff(d.activation, d.r);

  // a, b for horizontal stacking
  // a,
  // b for vertical stacking
  d.Rx << d.J, MatrixXs::Zero(activation_->get_nr(), state_->get_nv());
  d.Lx << d.J.transpose() * d.activation->Ar,
          MatrixXs::Zero(state_->get_nv(), 1);
  d.Lxx << d.J.transpose() * d.activation->Arr * d.J, MatrixXs::Zero(state_->get_nv(), state_->get_nv()),
          MatrixXs::Zero(state_->get_nv(), state_->get_ndx());
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelPairCollisionsTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

}  // namespace crocoddyl
