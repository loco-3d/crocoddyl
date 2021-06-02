///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/pair-collisions.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelPairCollisionsTpl<Scalar>::ResidualModelPairCollisionsTpl(boost::shared_ptr<StateMultibody> state,
                                                         const std::size_t& nu,
                                                         boost::shared_ptr<GeometryModel> geom_model,
                                                         const pinocchio::PairIndex& pair_id,
                                                         const pinocchio::JointIndex& joint_id)
    : Base(state, 3, nu, true, false, false), geom_model_(geom_model), pin_model_(*state->get_pinocchio()), pair_id_(pair_id), joint_id_(joint_id) 
{}

template <typename Scalar>
ResidualModelPairCollisionsTpl<Scalar>::~ResidualModelPairCollisionsTpl() {}

template <typename Scalar>
void ResidualModelPairCollisionsTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract> &data,
                                               const Eigen::Ref<const VectorXs> &x,
                                               const Eigen::Ref<const VectorXs> &) {
  Data* d  = static_cast<Data*>(data.get());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());

  // omputes the distance for the collision pair pair_id_
  pinocchio::updateGeometryPlacements(pin_model_, *d->pinocchio, *geom_model_.get(), d->geom_data, q);
  pinocchio::computeDistance(*geom_model_.get(), d->geom_data, pair_id_);

  //calculate residual
  data->r = d->geom_data.distanceResults[pair_id_].nearest_points[0] -
    d->geom_data.distanceResults[pair_id_].nearest_points[1];
}

template <typename Scalar>
void ResidualModelPairCollisionsTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                   const Eigen::Ref<const VectorXs> &x,
                                                   const Eigen::Ref<const VectorXs> &) {
  Data* d  = static_cast<Data*>(data.get());
  
  const std::size_t& nv = state_->get_nv();
        
  // Calculate the Jacobian at the joint
  const pinocchio::SE3Tpl<Scalar>& oMi = d->pinocchio->oMi[joint_id_];

  // Calculate the vector from the joint jointId to the collision p1, expressed in world frame
  const Vector3s p1_local_world = d->geom_data.distanceResults[pair_id_].nearest_points[0] - oMi.translation();
  d->J.setZero();
  pinocchio::getJointJacobian(pin_model_, *d->pinocchio, joint_id_,
                              pinocchio::LOCAL_WORLD_ALIGNED, d->J);
  
  // Calculate the Jacobian at p1
  d->J.template topRows<3>().noalias() += pinocchio::skew(p1_local_world).transpose() * (d->J.template bottomRows<3>());
  
  // --- Compute the residual derivatives ---
  d->Rx.topLeftCorner(3,nv).noalias() = d->J.template topRows<3>();
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelPairCollisionsTpl<Scalar>::createData(
    DataCollectorAbstract *const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
const pinocchio::GeometryModel& ResidualModelPairCollisionsTpl<Scalar>::get_geometryModel() const {
  return *geom_model_.get();
}

}  // namespace crocoddyl
