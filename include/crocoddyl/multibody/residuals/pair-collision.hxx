///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2022, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifdef PINOCCHIO_WITH_HPP_FCL

#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/multibody/fcl.hpp>

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelPairCollisionTpl<Scalar>::ResidualModelPairCollisionTpl(
    std::shared_ptr<StateMultibody> state, const std::size_t nu,
    std::shared_ptr<GeometryModel> geom_model,
    const pinocchio::PairIndex pair_id, const pinocchio::JointIndex joint_id)
    : Base(state, 3, nu, true, false, false),
      pin_model_(*state->get_pinocchio()),
      geom_model_(geom_model),
      pair_id_(pair_id),
      joint_id_(joint_id) {
  if (static_cast<pinocchio::FrameIndex>(geom_model->collisionPairs.size()) <=
      pair_id) {
    throw_pretty(
        "Invalid argument: "
        << "the pair index is wrong (it does not exist in the geometry model)");
  }
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->njoints) <=
      joint_id) {
    throw_pretty(
        "Invalid argument: "
        << "the joint index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
ResidualModelPairCollisionTpl<Scalar>::~ResidualModelPairCollisionTpl() {}

template <typename Scalar>
void ResidualModelPairCollisionTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());

  // computes the distance for the collision pair pair_id_
  pinocchio::updateGeometryPlacements(pin_model_, *d->pinocchio,
                                      *geom_model_.get(), d->geometry, q);
  pinocchio::computeDistance(*geom_model_.get(), d->geometry, pair_id_);

  // calculate residual
  data->r = d->geometry.distanceResults[pair_id_].nearest_points[0] -
            d->geometry.distanceResults[pair_id_].nearest_points[1];
}

template <typename Scalar>
void ResidualModelPairCollisionTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &, const Eigen::Ref<const VectorXs> &) {
  Data *d = static_cast<Data *>(data.get());

  const std::size_t nv = state_->get_nv();

  // calculate the vector from the joint jointId to the collision p1, expressed
  // in world frame
  d->d = d->geometry.distanceResults[pair_id_].nearest_points[0] -
         d->pinocchio->oMi[joint_id_].translation();
  pinocchio::getJointJacobian(pin_model_, *d->pinocchio, joint_id_,
                              pinocchio::LOCAL_WORLD_ALIGNED, d->J);

  // calculate the Jacobian at p1
  d->J.template topRows<3>().noalias() +=
      pinocchio::skew(d->d).transpose() * d->J.template bottomRows<3>();

  // compute the residual derivatives
  d->Rx.topLeftCorner(3, nv) = d->J.template topRows<3>();
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelPairCollisionTpl<Scalar>::createData(
    DataCollectorAbstract *const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
const pinocchio::GeometryModel &
ResidualModelPairCollisionTpl<Scalar>::get_geometry() const {
  return *geom_model_.get();
}

}  // namespace crocoddyl

#endif  // PINOCCHIO_WITH_HPP_FCL
