// BSD 3-Clause License
//
// Copyright (C) 2024, LAAS-CNRS.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.

#ifndef COLMPC_RESIDUAL_HXX_
#define COLMPC_RESIDUAL_HXX_
#ifdef PINOCCHIO_WITH_HPP_FCL

#include <pinocchio/collision/fcl-pinocchio-conversions.hpp>

#include "colmpc/residual-distance-collision.hpp"

namespace colmpc {
using namespace crocoddyl;

template <typename Scalar>
ResidualDistanceCollisionTpl<Scalar>::ResidualDistanceCollisionTpl(
    std::shared_ptr<crocoddyl::StateMultibody> state, const std::size_t nu,
    std::shared_ptr<GeometryModel> geom_model,
    const pinocchio::PairIndex pair_id)
    : Base(state, 1, nu, true, false, false),
      pin_model_(*state->get_pinocchio()),
      geom_model_(geom_model),
      pair_id_(pair_id) {
  if (static_cast<pinocchio::FrameIndex>(geom_model_->collisionPairs.size()) <=
      pair_id) {
    throw_pretty(
        "Invalid argument: "
        << "the pair index is wrong (it does not exist in the geometry model)");
  }
}

template <typename Scalar>
ResidualDistanceCollisionTpl<Scalar>::~ResidualDistanceCollisionTpl() {}

template <typename Scalar>
void ResidualDistanceCollisionTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // clear the coal results
  d->res.clear();

  // computes the distance for the collision pair pair_id_
  const auto& cp = geom_model_->collisionPairs[pair_id_];
  const auto& geom_1 = geom_model_->geometryObjects[cp.first];
  const auto& geom_2 = geom_model_->geometryObjects[cp.second];
  const pinocchio::Model::JointIndex joint_id_1 = geom_1.parentJoint;
  const pinocchio::Model::JointIndex joint_id_2 = geom_2.parentJoint;

  if (joint_id_1 > 0) {
    d->oMg_id_1 = d->pinocchio->oMi[joint_id_1] * geom_1.placement;
  } else {
    d->oMg_id_1 = geom_1.placement;
  }

  if (joint_id_2 > 0) {
    d->oMg_id_2 = d->pinocchio->oMi[joint_id_2] * geom_2.placement;
  } else {
    d->oMg_id_2 = geom_2.placement;
  }

  d->r[0] = coal::distance(geom_1.geometry.get(), toFclTransform3f(d->oMg_id_1),
                           geom_2.geometry.get(), toFclTransform3f(d->oMg_id_2),
                           d->req, d->res);
}

template <typename Scalar>
void ResidualDistanceCollisionTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  const std::size_t nv = state_->get_nv();

  const auto& cp = geom_model_->collisionPairs[pair_id_];
  const auto& geom_1 = geom_model_->geometryObjects[cp.first];
  const auto& geom_2 = geom_model_->geometryObjects[cp.second];

  pinocchio::getFrameJacobian(pin_model_, *d->pinocchio, geom_1.parentFrame,
                              pinocchio::LOCAL_WORLD_ALIGNED, d->J1);

  pinocchio::getFrameJacobian(pin_model_, *d->pinocchio, geom_2.parentFrame,
                              pinocchio::LOCAL_WORLD_ALIGNED, d->J2);

  // getting the nearest points belonging to the collision shapes
  const Vector3s& cp1 = d->res.nearest_points[0];
  const Vector3s& cp2 = d->res.nearest_points[1];
  d->cp1 = cp1;
  d->cp2 = cp2;
  // Transport the jacobian of frame 1 into the jacobian associated to cp1
  // Vector from frame 1 center to p1
  d->f1p1 = cp1 - d->pinocchio->oMf[geom_1.parentFrame].translation();
  d->f1Mp1.setIdentity();
  d->f1Mp1.translation(d->f1p1);
  // todo change me for simpler
  d->J1 = d->f1Mp1.toActionMatrixInverse() * d->J1;
  // Transport the jacobian of frame 2 into the jacobian associated to cp2
  // Vector from frame 2 center to p2
  d->f2p2 = cp2 - d->pinocchio->oMf[geom_2.parentFrame].translation();
  d->f2Mp2.setIdentity();
  d->f2Mp2.translation(d->f2p2);
  d->J2 = d->f2Mp2.toActionMatrixInverse() * d->J2;

  // calculate the Jacobian
  // compute the residual derivatives
  data->Rx.leftCols(nv) =
      -d->res.normal.transpose() *
      (d->J1.template topRows<3>() - d->J2.template topRows<3>());
}
template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar>>
ResidualDistanceCollisionTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
const pinocchio::GeometryModel&
ResidualDistanceCollisionTpl<Scalar>::get_geometry() const {
  return *geom_model_.get();
}

template <typename Scalar>
pinocchio::PairIndex ResidualDistanceCollisionTpl<Scalar>::get_pair_id() const {
  return pair_id_;
}

template <typename Scalar>
void ResidualDistanceCollisionTpl<Scalar>::set_pair_id(
    const pinocchio::PairIndex pair_id) {
  pair_id_ = pair_id;
}

}  // namespace colmpc

#endif  // PINOCCHIO_WITH_HPP_FCL
#endif  // COLMPC_RESIDUAL_HXX_
