// BSD 3-Clause License
//
// Copyright (C) 2024, LAAS-CNRS.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.

#ifndef COLMPC_RESIDUAL_2_HXX_
#define COLMPC_RESIDUAL_2_HXX_
#ifdef PINOCCHIO_WITH_HPP_FCL

namespace colmpc {
using namespace crocoddyl;

template <typename Scalar>
ResidualDistanceCollision2Tpl<Scalar>::ResidualDistanceCollision2Tpl(
    std::shared_ptr<StateMultibody> state, const std::size_t nu,
    const pinocchio::PairIndex pair_id)
    : Base(state, 1, nu, true, false, false), pair_id_(pair_id) {
  if (static_cast<pinocchio::FrameIndex>(
          state->get_geometry()->collisionPairs.size()) <= pair_id) {
    throw_pretty(
        "Invalid argument: "
        << "the pair index is wrong (it does not exist in the geometry model)");
  }
}

template <typename Scalar>
ResidualDistanceCollision2Tpl<Scalar>::~ResidualDistanceCollision2Tpl() {}

template <typename Scalar>
void ResidualDistanceCollision2Tpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  d->r[0] = d->geometry->distanceResults[pair_id_].min_distance;
}

template <typename Scalar>
void ResidualDistanceCollision2Tpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  const std::size_t nv = state_->get_nv();
  auto state = std::static_pointer_cast<StateMultibody>(state_);
  auto const& gmodel = state->get_geometry();
  auto const& res = d->geometry->distanceResults[pair_id_];

  const auto& cp = gmodel->collisionPairs[pair_id_];
  const auto& geom_1 = gmodel->geometryObjects[cp.first];
  const auto& geom_2 = gmodel->geometryObjects[cp.second];

  pinocchio::getFrameJacobian(*state->get_pinocchio(), *d->pinocchio,
                              geom_1.parentFrame,
                              pinocchio::LOCAL_WORLD_ALIGNED, d->J1);

  pinocchio::getFrameJacobian(*state->get_pinocchio(), *d->pinocchio,
                              geom_2.parentFrame,
                              pinocchio::LOCAL_WORLD_ALIGNED, d->J2);

  // getting the nearest points belonging to the collision shapes
  const Vector3s& cp1 = res.nearest_points[0];
  const Vector3s& cp2 = res.nearest_points[1];
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
      -res.normal.transpose() *
      (d->J1.template topRows<3>() - d->J2.template topRows<3>());
}
template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualDistanceCollision2Tpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
pinocchio::PairIndex ResidualDistanceCollision2Tpl<Scalar>::get_pair_id()
    const {
  return pair_id_;
}

template <typename Scalar>
void ResidualDistanceCollision2Tpl<Scalar>::set_pair_id(
    const pinocchio::PairIndex pair_id) {
  pair_id_ = pair_id;
}

}  // namespace colmpc

#endif  // PINOCCHIO_WITH_HPP_FCL
#endif  // COLMPC_RESIDUAL_2_HXX_
