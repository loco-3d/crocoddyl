///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/residuals/impulse-wrench-cone.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelImpulseWrenchConeTpl<Scalar>::ResidualModelImpulseWrenchConeTpl(boost::shared_ptr<StateMultibody> state,
                                                                             const pinocchio::FrameIndex id,
                                                                             const WrenchCone& fref)
    : Base(state, fref.get_nf() + 13, 0), id_(id), fref_(fref) {}

template <typename Scalar>
ResidualModelImpulseWrenchConeTpl<Scalar>::~ResidualModelImpulseWrenchConeTpl() {}

template <typename Scalar>
void ResidualModelImpulseWrenchConeTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                     const Eigen::Ref<const VectorXs>&,
                                                     const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the residual of the wrench cone. Note that we need to transform the wrench
  // to the contact frame
  data->r.noalias() = fref_.get_A() * d->impulse->jMf.actInv(d->impulse->f).toVector();
}

template <typename Scalar>
void ResidualModelImpulseWrenchConeTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                         const Eigen::Ref<const VectorXs>&,
                                                         const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  const MatrixXs& df_dx = d->impulse->df_dx;
  const MatrixX6s& A = fref_.get_A();
  data->Rx.noalias() = A * df_dx;
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelImpulseWrenchConeTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelImpulseWrenchConeTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const WrenchConeTpl<Scalar>& ResidualModelImpulseWrenchConeTpl<Scalar>::get_reference() const {
  return fref_;
}

template <typename Scalar>
void ResidualModelImpulseWrenchConeTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelImpulseWrenchConeTpl<Scalar>::set_reference(const WrenchCone& reference) {
  fref_ = reference;
}

}  // namespace crocoddyl
