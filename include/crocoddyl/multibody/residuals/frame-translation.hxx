///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>
#include "crocoddyl/multibody/residuals/frame-translation.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelFrameTranslationTpl<Scalar>::ResidualModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state,
                                                                           const pinocchio::FrameIndex id,
                                                                           const Vector3s& xref, const std::size_t nu)
    : Base(state, 3, nu, true, false, false), id_(id), xref_(xref), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ResidualModelFrameTranslationTpl<Scalar>::ResidualModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state,
                                                                           const pinocchio::FrameIndex id,
                                                                           const Vector3s& xref)
    : Base(state, 3, true, false, false), id_(id), xref_(xref), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ResidualModelFrameTranslationTpl<Scalar>::~ResidualModelFrameTranslationTpl() {}

template <typename Scalar>
void ResidualModelFrameTranslationTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                    const Eigen::Ref<const VectorXs>&,
                                                    const Eigen::Ref<const VectorXs>&) {
  // Compute the frame translation w.r.t. the reference frame
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
  data->r = d->pinocchio->oMf[id_].translation() - xref_;
}

template <typename Scalar>
void ResidualModelFrameTranslationTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                        const Eigen::Ref<const VectorXs>&,
                                                        const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the derivatives of the frame translation
  const std::size_t nv = state_->get_nv();
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_, pinocchio::LOCAL, d->fJf);
  d->Rx.leftCols(nv).noalias() = d->pinocchio->oMf[id_].rotation() * d->fJf.template topRows<3>();
  ;
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelFrameTranslationTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelFrameTranslationTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s& ResidualModelFrameTranslationTpl<Scalar>::get_reference() const {
  return xref_;
}

template <typename Scalar>
void ResidualModelFrameTranslationTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelFrameTranslationTpl<Scalar>::set_reference(const Vector3s& translation) {
  xref_ = translation;
}

}  // namespace crocoddyl
