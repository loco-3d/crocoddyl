///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2022, University of Duisburg-Essen, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/contact-cop-position.hpp"

namespace crocoddyl {

template <typename _Scalar>
ResidualModelContactCoPPositionTpl<_Scalar>::ResidualModelContactCoPPositionTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id, const CoPSupport& cref,
    const std::size_t nu, const bool fwddyn)
    : Base(state, 4, nu, fwddyn ? true : false, fwddyn ? true : false, true),
      fwddyn_(fwddyn),
      update_jacobians_(true),
      id_(id),
      cref_(cref) {}

template <typename _Scalar>
ResidualModelContactCoPPositionTpl<_Scalar>::ResidualModelContactCoPPositionTpl(
    boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id, const CoPSupport& cref)
    : Base(state, 4), fwddyn_(true), update_jacobians_(true), id_(id), cref_(cref) {}

template <typename Scalar>
ResidualModelContactCoPPositionTpl<Scalar>::~ResidualModelContactCoPPositionTpl() {}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                      const Eigen::Ref<const VectorXs>&,
                                                      const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the residual residual r =  A * f
  data->r.noalias() = cref_.get_A() * d->contact->jMf.actInv(d->contact->f).toVector();
}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                      const Eigen::Ref<const VectorXs>&) {
  data->r.setZero();
}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                          const Eigen::Ref<const VectorXs>&,
                                                          const Eigen::Ref<const VectorXs>&) {
  if (fwddyn_ || update_jacobians_) {
    updateJacobians(data);
  }
}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                          const Eigen::Ref<const VectorXs>&) {
  data->Rx.setZero();
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelContactCoPPositionTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  boost::shared_ptr<ResidualDataAbstract> d =
      boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  if (!fwddyn_) {
    updateJacobians(d);
  }
  return d;
}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::updateJacobians(const boost::shared_ptr<ResidualDataAbstract>& data) {
  Data* d = static_cast<Data*>(data.get());

  const MatrixXs& df_dx = d->contact->df_dx;
  const MatrixXs& df_du = d->contact->df_du;
  const Matrix46& A = cref_.get_A();
  data->Rx.noalias() = A * df_dx;
  data->Ru.noalias() = A * df_du;
  update_jacobians_ = false;
}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::print(std::ostream& os) const {
  boost::shared_ptr<StateMultibody> s = boost::static_pointer_cast<StateMultibody>(state_);
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[", "]");
  os << "ResidualModelContactCoPPosition {frame=" << s->get_pinocchio()->frames[id_].name
     << ", box=" << cref_.get_box().transpose().format(fmt) << "}";
}

template <typename Scalar>
bool ResidualModelContactCoPPositionTpl<Scalar>::is_fwddyn() const {
  return fwddyn_;
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelContactCoPPositionTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const CoPSupportTpl<Scalar>& ResidualModelContactCoPPositionTpl<Scalar>::get_reference() const {
  return cref_;
}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::set_reference(const CoPSupport& reference) {
  cref_ = reference;
  update_jacobians_ = true;
}

}  // namespace crocoddyl
