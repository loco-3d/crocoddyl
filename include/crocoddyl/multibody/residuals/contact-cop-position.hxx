///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, University of Duisburg-Essen,
//                          University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename _Scalar>
ResidualModelContactCoPPositionTpl<_Scalar>::ResidualModelContactCoPPositionTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const CoPSupport& cref, const std::size_t nu, const bool fwddyn)
    : Base(state, 4, nu, fwddyn ? true : false, fwddyn ? true : false, true),
      fwddyn_(fwddyn),
      update_jacobians_(true),
      id_(id),
      cref_(cref) {}

template <typename _Scalar>
ResidualModelContactCoPPositionTpl<_Scalar>::ResidualModelContactCoPPositionTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const CoPSupport& cref)
    : Base(state, 4),
      fwddyn_(true),
      update_jacobians_(true),
      id_(id),
      cref_(cref) {}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the residual residual r =  A * f
  data->r.noalias() = cref_.get_A() * d->contact->f.toVector();
}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  data->r.setZero();
}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  if (fwddyn_ || update_jacobians_) {
    updateJacobians(data);
  }
}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  data->Rx.setZero();
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelContactCoPPositionTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  std::shared_ptr<ResidualDataAbstract> d =
      std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  if (!fwddyn_) {
    updateJacobians(d);
  }
  return d;
}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::updateJacobians(
    const std::shared_ptr<ResidualDataAbstract>& data) {
  Data* d = static_cast<Data*>(data.get());

  const MatrixXs& df_dx = d->contact->df_dx;
  const MatrixXs& df_du = d->contact->df_du;
  const Matrix46& A = cref_.get_A();
  data->Rx.noalias() = A * df_dx;
  data->Ru.noalias() = A * df_du;
  update_jacobians_ = false;
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelContactCoPPositionTpl<NewScalar>
ResidualModelContactCoPPositionTpl<Scalar>::cast() const {
  typedef ResidualModelContactCoPPositionTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      id_, cref_.template cast<NewScalar>(), nu_, fwddyn_);
  return ret;
}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::print(std::ostream& os) const {
  std::shared_ptr<StateMultibody> s =
      std::static_pointer_cast<StateMultibody>(state_);
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  os << "ResidualModelContactCoPPosition {frame="
     << s->get_pinocchio()->frames[id_].name
     << ", box=" << cref_.get_box().transpose().format(fmt) << "}";
}

template <typename Scalar>
bool ResidualModelContactCoPPositionTpl<Scalar>::is_fwddyn() const {
  return fwddyn_;
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelContactCoPPositionTpl<Scalar>::get_id()
    const {
  return id_;
}

template <typename Scalar>
const CoPSupportTpl<Scalar>&
ResidualModelContactCoPPositionTpl<Scalar>::get_reference() const {
  return cref_;
}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::set_id(
    const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelContactCoPPositionTpl<Scalar>::set_reference(
    const CoPSupport& reference) {
  cref_ = reference;
  update_jacobians_ = true;
}

}  // namespace crocoddyl
