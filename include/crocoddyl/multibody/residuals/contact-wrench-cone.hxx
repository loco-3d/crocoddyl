///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelContactWrenchConeTpl<Scalar>::ResidualModelContactWrenchConeTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const WrenchCone& fref, const std::size_t nu, const bool fwddyn)
    : Base(state, fref.get_nf() + 13, nu, fwddyn ? true : false,
           fwddyn ? true : false, true),
      fwddyn_(fwddyn),
      update_jacobians_(true),
      id_(id),
      fref_(fref) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
ResidualModelContactWrenchConeTpl<Scalar>::ResidualModelContactWrenchConeTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const WrenchCone& fref)
    : Base(state, fref.get_nf() + 13),
      fwddyn_(true),
      update_jacobians_(true),
      id_(id),
      fref_(fref) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
void ResidualModelContactWrenchConeTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the residual of the wrench cone. Note that we need to transform the
  // wrench to the contact frame
  data->r.noalias() = fref_.get_A() * d->contact->f.toVector();
}

template <typename Scalar>
void ResidualModelContactWrenchConeTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  data->r.setZero();
}

template <typename Scalar>
void ResidualModelContactWrenchConeTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  if (fwddyn_ || update_jacobians_) {
    updateJacobians(data);
  }
}

template <typename Scalar>
void ResidualModelContactWrenchConeTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  data->Rx.setZero();
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelContactWrenchConeTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  std::shared_ptr<ResidualDataAbstract> d =
      std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  if (!fwddyn_) {
    updateJacobians(d);
  }
  return d;
}

template <typename Scalar>
void ResidualModelContactWrenchConeTpl<Scalar>::updateJacobians(
    const std::shared_ptr<ResidualDataAbstract>& data) {
  Data* d = static_cast<Data*>(data.get());

  const MatrixXs& df_dx = d->contact->df_dx;
  const MatrixXs& df_du = d->contact->df_du;
  const MatrixX6s& A = fref_.get_A();
  data->Rx.noalias() = A * df_dx;
  data->Ru.noalias() = A * df_du;
  update_jacobians_ = false;
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelContactWrenchConeTpl<NewScalar>
ResidualModelContactWrenchConeTpl<Scalar>::cast() const {
  typedef ResidualModelContactWrenchConeTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      id_, fref_.template cast<NewScalar>(), nu_, fwddyn_);
  return ret;
}

template <typename Scalar>
void ResidualModelContactWrenchConeTpl<Scalar>::print(std::ostream& os) const {
  std::shared_ptr<StateMultibody> s =
      std::static_pointer_cast<StateMultibody>(state_);
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  os << "ResidualModelContactWrenchCone {frame="
     << s->get_pinocchio()->frames[id_].name << ", mu=" << fref_.get_mu()
     << ", box=" << fref_.get_box().transpose().format(fmt) << "}";
}

template <typename Scalar>
bool ResidualModelContactWrenchConeTpl<Scalar>::is_fwddyn() const {
  return fwddyn_;
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelContactWrenchConeTpl<Scalar>::get_id()
    const {
  return id_;
}

template <typename Scalar>
const WrenchConeTpl<Scalar>&
ResidualModelContactWrenchConeTpl<Scalar>::get_reference() const {
  return fref_;
}

template <typename Scalar>
void ResidualModelContactWrenchConeTpl<Scalar>::set_id(
    const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelContactWrenchConeTpl<Scalar>::set_reference(
    const WrenchCone& reference) {
  fref_ = reference;
  update_jacobians_ = true;
}

}  // namespace crocoddyl
