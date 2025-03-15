///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/contact-force.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelContactForceTpl<Scalar>::ResidualModelContactForceTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Force& fref, const std::size_t nc, const std::size_t nu,
    const bool fwddyn)
    : Base(state, nc, nu, fwddyn ? true : false, fwddyn ? true : false, true),
      fwddyn_(fwddyn),
      update_jacobians_(true),
      id_(id),
      fref_(fref) {
  if (nc > 6) {
    throw_pretty(
        "Invalid argument in ResidualModelContactForce: nc should be less than "
        "6");
  }
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
ResidualModelContactForceTpl<Scalar>::ResidualModelContactForceTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Force& fref, const std::size_t nc)
    : Base(state, nc),
      fwddyn_(true),
      update_jacobians_(true),
      id_(id),
      fref_(fref) {
  if (nc > 6) {
    throw_pretty(
        "Invalid argument in ResidualModelContactForce: nc should be less than "
        "6");
  }
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
void ResidualModelContactForceTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // We transform the force to the contact frame
  switch (d->contact_type) {
    case Contact1D:
      data->r = ((d->contact->f - fref_).linear()).row(2);
      break;
    case Contact3D:
      data->r = (d->contact->f - fref_).linear();
      break;
    case Contact6D:
      data->r = (d->contact->f - fref_).toVector();
      break;
    default:
      break;
  }
}

template <typename Scalar>
void ResidualModelContactForceTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  data->r.setZero();
}

template <typename Scalar>
void ResidualModelContactForceTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  if (fwddyn_ || update_jacobians_) {
    updateJacobians(data);
  }
}

template <typename Scalar>
void ResidualModelContactForceTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&) {
  data->Rx.setZero();
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar> >
ResidualModelContactForceTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  std::shared_ptr<ResidualDataAbstract> d =
      std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  if (!fwddyn_) {
    updateJacobians(d);
  }
  return d;
}

template <typename Scalar>
void ResidualModelContactForceTpl<Scalar>::updateJacobians(
    const std::shared_ptr<ResidualDataAbstract>& data) {
  Data* d = static_cast<Data*>(data.get());

  const MatrixXs& df_dx = d->contact->df_dx;
  const MatrixXs& df_du = d->contact->df_du;
  switch (d->contact_type) {
    case Contact1D:
      data->Rx = df_dx.template topRows<1>();
      data->Ru = df_du.template topRows<1>();
      break;
    case Contact3D:
      data->Rx = df_dx.template topRows<3>();
      data->Ru = df_du.template topRows<3>();
      break;
    case Contact6D:
      data->Rx = df_dx;
      data->Ru = df_du;
      break;
    default:
      break;
  }
  update_jacobians_ = false;
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelContactForceTpl<NewScalar>
ResidualModelContactForceTpl<Scalar>::cast() const {
  typedef ResidualModelContactForceTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      id_, fref_.template cast<NewScalar>(), nr_, nu_, fwddyn_);
  return ret;
}

template <typename Scalar>
void ResidualModelContactForceTpl<Scalar>::print(std::ostream& os) const {
  std::shared_ptr<StateMultibody> s =
      std::static_pointer_cast<StateMultibody>(state_);
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  os << "ResidualModelContactForce {frame="
     << s->get_pinocchio()->frames[id_].name
     << ", fref=" << fref_.toVector().head(nr_).transpose().format(fmt) << "}";
}

template <typename Scalar>
bool ResidualModelContactForceTpl<Scalar>::is_fwddyn() const {
  return fwddyn_;
}

template <typename Scalar>
pinocchio::FrameIndex ResidualModelContactForceTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const pinocchio::ForceTpl<Scalar>&
ResidualModelContactForceTpl<Scalar>::get_reference() const {
  return fref_;
}

template <typename Scalar>
void ResidualModelContactForceTpl<Scalar>::set_id(
    const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualModelContactForceTpl<Scalar>::set_reference(
    const Force& reference) {
  fref_ = reference;
  update_jacobians_ = true;
}

}  // namespace crocoddyl
