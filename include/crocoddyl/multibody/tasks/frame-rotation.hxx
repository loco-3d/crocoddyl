///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2026, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
TaskModelFrameRotationTpl<Scalar>::TaskModelFrameRotationTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Matrix3s& Rref, const std::size_t nu)
    : TaskModelFrameRotationTpl(state, id, Rref, pinocchio::LOCAL, nu) {}

template <typename Scalar>
TaskModelFrameRotationTpl<Scalar>::TaskModelFrameRotationTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Matrix3s& Rref)
    : TaskModelFrameRotationTpl(state, id, Rref, pinocchio::LOCAL,
                                state->get_nv()) {}

template <typename Scalar>
TaskModelFrameRotationTpl<Scalar>::TaskModelFrameRotationTpl(
    std::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
    const Matrix3s& Rref, const pinocchio::ReferenceFrame type,
    const std::size_t nu)
    : Base(state, 3, nu, true, true, true),
      id_(id),
      Rref_(Rref),
      oRf_inv_(Rref.transpose()),
      type_(type),
      pin_model_(state->get_pinocchio()) {
  if (static_cast<pinocchio::FrameIndex>(state->get_pinocchio()->nframes) <=
      id) {
    throw_pretty(
        "Invalid argument: "
        << "the frame index is wrong (it does not exist in the robot)");
  }
}

template <typename Scalar>
void TaskModelFrameRotationTpl<Scalar>::calc(
    const std::shared_ptr<TaskDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
  const Matrix3s& oRf = d->pinocchio->oMf[id_].rotation();

  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      d->rRf.noalias() = oRf_inv_ * oRf;
      pinocchio::Jlog3(d->rRf, d->rJf);
      data->y = pinocchio::log3(d->rRf);
      d->vf = pinocchio::getFrameVelocity(*pin_model_.get(), *d->pinocchio, id_,
                                          pinocchio::LOCAL);
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      d->rRf.noalias() = oRf * oRf_inv_;
      const Matrix3s rRf_inv = d->rRf.transpose().eval();
      pinocchio::Jlog3(rRf_inv, d->rJf);
      data->y = pinocchio::log3(d->rRf);
      d->vf = pinocchio::getFrameVelocity(*pin_model_.get(), *d->pinocchio, id_,
                                          type_);
      break;
  }

  data->v.noalias() = d->rJf * d->vf.angular();
  if (data->compute_acceleration) {
    d->af = pinocchio::getFrameAcceleration(*pin_model_.get(), *d->pinocchio,
                                            id_, type_);
    data->a.noalias() = d->rJf * d->af.angular();
  } else {
    d->af = Motion::Zero();
    data->a.setZero();
  }
}

template <typename Scalar>
void TaskModelFrameRotationTpl<Scalar>::calcDiff(
    const std::shared_ptr<TaskDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();

  data->Yx.setZero();
  data->Yu.setZero();
  data->Vx.setZero();
  data->Vu.setZero();
  data->Ax.setZero();
  data->Au.setZero();

  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_, type_,
                              d->fJf);
  data->Yx.leftCols(nv).noalias() = d->rJf * d->fJf.template bottomRows<3>();

  const pinocchio::ReferenceFrame deriv_type =
      (type_ == pinocchio::ReferenceFrame::LOCAL) ? pinocchio::LOCAL
                                                  : pinocchio::WORLD;
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      pinocchio::Hlog3(d->rRf, d->vf.angular(), d->Hlogf);
      d->dJ_v.noalias() = d->Hlogf.transpose() * d->rRf;
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      const Matrix3s rRf_inv = d->rRf.transpose().eval();
      pinocchio::Hlog3(rRf_inv, -d->vf.angular(), d->Hlogf);
      d->dJ_v.noalias() = d->Hlogf.transpose() * rRf_inv;
      break;
  }

  if (!data->compute_acceleration) {
    pinocchio::getFrameVelocityDerivatives(*pin_model_.get(), *d->pinocchio,
                                           id_, deriv_type, d->fVdq, d->fVdv);
    auto Vx_q = data->Vx.leftCols(nv);
    Vx_q.noalias() = d->dJ_v * d->fJf.template bottomRows<3>();
    Vx_q.noalias() += d->rJf * d->fVdq.template bottomRows<3>();
    data->Vx.rightCols(nv).noalias() =
        d->rJf * d->fVdv.template bottomRows<3>();
    d->dJ_a.setZero();
    d->fAdq.setZero();
    d->fAdv.setZero();
    return;
  }

  pinocchio::getFrameAccelerationDerivatives(*pin_model_.get(), *d->pinocchio,
                                             id_, deriv_type, d->fVdq, d->fAdq,
                                             d->fAdv, d->fVdv);
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      pinocchio::Hlog3(d->rRf, d->af.angular(), d->Hlogf);
      d->dJ_a.noalias() = d->Hlogf.transpose() * d->rRf;
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      const Matrix3s rRf_inv = d->rRf.transpose().eval();
      pinocchio::Hlog3(rRf_inv, -d->af.angular(), d->Hlogf);
      d->dJ_a.noalias() = d->Hlogf.transpose() * rRf_inv;
      break;
  }

  auto Vx_q = data->Vx.leftCols(nv);
  Vx_q.noalias() = d->dJ_v * d->fJf.template bottomRows<3>();
  Vx_q.noalias() += d->rJf * d->fVdq.template bottomRows<3>();
  data->Vx.rightCols(nv).noalias() = d->rJf * d->fVdv.template bottomRows<3>();
  auto Ax_q = data->Ax.leftCols(nv);
  Ax_q.noalias() = d->dJ_a * d->fJf.template bottomRows<3>();
  Ax_q.noalias() += d->rJf * d->fAdq.template bottomRows<3>();
  data->Ax.rightCols(nv).noalias() = d->rJf * d->fAdv.template bottomRows<3>();
  d->a_partial_da.noalias() = d->rJf * d->fVdv.template bottomRows<3>();
  if (d->joint != nullptr) {
    data->Ax.noalias() += d->a_partial_da * d->joint->da_dx;
    data->Au.noalias() = d->a_partial_da * d->joint->da_du;
  }
}

template <typename Scalar>
std::shared_ptr<TaskDataAbstractTpl<Scalar>>
TaskModelFrameRotationTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
template <typename NewScalar>
TaskModelFrameRotationTpl<NewScalar> TaskModelFrameRotationTpl<Scalar>::cast()
    const {
  typedef TaskModelFrameRotationTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      id_, Rref_.template cast<NewScalar>(), type_, nu_);
  return ret;
}

template <typename Scalar>
pinocchio::FrameIndex TaskModelFrameRotationTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Matrix3s&
TaskModelFrameRotationTpl<Scalar>::get_reference() const {
  return Rref_;
}

template <typename Scalar>
pinocchio::ReferenceFrame TaskModelFrameRotationTpl<Scalar>::get_type() const {
  return type_;
}

template <typename Scalar>
void TaskModelFrameRotationTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void TaskModelFrameRotationTpl<Scalar>::set_reference(
    const Matrix3s& reference) {
  Rref_ = reference;
  oRf_inv_ = Rref_.transpose();
}

template <typename Scalar>
void TaskModelFrameRotationTpl<Scalar>::set_type(
    const pinocchio::ReferenceFrame type) {
  switch (type) {
    case pinocchio::ReferenceFrame::LOCAL:
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      type_ = type;
      return;
  }
}

template <typename Scalar>
void TaskModelFrameRotationTpl<Scalar>::print(std::ostream& os) const {
  typedef typename ScalarSelector<Scalar>::type PrintableScalar;
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[",
                            "]");
  typename pinocchio::SE3Tpl<Scalar>::Quaternion qref;
  pinocchio::quaternion::assignQuaternion(qref, Rref_);
  const char* type = nullptr;
  switch (type_) {
    case pinocchio::ReferenceFrame::LOCAL:
      type = "LOCAL";
      break;
    case pinocchio::ReferenceFrame::WORLD:
      type = "WORLD";
      break;
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      type = "LOCAL_WORLD_ALIGNED";
      break;
    default:
      type = "UNKNOWN";
      break;
  }
  os << "TaskModelFrameRotation {frame=" << pin_model_->frames[id_].name
     << ", type=" << type << ", qref="
     << qref.coeffs().transpose().template cast<PrintableScalar>().format(fmt)
     << "}";
}

}  // namespace crocoddyl
