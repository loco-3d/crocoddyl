///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ParameterPhaseModelTpl<Scalar>::ParameterPhaseModelTpl(
    std::shared_ptr<ParameterManager> params,
    std::shared_ptr<ConstraintModelManager> constraints)
    : params_(params), constraints_(constraints) {
  if (params_ == nullptr) {
    throw_pretty("Invalid argument: params is null");
  }
  if (constraints_ != nullptr &&
      (constraints_->get_state()->get_nx() != params_->get_state()->get_nx() ||
       constraints_->get_state()->get_ndx() !=
           params_->get_state()->get_ndx())) {
    throw_pretty("Invalid argument: constraints have an incompatible state");
  }
  if (constraints_ != nullptr && constraints_->get_np() != params_->get_np()) {
    throw_pretty("Invalid argument: constraints have np="
                 << constraints_->get_np()
                 << " but params have np=" << params_->get_np());
  }
}

template <typename Scalar>
std::shared_ptr<ParameterPhaseDataTpl<Scalar> >
ParameterPhaseModelTpl<Scalar>::createData() const {
  const std::shared_ptr<ParameterDataManager> params_data =
      params_->createData();
  const std::shared_ptr<ConstraintDataManager> constraints_data =
      constraints_ != nullptr ? constraints_->createData(params_data.get())
                              : std::shared_ptr<ConstraintDataManager>();
  return std::allocate_shared<ParameterPhaseData>(
      Eigen::aligned_allocator<ParameterPhaseData>(), params_data,
      constraints_data);
}

template <typename Scalar>
void ParameterPhaseModelTpl<Scalar>::update(
    const std::shared_ptr<ParameterPhaseData>& data,
    const Eigen::Ref<const VectorXs>& p) const {
  if (data == nullptr) {
    throw_pretty("Invalid argument: data is null");
  }
  params_->update(data->params, p);
}

template <typename Scalar>
void ParameterPhaseModelTpl<Scalar>::calc(
    const std::shared_ptr<ParameterPhaseData>& data,
    const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) const {
  if (constraints_ == nullptr) {
    return;
  }
  if (data == nullptr || data->constraints == nullptr) {
    throw_pretty("Invalid argument: constraint data is null");
  }
  data->constraints->resize(constraints_.get(), true);
  constraints_->calc(data->constraints, x, u);
}

template <typename Scalar>
void ParameterPhaseModelTpl<Scalar>::calcDiff(
    const std::shared_ptr<ParameterPhaseData>& data,
    const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) const {
  if (constraints_ == nullptr) {
    return;
  }
  if (data == nullptr || data->constraints == nullptr) {
    throw_pretty("Invalid argument: constraint data is null");
  }
  constraints_->calcDiff(data->constraints, x, u);
}

template <typename Scalar>
template <typename NewScalar>
ParameterPhaseModelTpl<NewScalar> ParameterPhaseModelTpl<Scalar>::cast() const {
  typedef ParameterManagerTpl<NewScalar> NewParameterManager;
  typedef ConstraintModelManagerTpl<NewScalar> NewConstraintModelManager;
  const std::shared_ptr<NewParameterManager> params =
      std::make_shared<NewParameterManager>(
          params_->template cast<NewScalar>());
  const std::shared_ptr<NewConstraintModelManager> constraints =
      constraints_ != nullptr ? std::make_shared<NewConstraintModelManager>(
                                    constraints_->template cast<NewScalar>())
                              : std::shared_ptr<NewConstraintModelManager>();
  return ParameterPhaseModelTpl<NewScalar>(params, constraints);
}

template <typename Scalar>
const std::shared_ptr<
    typename ParameterPhaseModelTpl<Scalar>::ParameterManager>&
ParameterPhaseModelTpl<Scalar>::get_params() const {
  return params_;
}

template <typename Scalar>
const std::shared_ptr<
    typename ParameterPhaseModelTpl<Scalar>::ConstraintModelManager>&
ParameterPhaseModelTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
const std::shared_ptr<typename ParameterPhaseModelTpl<Scalar>::StateAbstract>&
ParameterPhaseModelTpl<Scalar>::get_state() const {
  return params_->get_state();
}

template <typename Scalar>
std::size_t ParameterPhaseModelTpl<Scalar>::get_np() const {
  return params_->get_np();
}

template <typename Scalar>
bool ParameterPhaseModelTpl<Scalar>::has_constraints() const {
  return constraints_ != nullptr &&
         (constraints_->get_ng() != 0 || constraints_->get_nh() != 0);
}

}  // namespace crocoddyl
