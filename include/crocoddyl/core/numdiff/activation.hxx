///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          New York University, Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ActivationModelNumDiffTpl<Scalar>::ActivationModelNumDiffTpl(
    std::shared_ptr<Base> model)
    : Base(model->get_nr()),
      model_(model),
      e_jac_(sqrt(Scalar(2.0) * std::numeric_limits<Scalar>::epsilon())) {}

template <typename Scalar>
ActivationModelNumDiffTpl<Scalar>::~ActivationModelNumDiffTpl() {}

template <typename Scalar>
void ActivationModelNumDiffTpl<Scalar>::calc(
    const std::shared_ptr<ActivationDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& r) {
  if (static_cast<std::size_t>(r.size()) != model_->get_nr()) {
    throw_pretty(
        "Invalid argument: " << "r has wrong dimension (it should be " +
                                    std::to_string(model_->get_nr()) + ")");
  }
  std::shared_ptr<Data> data_nd = std::static_pointer_cast<Data>(data);
  model_->calc(data_nd->data_0, r);
  data->a_value = data_nd->data_0->a_value;
}

template <typename Scalar>
void ActivationModelNumDiffTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActivationDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& r) {
  if (static_cast<std::size_t>(r.size()) != model_->get_nr()) {
    throw_pretty(
        "Invalid argument: " << "r has wrong dimension (it should be " +
                                    std::to_string(model_->get_nr()) + ")");
  }
  std::shared_ptr<Data> data_nd = std::static_pointer_cast<Data>(data);

  const Scalar a_value0 = data_nd->data_0->a_value;
  data->a_value = data_nd->data_0->a_value;
  const std::size_t nr = model_->get_nr();

  // Computing the d activation(r) / dr
  const Scalar rh_jac = e_jac_ * std::max(Scalar(1.), r.norm());
  data_nd->rp = r;
  for (unsigned int i_r = 0; i_r < nr; ++i_r) {
    data_nd->rp(i_r) += rh_jac;
    model_->calc(data_nd->data_rp[i_r], data_nd->rp);
    data_nd->rp(i_r) -= rh_jac;
    data->Ar(i_r) = (data_nd->data_rp[i_r]->a_value - a_value0) / rh_jac;
  }

  // Computing the d^2 action(r) / dr^2
  data_nd->Arr_.noalias() = data->Ar * data->Ar.transpose();
  data->Arr.diagonal() = data_nd->Arr_.diagonal();
}

template <typename Scalar>
std::shared_ptr<ActivationDataAbstractTpl<Scalar> >
ActivationModelNumDiffTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
template <typename NewScalar>
ActivationModelNumDiffTpl<NewScalar> ActivationModelNumDiffTpl<Scalar>::cast()
    const {
  typedef ActivationModelNumDiffTpl<NewScalar> ReturnType;
  ReturnType res(model_->template cast<NewScalar>());
  return res;
}

template <typename Scalar>
const std::shared_ptr<ActivationModelAbstractTpl<Scalar> >&
ActivationModelNumDiffTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
const Scalar ActivationModelNumDiffTpl<Scalar>::get_disturbance() const {
  return e_jac_;
}

template <typename Scalar>
void ActivationModelNumDiffTpl<Scalar>::set_disturbance(
    const Scalar disturbance) {
  if (disturbance < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "Disturbance constant is positive");
  }
  e_jac_ = disturbance;
}

}  // namespace crocoddyl
