///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ResidualModelStateTpl<Scalar>::ResidualModelStateTpl(
    std::shared_ptr<typename Base::StateAbstract> state, const VectorXs& xref,
    const std::size_t nu)
    : Base(state, state->get_ndx(), nu, true, true, false), xref_(xref) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "xref has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  // Define the pinocchio model for the multibody state case
  const std::shared_ptr<StateMultibody>& s =
      std::dynamic_pointer_cast<StateMultibody>(state);
  if (s) {
    pin_model_ = s->get_pinocchio();
  }
}

template <typename Scalar>
ResidualModelStateTpl<Scalar>::ResidualModelStateTpl(
    std::shared_ptr<typename Base::StateAbstract> state, const VectorXs& xref)
    : Base(state, state->get_ndx(), true, true, false), xref_(xref) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "xref has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  // Define the pinocchio model for the multibody state case
  const std::shared_ptr<StateMultibody>& s =
      std::dynamic_pointer_cast<StateMultibody>(state);
  if (s) {
    pin_model_ = s->get_pinocchio();
  }
}

template <typename Scalar>
ResidualModelStateTpl<Scalar>::ResidualModelStateTpl(
    std::shared_ptr<typename Base::StateAbstract> state, const std::size_t nu)
    : Base(state, state->get_ndx(), nu, true, true, false),
      xref_(state->zero()) {
  // Define the pinocchio model for the multibody state case
  const std::shared_ptr<StateMultibody>& s =
      std::dynamic_pointer_cast<StateMultibody>(state);
  if (s) {
    pin_model_ = s->get_pinocchio();
  }
}

template <typename Scalar>
ResidualModelStateTpl<Scalar>::ResidualModelStateTpl(
    std::shared_ptr<typename Base::StateAbstract> state)
    : Base(state, state->get_ndx(), true, true, false), xref_(state->zero()) {
  // Define the pinocchio model for the multibody state case
  const std::shared_ptr<StateMultibody>& s =
      std::dynamic_pointer_cast<StateMultibody>(state);
  if (s) {
    pin_model_ = s->get_pinocchio();
  }
}

template <typename Scalar>
void ResidualModelStateTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  state_->diff(xref_, x, data->r);
}

template <typename Scalar>
void ResidualModelStateTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  state_->Jdiff(xref_, x, data->Rx, data->Rx, second);
}

template <typename Scalar>
void ResidualModelStateTpl<Scalar>::calcCostDiff(
    const std::shared_ptr<CostDataAbstract>& cdata,
    const std::shared_ptr<ResidualDataAbstract>& rdata,
    const std::shared_ptr<ActivationDataAbstract>& adata, const bool) {
  const std::size_t nv = state_->get_nv();
  if (pin_model_) {
    typedef Eigen::Block<MatrixXs> MatrixBlock;
    for (pinocchio::JointIndex i = 1;
         i < (pinocchio::JointIndex)pin_model_->njoints; ++i) {
      const MatrixBlock& RxBlock =
          rdata->Rx.block(pin_model_->idx_vs[i], pin_model_->idx_vs[i],
                          pin_model_->nvs[i], pin_model_->nvs[i]);
      cdata->Lx.segment(pin_model_->idx_vs[i], pin_model_->nvs[i]).noalias() =
          RxBlock.transpose() *
          adata->Ar.segment(pin_model_->idx_vs[i], pin_model_->nvs[i]);
      cdata->Lxx
          .block(pin_model_->idx_vs[i], pin_model_->idx_vs[i],
                 pin_model_->nvs[i], pin_model_->nvs[i])
          .noalias() = RxBlock.transpose() *
                       adata->Arr.diagonal()
                           .segment(pin_model_->idx_vs[i], pin_model_->nvs[i])
                           .asDiagonal() *
                       RxBlock;
    }
    cdata->Lx.tail(nv) = adata->Ar.tail(nv);
    cdata->Lxx.diagonal().tail(nv) = adata->Arr.diagonal().tail(nv);
  } else {
    cdata->Lx = adata->Ar;
    cdata->Lxx.diagonal() = adata->Arr.diagonal();
  }
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelStateTpl<NewScalar> ResidualModelStateTpl<Scalar>::cast() const {
  typedef ResidualModelStateTpl<NewScalar> ReturnType;
  typedef StateAbstractTpl<NewScalar> StateType;
  ReturnType ret(
      std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
      xref_.template cast<NewScalar>(), nu_);
  return ret;
}

template <typename Scalar>
void ResidualModelStateTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelState";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ResidualModelStateTpl<Scalar>::get_reference() const {
  return xref_;
}

template <typename Scalar>
void ResidualModelStateTpl<Scalar>::set_reference(const VectorXs& reference) {
  if (static_cast<std::size_t>(reference.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: "
        << "the state reference has wrong dimension (" << reference.size()
        << " provided - it should be " + std::to_string(state_->get_nx()) + ")")
  }
  xref_ = reference;
}

}  // namespace crocoddyl
