///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/state.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                             const VectorXs& xref, const std::size_t& nu)
    : Base(state, activation, boost::make_shared<ResidualModelState>(state, xref, nu)), xref_(xref) {
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
  // Define the pinocchio model for the multibody state case
  const boost::shared_ptr<StateMultibody>& s = boost::dynamic_pointer_cast<StateMultibody>(state);
  if (s) {
    pin_model_ = s->get_pinocchio();
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                             const VectorXs& xref)
    : Base(state, activation, boost::make_shared<ResidualModelState>(state, xref)), xref_(xref) {
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
  // Define the pinocchio model for the multibody state case
  const boost::shared_ptr<StateMultibody>& s = boost::dynamic_pointer_cast<StateMultibody>(state);
  if (s) {
    pin_model_ = s->get_pinocchio();
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                             const VectorXs& xref, const std::size_t& nu)
    : Base(state, boost::make_shared<ResidualModelState>(state, xref, nu)), xref_(xref) {
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
  // Define the pinocchio model for the multibody state case
  const boost::shared_ptr<StateMultibody>& s = boost::dynamic_pointer_cast<StateMultibody>(state);
  if (s) {
    pin_model_ = s->get_pinocchio();
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                             const VectorXs& xref)
    : Base(state, boost::make_shared<ResidualModelState>(state, xref)), xref_(xref) {
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
  // Define the pinocchio model for the multibody state case
  const boost::shared_ptr<StateMultibody>& s = boost::dynamic_pointer_cast<StateMultibody>(state);
  if (s) {
    pin_model_ = s->get_pinocchio();
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                             const std::size_t& nu)
    : Base(state, activation, boost::make_shared<ResidualModelState>(state, nu)), xref_(state->zero()) {
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
  // Define the pinocchio model for the multibody state case
  const boost::shared_ptr<StateMultibody>& s = boost::dynamic_pointer_cast<StateMultibody>(state);
  if (s) {
    pin_model_ = s->get_pinocchio();
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                             const std::size_t& nu)
    : Base(state, boost::make_shared<ResidualModelState>(state, nu)), xref_(state->zero()) {
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
  // Define the pinocchio model for the multibody state case
  const boost::shared_ptr<StateMultibody>& s = boost::dynamic_pointer_cast<StateMultibody>(state);
  if (s) {
    pin_model_ = s->get_pinocchio();
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                             boost::shared_ptr<ActivationModelAbstract> activation)
    : Base(state, activation, boost::make_shared<ResidualModelState>(state)), xref_(state->zero()) {
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
  // Define the pinocchio model for the multibody state case
  const boost::shared_ptr<StateMultibody>& s = boost::dynamic_pointer_cast<StateMultibody>(state);
  if (s) {
    pin_model_ = s->get_pinocchio();
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state)
    : Base(state, boost::make_shared<ResidualModelState>(state)), xref_(state->zero()) {
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
  // Define the pinocchio model for the multibody state case
  const boost::shared_ptr<StateMultibody>& s = boost::dynamic_pointer_cast<StateMultibody>(state);
  if (s) {
    pin_model_ = s->get_pinocchio();
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::~CostModelStateTpl() {}

template <typename Scalar>
void CostModelStateTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                     const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  residual_->calc(data->residual, x, u);
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelStateTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  residual_->calcDiff(data->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);

  if (pin_model_) {
    typedef Eigen::Block<MatrixXs> MatrixBlock;
    for (pinocchio::JointIndex i = 1; i < (pinocchio::JointIndex)pin_model_->njoints; ++i) {
      const MatrixBlock& RxBlock = data->residual->Rx.block(pin_model_->idx_vs[i], pin_model_->idx_vs[i],
                                                            pin_model_->nvs[i], pin_model_->nvs[i]);
      data->Lx.segment(pin_model_->idx_vs[i], pin_model_->nvs[i]).noalias() =
          RxBlock.transpose() * data->activation->Ar.segment(pin_model_->idx_vs[i], pin_model_->nvs[i]);
      data->Lxx.block(pin_model_->idx_vs[i], pin_model_->idx_vs[i], pin_model_->nvs[i], pin_model_->nvs[i]).noalias() =
          RxBlock.transpose() *
          data->activation->Arr.diagonal().segment(pin_model_->idx_vs[i], pin_model_->nvs[i]).asDiagonal() * RxBlock;
    }
    data->Lx.tail(state_->get_nv()) = data->activation->Ar.tail(state_->get_nv());
    data->Lxx.diagonal().tail(state_->get_nv()) = data->activation->Arr.diagonal().tail(state_->get_nv());
  } else {
    data->Lx = data->activation->Ar;
    data->Lxx.diagonal() = data->activation->Arr.diagonal();
  }
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelStateTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataStateTpl<Scalar> >(this, data);
}

template <typename Scalar>
void CostModelStateTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(VectorXs)) {
    VectorXs& tmp = *static_cast<VectorXs*>(pv);
    tmp.resize(state_->get_nx());
    Eigen::Map<VectorXs> ref_map(static_cast<VectorXs*>(pv)->data(), state_->get_nx());
    for (std::size_t i = 0; i < state_->get_nx(); ++i) {
      ref_map[i] = xref_[i];
    }
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be VectorXs)");
  }
}

template <typename Scalar>
void CostModelStateTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(VectorXs)) {
    if (static_cast<std::size_t>(static_cast<const VectorXs*>(pv)->size()) != state_->get_nx()) {
      throw_pretty("Invalid argument: "
                   << "reference has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    xref_ = *static_cast<const VectorXs*>(pv);
    ResidualModelState* residual = static_cast<ResidualModelState*>(residual_.get());
    residual->set_reference(xref_);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be VectorXs)");
  }
}

}  // namespace crocoddyl
