///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/state.hpp"

namespace crocoddyl {

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<StateMultibody> state,
                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                             const VectorXs& xref, const std::size_t& nu)
    : Base(state, activation, nu), xref_(xref) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<StateMultibody> state,
                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                             const VectorXs& xref)
    : Base(state, activation), xref_(xref) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<StateMultibody> state, const VectorXs& xref,
                                             const std::size_t& nu)
    : Base(state, state->get_ndx(), nu), xref_(xref) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<StateMultibody> state, const VectorXs& xref)
    : Base(state, state->get_ndx()), xref_(xref) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<StateMultibody> state,
                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                             const std::size_t& nu)
    : Base(state, activation, nu), xref_(state->zero()) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& nu)
    : Base(state, state->get_ndx(), nu), xref_(state->zero()) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<StateMultibody> state,
                                             boost::shared_ptr<ActivationModelAbstract> activation)
    : Base(state, activation), xref_(state->zero()) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::CostModelStateTpl(boost::shared_ptr<StateMultibody> state)
    : Base(state, state->get_ndx()), xref_(state->zero()) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (activation_->get_nr() != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_ndx()));
  }
}

template <typename Scalar>
CostModelStateTpl<Scalar>::~CostModelStateTpl() {}

template <typename Scalar>
void CostModelStateTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                     const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  state_->diff(xref_, x, data->r);
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelStateTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  CostDataStateTpl<Scalar>* d = static_cast<CostDataStateTpl<Scalar>*>(data.get());
  state_->Jdiff(xref_, x, data->Rx, data->Rx, second);
  activation_->calcDiff(data->activation, data->r);
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  d->Arr_Rx.noalias() = data->activation->Arr * data->Rx;
  data->Lxx.noalias() = data->Rx.transpose() * d->Arr_Rx;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelStateTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataStateTpl<Scalar> >(this, data);
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& CostModelStateTpl<Scalar>::get_xref() const {
  return xref_;
}

template <typename Scalar>
void CostModelStateTpl<Scalar>::set_xref(const VectorXs& xref_in) {
  if (static_cast<std::size_t>(xref_in.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  xref_ = xref_in;
}

}  // namespace crocoddyl
