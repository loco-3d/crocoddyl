///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelStateTpl<Scalar>::ResidualModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                     const VectorXs& xref, const std::size_t nu)
    : Base(state, state->get_ndx(), nu), xref_(xref) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
}

template <typename Scalar>
ResidualModelStateTpl<Scalar>::ResidualModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                     const VectorXs& xref)
    : Base(state, state->get_ndx()), xref_(xref) {
  if (static_cast<std::size_t>(xref_.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "xref has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
}

template <typename Scalar>
ResidualModelStateTpl<Scalar>::ResidualModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                     const std::size_t nu)
    : Base(state, state->get_ndx(), nu), xref_(state->zero()) {}

template <typename Scalar>
ResidualModelStateTpl<Scalar>::ResidualModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state)
    : Base(state, state->get_ndx()), xref_(state->zero()) {}

template <typename Scalar>
ResidualModelStateTpl<Scalar>::~ResidualModelStateTpl() {}

template <typename Scalar>
void ResidualModelStateTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                         const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  state_->diff(xref_, x, data->r);
}

template <typename Scalar>
void ResidualModelStateTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                             const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>&) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  state_->Jdiff(xref_, x, data->Rx, data->Rx, second);
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelStateTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::make_shared<ResidualDataStateTpl<Scalar> >(this, data);
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ResidualModelStateTpl<Scalar>::get_reference() const {
  return xref_;
}

template <typename Scalar>
void ResidualModelStateTpl<Scalar>::set_reference(const VectorXs& reference) {
  xref_ = reference;
}

}  // namespace crocoddyl
