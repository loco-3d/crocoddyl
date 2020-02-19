///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <crocoddyl/core/utils/exception.hpp>

namespace crocoddyl {

template <typename Scalar>
ActionModelAbstractTpl<Scalar>::ActionModelAbstractTpl(boost::shared_ptr<StateAbstractTpl<Scalar> > state,
                                                       const std::size_t& nu,
                                                       const std::size_t& nr)
    : nu_(nu),
      nr_(nr),
      state_(state),
      unone_(MathBase::VectorXs::Zero(nu)),
      u_lb_(MathBase::VectorXs::Constant(nu, -std::numeric_limits<Scalar>::infinity())),
      u_ub_(MathBase::VectorXs::Constant(nu, std::numeric_limits<Scalar>::infinity())),
      has_control_limits_(false) {}
template <typename Scalar>
ActionModelAbstractTpl<Scalar>::ActionModelAbstractTpl()
    : nu_(0),
      nr_(0),
      state_(NULL),
      unone_(MathBase::VectorXs::Zero(0)),
      u_lb_(MathBase::VectorXs::Constant(0, -std::numeric_limits<Scalar>::infinity())),
      u_ub_(MathBase::VectorXs::Constant(0, std::numeric_limits<Scalar>::infinity())),
      has_control_limits_(false) {}

template <typename Scalar>
ActionModelAbstractTpl<Scalar>::~ActionModelAbstractTpl() {}
template <typename Scalar>
  void ActionModelAbstractTpl<Scalar>::calc(const boost::shared_ptr<ActionDataAbstractTpl<Scalar> >& data,
                               const Eigen::Ref<const typename MathBase::VectorXs>& x) {
  calc(data, x, unone_);
}
template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::calcDiff(const boost::shared_ptr<ActionDataAbstractTpl<Scalar> >& data,
                                   const Eigen::Ref<const typename MathBase::VectorXs>& x) {
  calcDiff(data, x, unone_);
}
template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::quasiStatic(const boost::shared_ptr<ActionDataAbstractTpl<Scalar> >& data, Eigen::Ref<typename MathBase::VectorXs> u,
                                      const Eigen::Ref<const typename MathBase::VectorXs>& x, const std::size_t& maxiter,
                                      const Scalar& tol) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  const std::size_t& ndx = state_->get_ndx();
  typename MathBase::VectorXs dx = MathBase::VectorXs::Zero(ndx);
  if (nu_ == 0) {
    // TODO(cmastalli): create a method for autonomous systems
  } else {
    typename MathBase::VectorXs du = MathBase::VectorXs::Zero(nu_);
    for (std::size_t i = 0; i < maxiter; ++i) {
      calc(data, x, u);
      calcDiff(data, x, u);
      state_->diff(x, data->xnext, dx);
      du = -pseudoInverse(data->Fu) * data->Fx * dx;
      u += du;
      if (du.norm() <= tol) {
        break;
      }
    }
  }
}
template <typename Scalar>
boost::shared_ptr<ActionDataAbstractTpl<Scalar> > ActionModelAbstractTpl<Scalar>::createData() {
  return boost::make_shared<ActionDataAbstractTpl<Scalar> >(this);
}
template <typename Scalar>
const std::size_t& ActionModelAbstractTpl<Scalar>::get_nu() const { return nu_; }
template <typename Scalar>
const std::size_t& ActionModelAbstractTpl<Scalar>::get_nr() const { return nr_; }
template <typename Scalar>
const boost::shared_ptr<StateAbstractTpl<Scalar> >& ActionModelAbstractTpl<Scalar>::get_state() const { return state_; }
template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ActionModelAbstractTpl<Scalar>::get_u_lb() const { return u_lb_; }
template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ActionModelAbstractTpl<Scalar>::get_u_ub() const { return u_ub_; }
template <typename Scalar>
bool const& ActionModelAbstractTpl<Scalar>::get_has_control_limits() const { return has_control_limits_; }
template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::set_u_lb(const typename MathBase::VectorXs& u_lb) {
  if (static_cast<std::size_t>(u_lb.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "lower bound has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  u_lb_ = u_lb;
  update_has_control_limits();
}
template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::set_u_ub(const typename MathBase::VectorXs& u_ub) {
  if (static_cast<std::size_t>(u_ub.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "upper bound has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  u_ub_ = u_ub;
  update_has_control_limits();
}
template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::update_has_control_limits() {
  has_control_limits_ = isfinite(u_lb_.array()).any() && isfinite(u_ub_.array()).any();
}

}  // namespace crocoddyl
