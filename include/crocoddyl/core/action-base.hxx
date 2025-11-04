///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ActionModelAbstractTpl<Scalar>::ActionModelAbstractTpl(
    std::shared_ptr<StateAbstractTpl<Scalar> > state, const std::size_t nu,
    const std::size_t nr, const std::size_t ng, const std::size_t nh,
    const std::size_t ng_T, const std::size_t nh_T)
    : nu_(nu),
      nr_(nr),
      ng_(ng),
      nh_(nh),
      ng_T_(ng_T),
      nh_T_(nh_T),
      state_(state),
      unone_(MathBase::VectorXs::Zero(nu)),
      g_lb_(VectorXs::Constant(ng > ng_T ? ng : ng_T,
                               -std::numeric_limits<Scalar>::infinity())),
      g_ub_(VectorXs::Constant(ng > ng_T ? ng : ng_T,
                               std::numeric_limits<Scalar>::infinity())),
      u_lb_(MathBase::VectorXs::Constant(
          nu, -std::numeric_limits<Scalar>::infinity())),
      u_ub_(MathBase::VectorXs::Constant(
          nu, std::numeric_limits<Scalar>::infinity())),
      has_control_limits_(false) {}

template <typename Scalar>
ActionModelAbstractTpl<Scalar>::ActionModelAbstractTpl(
    const ActionModelAbstractTpl<Scalar>& other)
    : nu_(other.nu_),
      nr_(other.nr_),
      ng_(other.ng_),
      nh_(other.nh_),
      ng_T_(other.ng_T_),
      nh_T_(other.nh_T_),
      state_(other.state_),
      g_lb_(other.g_lb_),
      g_ub_(other.g_ub_),
      u_lb_(other.u_lb_),
      u_ub_(other.u_ub_),
      has_control_limits_(other.has_control_limits_) {}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calcDiff(data, x, unone_);
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::quasiStatic(
    const std::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
    const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter,
    const Scalar tol) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  // Check the velocity input is zero
  assert_pretty(x.tail(state_->get_nv()).isZero(),
                "The velocity input should be zero for quasi-static to work.");

  const std::size_t ndx = state_->get_ndx();
  VectorXs dx = VectorXs::Zero(ndx);
  if (nu_ != 0) {
    VectorXs du = VectorXs::Zero(nu_);
    for (std::size_t i = 0; i < maxiter; ++i) {
      calc(data, x, u);
      calcDiff(data, x, u);
      state_->diff(x, data->xnext, dx);
      du.noalias() = -pseudoInverse(data->Fu) * dx;
      u += du;
      if (du.norm() <= tol) {
        break;
      }
    }
  }
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs
ActionModelAbstractTpl<Scalar>::quasiStatic_x(
    const std::shared_ptr<ActionDataAbstract>& data, const VectorXs& x,
    const std::size_t maxiter, const Scalar tol) {
  VectorXs u(nu_);
  u.setZero();
  quasiStatic(data, u, x, maxiter, tol);
  return u;
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar> >
ActionModelAbstractTpl<Scalar>::createData() {
  return std::allocate_shared<ActionDataAbstract>(
      Eigen::aligned_allocator<ActionDataAbstract>(), this);
}

template <typename Scalar>
bool ActionModelAbstractTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>&) {
  return false;
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

template <typename Scalar>
std::size_t ActionModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
std::size_t ActionModelAbstractTpl<Scalar>::get_nr() const {
  return nr_;
}

template <typename Scalar>
std::size_t ActionModelAbstractTpl<Scalar>::get_ng() const {
  return ng_;
}

template <typename Scalar>
std::size_t ActionModelAbstractTpl<Scalar>::get_nh() const {
  return nh_;
}

template <typename Scalar>
std::size_t ActionModelAbstractTpl<Scalar>::get_ng_T() const {
  return ng_T_;
}

template <typename Scalar>
std::size_t ActionModelAbstractTpl<Scalar>::get_nh_T() const {
  return nh_T_;
}

template <typename Scalar>
const std::shared_ptr<StateAbstractTpl<Scalar> >&
ActionModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ActionModelAbstractTpl<Scalar>::get_g_lb() const {
  return g_lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ActionModelAbstractTpl<Scalar>::get_g_ub() const {
  return g_ub_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ActionModelAbstractTpl<Scalar>::get_u_lb() const {
  return u_lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ActionModelAbstractTpl<Scalar>::get_u_ub() const {
  return u_ub_;
}

template <typename Scalar>
bool ActionModelAbstractTpl<Scalar>::get_has_control_limits() const {
  return has_control_limits_;
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::set_g_lb(const VectorXs& g_lb) {
  const std::size_t ng = ng_ > ng_T_ ? ng_ : ng_T_;
  if (static_cast<std::size_t>(g_lb.size()) != ng) {
    throw_pretty(
        "Invalid argument: "
        << "inequality lower bound has wrong dimension (it should be " +
               std::to_string(ng) + ")");
  }
  g_lb_ = g_lb;
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::set_g_ub(const VectorXs& g_ub) {
  const std::size_t ng = ng_ > ng_T_ ? ng_ : ng_T_;
  if (static_cast<std::size_t>(g_ub.size()) != ng) {
    throw_pretty(
        "Invalid argument: "
        << "inequality upper bound has wrong dimension (it should be " +
               std::to_string(ng_) + ")");
  }
  g_ub_ = g_ub;
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::set_u_lb(const VectorXs& u_lb) {
  if (static_cast<std::size_t>(u_lb.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "control lower bound has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  u_lb_ = u_lb;
  update_has_control_limits();
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::set_u_ub(const VectorXs& u_ub) {
  if (static_cast<std::size_t>(u_ub.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "control upper bound has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  u_ub_ = u_ub;
  update_has_control_limits();
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::update_has_control_limits() {
  has_control_limits_ =
      isfinite(u_lb_.template cast<ScalarType>().array()).any() &&
      isfinite(u_ub_.template cast<ScalarType>().array()).any();
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os,
                         const ActionModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

}  // namespace crocoddyl
