///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ConstraintModelAbstractTpl<Scalar>::ConstraintModelAbstractTpl(
    std::shared_ptr<StateAbstract> state,
    std::shared_ptr<ResidualModelAbstract> residual, const std::size_t ng,
    const std::size_t nh)
    : ng_internal_(ng),
      nh_internal_(nh),
      state_(state),
      residual_(residual),
      type_((ng > 0 && nh > 0) ? ConstraintType::Both
                               : (ng > 0 ? ConstraintType::Inequality
                                         : ConstraintType::Equality)),
      lb_(VectorXs::Constant(ng, -std::numeric_limits<Scalar>::infinity())),
      ub_(VectorXs::Constant(ng, std::numeric_limits<Scalar>::infinity())),
      nu_(residual->get_nu()),
      ng_(ng),
      nh_(nh),
      T_constraint_(residual->get_q_dependent() || residual->get_v_dependent()
                        ? true
                        : false),
      unone_(VectorXs::Zero(residual->get_nu())) {
  if (nh_ > residual_->get_nr()) {
    throw_pretty("Invalid argument: "
                 << "the number of equality constraints (nh) is wrong as it is "
                    "bigger than the residual dimension.")
  }
  std::size_t max_ng = 2 * (residual_->get_nr() - nh_);
  if (ng_ > max_ng) {
    throw_pretty("Invalid argument: "
                 << "the number of inequality constraints (ng) is wrong as it "
                    "should be in the range [0, " +
                        std::to_string(max_ng) + "]");
  }
}

template <typename Scalar>
ConstraintModelAbstractTpl<Scalar>::ConstraintModelAbstractTpl(
    std::shared_ptr<StateAbstract> state, const std::size_t nu,
    const std::size_t ng, const std::size_t nh, const bool T_const)
    : ng_internal_(ng),
      nh_internal_(nh),
      state_(state),
      residual_(std::make_shared<ResidualModelAbstractTpl<Scalar>>(
          state, ng + nh, nu)),
      type_((ng > 0 && nh > 0) ? ConstraintType::Both
                               : (ng > 0 ? ConstraintType::Inequality
                                         : ConstraintType::Equality)),
      lb_(VectorXs::Constant(ng, -std::numeric_limits<Scalar>::infinity())),
      ub_(VectorXs::Constant(ng, std::numeric_limits<Scalar>::infinity())),
      nu_(nu),
      ng_(ng),
      nh_(nh),
      T_constraint_(T_const),
      unone_(VectorXs::Zero(nu)) {}

template <typename Scalar>
ConstraintModelAbstractTpl<Scalar>::ConstraintModelAbstractTpl(
    std::shared_ptr<StateAbstract> state, const std::size_t ng,
    const std::size_t nh, const bool T_const)
    : ng_internal_(ng),
      nh_internal_(nh),
      state_(state),
      residual_(
          std::make_shared<ResidualModelAbstractTpl<Scalar>>(state, ng + nh)),
      type_((ng > 0 && nh > 0) ? ConstraintType::Both
                               : (ng > 0 ? ConstraintType::Inequality
                                         : ConstraintType::Equality)),
      lb_(VectorXs::Constant(ng, -std::numeric_limits<Scalar>::infinity())),
      ub_(VectorXs::Constant(ng, std::numeric_limits<Scalar>::infinity())),
      nu_(state->get_nv()),
      ng_(ng),
      nh_(nh),
      T_constraint_(T_const),
      unone_(VectorXs::Zero(state->get_nv())) {}

template <typename Scalar>
void ConstraintModelAbstractTpl<Scalar>::calc(
    const std::shared_ptr<ConstraintDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void ConstraintModelAbstractTpl<Scalar>::calcDiff(
    const std::shared_ptr<ConstraintDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calcDiff(data, x, unone_);
}

template <typename Scalar>
std::shared_ptr<ConstraintDataAbstractTpl<Scalar>>
ConstraintModelAbstractTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return std::allocate_shared<ConstraintDataAbstract>(
      Eigen::aligned_allocator<ConstraintDataAbstract>(), this, data);
}

template <typename Scalar>
void ConstraintModelAbstractTpl<Scalar>::update_bounds(const VectorXs& lower,
                                                       const VectorXs& upper) {
  if (static_cast<std::size_t>(upper.size()) != ng_internal_ ||
      static_cast<std::size_t>(lower.size()) != ng_internal_) {
    throw_pretty(
        "Invalid argument: the dimension of the lower/upper bound is not the "
        "same to ng.")
  }
  if (((upper - lower).array() <= Scalar(0.)).any()) {
    throw_pretty(
        "Invalid argument: the upper bound is not higher than the lower bound.")
  }
  if ((lb_.array() == std::numeric_limits<Scalar>::infinity()).any() ||
      (lb_.array() == std::numeric_limits<Scalar>::max()).any()) {
    throw_pretty(
        "Invalid argument: the lower bound cannot contain a positive "
        "infinity/max value");
  }
  if ((ub_.array() == -std::numeric_limits<Scalar>::infinity()).any() ||
      (ub_.array() == -std::numeric_limits<Scalar>::infinity()).any()) {
    throw_pretty(
        "Invalid argument: the lower bound cannot contain a negative "
        "infinity/min value");
  }
  ng_ = ng_internal_;
  nh_ = nh_internal_;
  lb_ = lower;
  ub_ = upper;
  if (nh_ == 0) {
    type_ = ConstraintType::Inequality;
  } else {
    type_ = ConstraintType::Both;
  }
}

template <typename Scalar>
void ConstraintModelAbstractTpl<Scalar>::remove_bounds() {
  ng_ = 0;
  nh_ = nh_internal_ + ng_internal_;
  lb_.setConstant(-std::numeric_limits<Scalar>::infinity());
  ub_.setConstant(std::numeric_limits<Scalar>::infinity());
  type_ = ConstraintType::Equality;
}

template <typename Scalar>
void ConstraintModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

template <typename Scalar>
const std::shared_ptr<StateAbstractTpl<Scalar>>&
ConstraintModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const std::shared_ptr<ResidualModelAbstractTpl<Scalar>>&
ConstraintModelAbstractTpl<Scalar>::get_residual() const {
  return residual_;
}

template <typename Scalar>
ConstraintType ConstraintModelAbstractTpl<Scalar>::get_type() const {
  return type_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ConstraintModelAbstractTpl<Scalar>::get_lb() const {
  return lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ConstraintModelAbstractTpl<Scalar>::get_ub() const {
  return ub_;
}

template <typename Scalar>
std::size_t ConstraintModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
std::size_t ConstraintModelAbstractTpl<Scalar>::get_ng() const {
  return ng_;
}

template <typename Scalar>
std::size_t ConstraintModelAbstractTpl<Scalar>::get_nh() const {
  return nh_;
}

template <typename Scalar>
bool ConstraintModelAbstractTpl<Scalar>::get_T_constraint() const {
  return T_constraint_;
}

template <class Scalar>
std::ostream& operator<<(std::ostream& os,
                         const ConstraintModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

}  // namespace crocoddyl
