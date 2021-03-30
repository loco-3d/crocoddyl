///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ConstraintModelAbstractTpl<Scalar>::ConstraintModelAbstractTpl(boost::shared_ptr<StateAbstract> state,
                                                               boost::shared_ptr<ResidualModelAbstract> residual,
                                                               const std::size_t ng, const std::size_t nh)
    : state_(state),
      residual_(residual),
      nu_(residual->get_nu()),
      ng_(ng),
      nh_(nh),
      unone_(VectorXs::Zero(residual->get_nu())) {
  if (nh_ > residual_->get_nr()) {
    throw_pretty("Invalid argument: "
                 << "the number of equality constraints (nh) is wrong as it is bigger than the residual dimension.")
  }
  std::size_t max_ng = 2 * (residual_->get_nr() - nh_);
  if (0 > ng_ || ng_ > max_ng) {
    throw_pretty("Invalid argument: "
                 << "the number of inequality constraints (ng) is wrong as it should be in the range [0, " +
                        std::to_string(max_ng) + "]");
  }
}

template <typename Scalar>
ConstraintModelAbstractTpl<Scalar>::ConstraintModelAbstractTpl(boost::shared_ptr<StateAbstract> state,
                                                               const std::size_t nu, const std::size_t ng,
                                                               const std::size_t nh)
    : state_(state),
      residual_(boost::make_shared<ResidualModelAbstract>(state, ng + nh, nu)),
      nu_(nu),
      ng_(ng),
      nh_(nh),
      unone_(VectorXs::Zero(nu)) {}

template <typename Scalar>
ConstraintModelAbstractTpl<Scalar>::ConstraintModelAbstractTpl(boost::shared_ptr<StateAbstract> state,
                                                               const std::size_t ng, const std::size_t nh)
    : state_(state),
      residual_(boost::make_shared<ResidualModelAbstract>(state, ng + nh)),
      nu_(state->get_nv()),
      ng_(ng),
      nh_(nh),
      unone_(VectorXs::Zero(state->get_nv())) {}

template <typename Scalar>
ConstraintModelAbstractTpl<Scalar>::~ConstraintModelAbstractTpl() {}

template <typename Scalar>
void ConstraintModelAbstractTpl<Scalar>::calc(const boost::shared_ptr<ConstraintDataAbstract>& data,
                                              const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void ConstraintModelAbstractTpl<Scalar>::calcDiff(const boost::shared_ptr<ConstraintDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>& x) {
  calcDiff(data, x, unone_);
}

template <typename Scalar>
boost::shared_ptr<ConstraintDataAbstractTpl<Scalar> > ConstraintModelAbstractTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<ConstraintDataAbstract>(Eigen::aligned_allocator<ConstraintDataAbstract>(), this,
                                                        data);
}

template <typename Scalar>
const boost::shared_ptr<StateAbstractTpl<Scalar> >& ConstraintModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const boost::shared_ptr<ResidualModelAbstractTpl<Scalar> >& ConstraintModelAbstractTpl<Scalar>::get_residual() const {
  return residual_;
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

}  // namespace crocoddyl
