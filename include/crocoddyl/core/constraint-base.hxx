///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ConstraintModelAbstractTpl<Scalar>::ConstraintModelAbstractTpl(boost::shared_ptr<StateAbstract> state,
                                                               const std::size_t& nu, const std::size_t& ng,
                                                               const std::size_t& nh)
    : state_(state), nu_(nu), ng_(ng), nh_(nh), unone_(VectorXs::Zero(nu)) {}

template <typename Scalar>
ConstraintModelAbstractTpl<Scalar>::ConstraintModelAbstractTpl(boost::shared_ptr<StateAbstract> state,
                                                               const std::size_t& ng, const std::size_t& nh)
    : state_(state), nu_(state->get_nv()), ng_(ng), nh_(nh), unone_(VectorXs::Zero(state->get_nv())) {}

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
const std::size_t& ConstraintModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
const std::size_t& ConstraintModelAbstractTpl<Scalar>::get_ng() const {
  return ng_;
}

template <typename Scalar>
const std::size_t& ConstraintModelAbstractTpl<Scalar>::get_nh() const {
  return nh_;
}

template <typename Scalar>
template <class ReferenceType>
void ConstraintModelAbstractTpl<Scalar>::set_reference(ReferenceType ref) {
  set_referenceImpl(typeid(ref), &ref);
}

template <typename Scalar>
void ConstraintModelAbstractTpl<Scalar>::set_referenceImpl(const std::type_info&, const void*) {
  throw_pretty("It has not been implemented the set_referenceImpl() function");
}

template <typename Scalar>
template <class ReferenceType>
ReferenceType ConstraintModelAbstractTpl<Scalar>::get_reference() const {
  ReferenceType ref;
  get_referenceImpl(typeid(ref), &ref);
  return ref;
}

template <typename Scalar>
void ConstraintModelAbstractTpl<Scalar>::get_referenceImpl(const std::type_info&, void*) const {
  throw_pretty("It has not been implemented the set_referenceImpl() function");
}

}  // namespace crocoddyl
