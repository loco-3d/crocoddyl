///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <boost/core/demangle.hpp>

namespace crocoddyl {

template <typename Scalar>
ResidualModelAbstractTpl<Scalar>::ResidualModelAbstractTpl(boost::shared_ptr<StateAbstract> state,
                                                           const std::size_t nr, const std::size_t nu,
                                                           const bool q_dependent, const bool v_dependent,
                                                           const bool u_dependent)
    : state_(state),
      nr_(nr),
      nu_(nu),
      unone_(VectorXs::Zero(nu)),
      q_dependent_(q_dependent),
      v_dependent_(v_dependent),
      u_dependent_(u_dependent) {}

template <typename Scalar>
ResidualModelAbstractTpl<Scalar>::ResidualModelAbstractTpl(boost::shared_ptr<StateAbstract> state,
                                                           const std::size_t nr, const bool q_dependent,
                                                           const bool v_dependent, const bool u_dependent)
    : state_(state),
      nr_(nr),
      nu_(state->get_nv()),
      unone_(VectorXs::Zero(state->get_nv())),
      q_dependent_(q_dependent),
      v_dependent_(v_dependent),
      u_dependent_(u_dependent) {}

template <typename Scalar>
ResidualModelAbstractTpl<Scalar>::~ResidualModelAbstractTpl() {}

template <typename Scalar>
void ResidualModelAbstractTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>&,
                                            const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {}

template <typename Scalar>
void ResidualModelAbstractTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                            const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void ResidualModelAbstractTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>&,
                                                const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
}

template <typename Scalar>
void ResidualModelAbstractTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                const Eigen::Ref<const VectorXs>& x) {
  calcDiff(data, x, unone_);
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelAbstractTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<ResidualDataAbstract>(Eigen::aligned_allocator<ResidualDataAbstract>(), this, data);
}

template <typename Scalar>
void ResidualModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

template <typename Scalar>
const boost::shared_ptr<StateAbstractTpl<Scalar> >& ResidualModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
std::size_t ResidualModelAbstractTpl<Scalar>::get_nr() const {
  return nr_;
}

template <typename Scalar>
std::size_t ResidualModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
bool ResidualModelAbstractTpl<Scalar>::get_q_dependent() const {
  return q_dependent_;
}

template <typename Scalar>
bool ResidualModelAbstractTpl<Scalar>::get_v_dependent() const {
  return v_dependent_;
}

template <typename Scalar>
bool ResidualModelAbstractTpl<Scalar>::get_u_dependent() const {
  return u_dependent_;
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const ResidualModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

}  // namespace crocoddyl
