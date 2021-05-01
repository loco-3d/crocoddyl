///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"

namespace crocoddyl {

template <typename Scalar>
ImpulseModelAbstractTpl<Scalar>::ImpulseModelAbstractTpl(boost::shared_ptr<StateMultibody> state, const std::size_t nc)
    : state_(state), nc_(nc) {}

template <typename Scalar>
ImpulseModelAbstractTpl<Scalar>::~ImpulseModelAbstractTpl() {}

template <typename Scalar>
void ImpulseModelAbstractTpl<Scalar>::updateForceDiff(const boost::shared_ptr<ImpulseDataAbstract>& data,
                                                      const MatrixXs& df_dx) const {
  if (static_cast<std::size_t>(df_dx.rows()) != nc_ || static_cast<std::size_t>(df_dx.cols()) != state_->get_ndx())
    throw_pretty("df_dq has wrong dimension");

  data->df_dx = df_dx;
}

template <typename Scalar>
void ImpulseModelAbstractTpl<Scalar>::setZeroForce(const boost::shared_ptr<ImpulseDataAbstract>& data) const {
  data->f.setZero();
}

template <typename Scalar>
void ImpulseModelAbstractTpl<Scalar>::setZeroForceDiff(const boost::shared_ptr<ImpulseDataAbstract>& data) const {
  data->df_dx.setZero();
}

template <typename Scalar>
boost::shared_ptr<ImpulseDataAbstractTpl<Scalar> > ImpulseModelAbstractTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<ImpulseDataAbstract>(Eigen::aligned_allocator<ImpulseDataAbstract>(), this, data);
}

template <typename Scalar>
const boost::shared_ptr<StateMultibodyTpl<Scalar> >& ImpulseModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
std::size_t ImpulseModelAbstractTpl<Scalar>::get_nc() const {
  return nc_;
}

template <typename Scalar>
std::size_t ImpulseModelAbstractTpl<Scalar>::get_ni() const {
  return nc_;
}

template <typename Scalar>
std::size_t ImpulseModelAbstractTpl<Scalar>::get_nu() const {
  return 0;
}

}  // namespace crocoddyl
