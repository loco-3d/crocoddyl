///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelCoMPositionTpl<Scalar>::ResidualModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state,
                                                                 const Vector3s& cref, const std::size_t nu)
    : Base(state, 3, nu), cref_(cref) {
  v_dependent_ = false;
  u_dependent_ = false;
}

template <typename Scalar>
ResidualModelCoMPositionTpl<Scalar>::ResidualModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state,
                                                                 const Vector3s& cref)
    : Base(state, 3), cref_(cref) {
  v_dependent_ = false;
  u_dependent_ = false;
}

template <typename Scalar>
ResidualModelCoMPositionTpl<Scalar>::~ResidualModelCoMPositionTpl() {}

template <typename Scalar>
void ResidualModelCoMPositionTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                               const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  // Compute the residual residual give the reference CoMPosition position
  Data* d = static_cast<Data*>(data.get());
  data->r = d->pinocchio->com[0] - cref_;
}

template <typename Scalar>
void ResidualModelCoMPositionTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                   const Eigen::Ref<const VectorXs>&,
                                                   const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the derivatives of the frame placement
  const std::size_t nv = state_->get_nv();
  data->Rx.leftCols(nv) = d->pinocchio->Jcom;
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelCoMPositionTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s& ResidualModelCoMPositionTpl<Scalar>::get_reference() const {
  return cref_;
}

template <typename Scalar>
void ResidualModelCoMPositionTpl<Scalar>::set_reference(const Vector3s& cref) {
  cref_ = cref;
}

}  // namespace crocoddyl
