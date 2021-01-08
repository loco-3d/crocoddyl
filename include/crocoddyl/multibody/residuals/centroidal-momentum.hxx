///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/residuals/centroidal-momentum.hpp"

#include <pinocchio/algorithm/centroidal-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
ResidualModelCentroidalMomentumTpl<Scalar>::ResidualModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state,
                                                                               const Vector6s& href,
                                                                               const std::size_t nu)
    : Base(state, 6, nu), href_(href), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ResidualModelCentroidalMomentumTpl<Scalar>::ResidualModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state,
                                                                               const Vector6s& href)
    : Base(state, 6), href_(href), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ResidualModelCentroidalMomentumTpl<Scalar>::~ResidualModelCentroidalMomentumTpl() {}

template <typename Scalar>
void ResidualModelCentroidalMomentumTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                      const Eigen::Ref<const VectorXs>&,
                                                      const Eigen::Ref<const VectorXs>&) {
  // Compute the residual residual give the reference centroidal momentum
  Data* d = static_cast<Data*>(data.get());
  data->r = d->pinocchio->hg.toVector() - href_;
}

template <typename Scalar>
void ResidualModelCentroidalMomentumTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                          const Eigen::Ref<const VectorXs>&,
                                                          const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  const std::size_t& nv = state_->get_nv();
  Eigen::Ref<Matrix6xs> Rq(data->Rx.leftCols(nv));
  Eigen::Ref<Matrix6xs> Rv(data->Rx.rightCols(nv));
  pinocchio::getCentroidalDynamicsDerivatives(*pin_model_.get(), *d->pinocchio, Rq, d->dhd_dq, d->dhd_dv, Rv);

  // The derivative computation in pinocchio does not take the frame of reference into
  // account. So we need to update the com frame as well.
  for (int i = 0; i < d->pinocchio->Jcom.cols(); ++i) {
    data->Rx.template block<3, 1>(3, i) -= d->pinocchio->Jcom.col(i).cross(d->pinocchio->hg.linear());
  }
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualModelCentroidalMomentumTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector6s& ResidualModelCentroidalMomentumTpl<Scalar>::get_reference() const {
  return href_;
}

template <typename Scalar>
void ResidualModelCentroidalMomentumTpl<Scalar>::set_reference(const Vector6s& href) {
  href_ = href;
}

}  // namespace crocoddyl
