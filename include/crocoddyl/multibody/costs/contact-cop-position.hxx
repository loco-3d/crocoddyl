///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Duisburg-Essen, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/contact-cop-position.hpp"

namespace crocoddyl {

template<typename _Scalar>
CostModelContactCoPPositionTpl<_Scalar>::CostModelContactCoPPositionTpl(boost::shared_ptr<StateMultibody> state,
                                          boost::shared_ptr<ActivationModelAbstract> activation,
                                          const CoPSupport& cop_support, const std::size_t& nu)
    : Base(state, activation, nu), cop_support_(cop_support) {
      cop_support_.update_A();
    }

template <typename Scalar>
CostModelContactCoPPositionTpl<Scalar>::~CostModelContactCoPPositionTpl() {}

template <typename Scalar>
void CostModelContactCoPPositionTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                           const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Transform the contact force
  d->f = d->contact->jMf.actInv(d->contact->f);

  // Compute the cost residual respecting A * f
  data->r.noalias() = cop_support_.get_A() * d->f.toVector();

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelContactCoPPositionTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                                      const Eigen::Ref<const VectorXs>&, 
                                                      const Eigen::Ref<const VectorXs>&) {
  // Update all data
  Data* d = static_cast<Data*>(data.get());

  // Get the derivatives of the contact wrench
  const MatrixXs& df_dx = d->contact->df_dx;
  const MatrixXs& df_du = d->contact->df_du;
  const Matrix46& A = cop_support_.get_A();

  //Compute the derivatives of the activation function
  activation_->calcDiff(data->activation, data->r);

  //Compute the derivatives of the cost residual
  data->Rx.noalias() = A * df_dx;
  data->Ru.noalias() = A * df_du;
  d->Arr_Ru.noalias() = data->activation->Arr * data->Ru;

  //Compute the first order derivatives of the cost function
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  data->Lu.noalias() = data->Ru.transpose() * data->activation->Ar;

  //Compute the second order derivatives of the cost function
  data->Lxx.noalias() = data->Rx.transpose() * data->activation->Arr * data->Rx;
  data->Lxu.noalias() = data->Rx.transpose() * d->Arr_Ru;
  data->Luu.noalias() = data->Ru.transpose() * d->Arr_Ru;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelContactCoPPositionTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template<typename Scalar>
const FrameCoPSupportTpl<Scalar>& CostModelContactCoPPositionTpl<Scalar>::get_copSupport() const {
  return cop_support_;
}

}  // namespace crocoddyl
