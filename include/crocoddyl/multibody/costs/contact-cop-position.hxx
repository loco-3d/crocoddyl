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
                                          const FootGeometry& foot_geom, const Vector3s normal)
    : Base(state, activation), foot_geom_(foot_geom), normal_(normal) {
      foot_geom_.update_A(); //TODO: Call here?
    }

template <typename Scalar>
CostModelContactCoPPositionTpl<Scalar>::~CostModelContactCoPPositionTpl() {}

template <typename Scalar>
void CostModelContactCoPPositionTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                           const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Transform the spatial force to a cartesian force expressed in world coordinates  
  d->fiMo.rotation(d->pinocchio->oMi[d->contact->joint].rotation());
  d->f = d->fiMo.actInv(d->contact->f);
  
  // Compute the CoP (TODO: Remove after evaluation)
  // OC = (tau_0^p x n) / (n * f^p) compare eq.(13) in https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.138.8014&rep=rep1&type=pdf
  d->cop << normal_[1] * d->f.angular()[2] - normal_[2] * d->f.angular()[1], 
            normal_[2] * d->f.angular()[0] - normal_[0] * d->f.angular()[2],
            normal_[0] * d->f.angular()[1] - normal_[1] * d->f.angular()[0]; 
  d->cop *= 1 / (normal_[0] * d->f.linear()[0] + normal_[1] * d->f.linear()[1] + normal_[2] * d->f.linear()[2]);
  // Get foot position (for evaluation)
  // foot_pos_ = d->pinocchio->oMf[d->contact->frame].translation();   

  // Compute the cost residual respecting A * f <= 0
  data->r.noalias() = foot_geom_.get_A() * d->f.toVector(); //TODO: Debug error: static assertion failed: INVALID_MATRIX_PRODUCT

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
  const MatrixXs& df_dx = d->contact->df_dx; // TODO: Evantually transform derivative to Caron frame
  const MatrixXs& df_du = d->contact->df_du;
  const Matrix46s& A = foot_geom_.get_A();

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
const FrameFootGeometryTpl<Scalar>& CostModelContactCoPPositionTpl<Scalar>::get_footGeom() const {
  return foot_geom_;
}

}  // namespace crocoddyl
