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
                                          const FootGeometry& foot_geom)
    : Base(state, activation), foot_geom_(foot_geom) {}

template <typename Scalar>
CostModelContactCoPPositionTpl<Scalar>::~CostModelContactCoPPositionTpl() {}

template <typename Scalar>
void CostModelContactCoPPositionTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                           const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Transform the spatial force to a cartesian force expressed in world coordinates  
  data->fiMo = d->pinnochio.SE3(d->pinocchio->oMi[d->contact->joint].rotation().T, d->contact->jMf.translation());
  data->f = data->fiMo.actInv(d->contact->f);
  
  // Compute the CoP (for evaluation)
  // OC = (tau_0^p x n) / (n * f^p) compare eq.(13) in https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.138.8014&rep=rep1&type=pdf
  data->cop(-data->f.angular().at(1) / data->f.linear().at(2), data->f.angular().at(0) / data->f.linear().at(2), 0.0);

  // Get foot position (for evaluation)
  // foot_pos_ = d->pinocchio->oMf[d->contact->frame].translation();   

  // Define the inequality matrix as A * f <= 0
  //Matrix3s c_R_o = Quaternions::FromTwoVectors(nsurf_, Vector3s::UnitZ()).toRotationMatrix(); TODO: Rotation necessary for each row of A?
  data->A << 0, 0, -foot_geom_.dim.at(1) / 2, 1, 0, 0,
             0, 0, -foot_geom_.dim.at(1) / 2, -1, 0, 0,
             0, 0, -foot_geom_.dim.at(0) / 2, 0, 1, 0,
             0, 0, -foot_geom_.dim.at(0) / 2, 0, -1, 0;

  // Compute the cost residual   
  data->r.noalias() = data->A * data->f;           

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelContactCoPPositionTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data, //TODO: Correct assignment of the derivatives?
                                                      const Eigen::Ref<const VectorXs>&, 
                                                      const Eigen::Ref<const VectorXs>&) {
  // Update all data
  Data* d = static_cast<Data*>(data.get());

  // Get the derivatives of the contact wrench
  const MatrixXs& df_dx = data->f->df_dx;
  const MatrixXs& df_du = data->f->df_du;

  //Compute the derivatives of the activation function
  activation_->calcDiff(data->activation, data->r);

  //Compute the derivatives of the cost residual
  data->Rx.noalias() = data->A * df_dx;
  data->Ru.noalias() = data->A * df_du;
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
