///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/contact-friction-cone.hpp"

namespace crocoddyl {
  
template<typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::CostModelContactFrictionConeTpl(boost::shared_ptr<StateMultibody> state,
                                                           boost::shared_ptr<ActivationModelAbstract> activation,
                                                           const FrictionCone& cone, const FrameIndex& frame,
                                                           const std::size_t& nu)
    : Base(state, activation, nu), friction_cone_(cone), frame_(frame) {
  if (activation_->get_nr() != friction_cone_.get_nf() + 1) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " << friction_cone_.get_nf() + 1);
  }
}

template<typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::CostModelContactFrictionConeTpl(boost::shared_ptr<StateMultibody> state,
                                                           boost::shared_ptr<ActivationModelAbstract> activation,
                                                           const FrictionCone& cone, const FrameIndex& frame)
    : Base(state, activation), friction_cone_(cone), frame_(frame) {
  if (activation_->get_nr() != friction_cone_.get_nf() + 1) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " << friction_cone_.get_nf() + 1);
  }
}

template<typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::CostModelContactFrictionConeTpl(boost::shared_ptr<StateMultibody> state,
                                                           const FrictionCone& cone, const FrameIndex& frame,
                                                           const std::size_t& nu)
    : Base(state, cone.get_nf() + 1, nu), friction_cone_(cone), frame_(frame) {}

template<typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::CostModelContactFrictionConeTpl(boost::shared_ptr<StateMultibody> state,
                                                           const FrictionCone& cone, const FrameIndex& frame)
    : Base(state, cone.get_nf() + 1), friction_cone_(cone), frame_(frame) {}

template<typename Scalar>
CostModelContactFrictionConeTpl<Scalar>::~CostModelContactFrictionConeTpl() {}

template<typename Scalar>
void CostModelContactFrictionConeTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                        const Eigen::Ref<const VectorXs>&,
                                        const Eigen::Ref<const VectorXs>&) {
  CostDataContactFrictionCone* d = static_cast<CostDataContactFrictionCone*>(data.get());

  // Compute the residual of the friction cone. Note that we need to transform the force
  // to the contact frame
  data->r.noalias() = friction_cone_.get_A() * d->contact->jMf.actInv(d->contact->f).linear();

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template<typename Scalar>
void CostModelContactFrictionConeTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                            const Eigen::Ref<const VectorXs>&,
                                            const Eigen::Ref<const VectorXs>&) {
  CostDataContactFrictionConeTpl<Scalar>* d =
    static_cast<CostDataContactFrictionConeTpl<Scalar>* >(data.get());

  const MatrixXs& df_dx = d->contact->df_dx;
  const MatrixXs& df_du = d->contact->df_du;
  const MatrixX3s& A = friction_cone_.get_A();

  activation_->calcDiff(data->activation, data->r);
  if (d->more_than_3_constraints) {
    data->Rx.noalias() = A * df_dx.template topRows<3>();
    data->Ru.noalias() = A * df_du.template topRows<3>();
  } else {
    data->Rx.noalias() = A * df_dx;
    data->Ru.noalias() = A * df_du;
  }
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  data->Lu.noalias() = data->Ru.transpose() * data->activation->Ar;

  d->Arr_Ru.noalias() = data->activation->Arr * data->Ru;

  data->Lxx.noalias() = data->Rx.transpose() * data->activation->Arr * data->Rx;
  data->Lxu.noalias() = data->Rx.transpose() * d->Arr_Ru;
  data->Luu.noalias() = data->Ru.transpose() * d->Arr_Ru;
}

template<typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelContactFrictionConeTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataContactFrictionConeTpl<Scalar> >(this, data);
}

template<typename Scalar>
const FrictionConeTpl<Scalar>& CostModelContactFrictionConeTpl<Scalar>::get_friction_cone() const { return friction_cone_; }

template<typename Scalar>
const pinocchio::FrameIndex& CostModelContactFrictionConeTpl<Scalar>::get_frame() const { return frame_; }

template<typename Scalar>
void CostModelContactFrictionConeTpl<Scalar>::set_friction_cone(const FrictionCone& cone) { friction_cone_ = cone; }

template<typename Scalar>
void CostModelContactFrictionConeTpl<Scalar>::set_frame(const FrameIndex& frame) { frame_ = frame; }

}  // namespace crocoddyl
