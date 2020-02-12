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

CostModelContactFrictionCone::CostModelContactFrictionCone(boost::shared_ptr<StateMultibody> state,
                                                           boost::shared_ptr<ActivationModelAbstract> activation,
                                                           const FrictionCone& cone, const FrameIndex& frame,
                                                           const std::size_t& nu)
    : CostModelAbstract(state, activation, nu), friction_cone_(cone), frame_(frame) {
  if (activation_->get_nr() != friction_cone_.get_nf() + 1) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " << friction_cone_.get_nf() + 1);
  }
}

CostModelContactFrictionCone::CostModelContactFrictionCone(boost::shared_ptr<StateMultibody> state,
                                                           boost::shared_ptr<ActivationModelAbstract> activation,
                                                           const FrictionCone& cone, const FrameIndex& frame)
    : CostModelAbstract(state, activation), friction_cone_(cone), frame_(frame) {
  if (activation_->get_nr() != friction_cone_.get_nf() + 1) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " << friction_cone_.get_nf() + 1);
  }
}

CostModelContactFrictionCone::CostModelContactFrictionCone(boost::shared_ptr<StateMultibody> state,
                                                           const FrictionCone& cone, const FrameIndex& frame,
                                                           const std::size_t& nu)
    : CostModelAbstract(state, cone.get_nf() + 1, nu), friction_cone_(cone), frame_(frame) {}

CostModelContactFrictionCone::CostModelContactFrictionCone(boost::shared_ptr<StateMultibody> state,
                                                           const FrictionCone& cone, const FrameIndex& frame)
    : CostModelAbstract(state, cone.get_nf() + 1), friction_cone_(cone), frame_(frame) {}

CostModelContactFrictionCone::~CostModelContactFrictionCone() {}

void CostModelContactFrictionCone::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                        const Eigen::Ref<const Eigen::VectorXd>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&) {
  CostDataContactFrictionCone* d = static_cast<CostDataContactFrictionCone*>(data.get());

  // Compute the residual of the friction cone. Note that we need to transform the force
  // to the contact frame
  data->r.noalias() = friction_cone_.get_A() * d->contact->jMf.actInv(d->contact->f).linear();

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelContactFrictionCone::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                            const Eigen::Ref<const Eigen::VectorXd>& x,
                                            const Eigen::Ref<const Eigen::VectorXd>& u, const bool&) {

  CostDataContactFrictionCone* d = static_cast<CostDataContactFrictionCone*>(data.get());

  const Eigen::MatrixXd& df_dx = d->contact->df_dx;
  const Eigen::MatrixXd& df_du = d->contact->df_du;
  const FrictionCone::MatrixX3& A = friction_cone_.get_A();

  activation_->calcDiff(data->activation, data->r);
  if (d->more_than_3_constraints) {
    data->Rx.noalias() = A * df_dx.topRows<3>();
    data->Ru.noalias() = A * df_du.topRows<3>();
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

boost::shared_ptr<CostDataAbstract> CostModelContactFrictionCone::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataContactFrictionCone>(this, data);
}

const FrictionCone& CostModelContactFrictionCone::get_friction_cone() const { return friction_cone_; }

const FrameIndex& CostModelContactFrictionCone::get_frame() const { return frame_; }

void CostModelContactFrictionCone::set_friction_cone(const FrictionCone& cone) { friction_cone_ = cone; }

void CostModelContactFrictionCone::set_frame(const FrameIndex& frame) { frame_ = frame; }

}  // namespace crocoddyl
