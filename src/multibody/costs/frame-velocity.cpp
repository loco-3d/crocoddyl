///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include "crocoddyl/multibody/costs/frame-velocity.hpp"

namespace crocoddyl {

CostModelFrameVelocity::CostModelFrameVelocity(boost::shared_ptr<StateMultibody> state,
                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                               const FrameMotion& vref, const std::size_t& nu)
    : CostModelAbstract(state, activation, nu), vref_(vref) {
  if (activation_->get_nr() != 6) {
    throw std::invalid_argument("nr is equals to 6");
  }
}

CostModelFrameVelocity::CostModelFrameVelocity(boost::shared_ptr<StateMultibody> state,
                                               boost::shared_ptr<ActivationModelAbstract> activation,
                                               const FrameMotion& vref)
    : CostModelAbstract(state, activation), vref_(vref) {
  if (activation_->get_nr() != 6) {
    throw std::invalid_argument("nr is equals to 6");
  }
}

CostModelFrameVelocity::CostModelFrameVelocity(boost::shared_ptr<StateMultibody> state, const FrameMotion& vref,
                                               const std::size_t& nu)
    : CostModelAbstract(state, 6, nu), vref_(vref) {}

CostModelFrameVelocity::CostModelFrameVelocity(boost::shared_ptr<StateMultibody> state, const FrameMotion& vref)
    : CostModelAbstract(state, 6), vref_(vref) {}

CostModelFrameVelocity::~CostModelFrameVelocity() {}

void CostModelFrameVelocity::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                  const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&) {
  CostDataFrameVelocity* d = static_cast<CostDataFrameVelocity*>(data.get());

  // Compute the frame velocity w.r.t. the reference frame
  d->vr = pinocchio::getFrameVelocity(state_->get_pinocchio(), *data->pinocchio, vref_.frame) - vref_.oMf;
  data->r = d->vr.toVector();

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelFrameVelocity::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                      const Eigen::Ref<const Eigen::VectorXd>& x,
                                      const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }

  // Get the partial derivatives of the local frame velocity
  CostDataFrameVelocity* d = static_cast<CostDataFrameVelocity*>(data.get());
  pinocchio::getJointVelocityDerivatives(state_->get_pinocchio(), *data->pinocchio, d->joint, pinocchio::LOCAL,
                                         d->v_partial_dq, d->v_partial_dv);

  // Compute the derivatives of the frame velocity
  const std::size_t& nv = state_->get_nv();
  activation_->calcDiff(data->activation, data->r, recalc);
  data->Rx.leftCols(nv).noalias() = d->fXj * d->v_partial_dq;
  data->Rx.rightCols(nv).noalias() = d->fXj * d->v_partial_dv;
  data->Lx.noalias() = data->Rx.transpose() * data->activation->Ar;
  d->Arr_Rx.noalias() = data->activation->Arr * data->Rx;
  data->Lxx.noalias() = data->Rx.transpose() * d->Arr_Rx;
}

boost::shared_ptr<CostDataAbstract> CostModelFrameVelocity::createData(pinocchio::Data* const data) {
  return boost::make_shared<CostDataFrameVelocity>(this, data);
}

const FrameMotion& CostModelFrameVelocity::get_vref() const { return vref_; }

void CostModelFrameVelocity::set_vref(const FrameMotion& vref_in) { vref_ = vref_in; }

}  // namespace crocoddyl
