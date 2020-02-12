///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/contact-force.hpp"

namespace crocoddyl {

CostModelContactForce::CostModelContactForce(boost::shared_ptr<StateMultibody> state,
                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                             const FrameForce& fref, const std::size_t& nu)
    : CostModelAbstract(state, activation, nu), fref_(fref) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

CostModelContactForce::CostModelContactForce(boost::shared_ptr<StateMultibody> state,
                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                             const FrameForce& fref)
    : CostModelAbstract(state, activation), fref_(fref) {
  if (activation_->get_nr() != 6) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 6");
  }
}

CostModelContactForce::CostModelContactForce(boost::shared_ptr<StateMultibody> state, const FrameForce& fref,
                                             const std::size_t& nu)
    : CostModelAbstract(state, 6, nu), fref_(fref) {}

CostModelContactForce::CostModelContactForce(boost::shared_ptr<StateMultibody> state, const FrameForce& fref)
    : CostModelAbstract(state, 6), fref_(fref) {}

CostModelContactForce::~CostModelContactForce() {}

void CostModelContactForce::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                 const Eigen::Ref<const Eigen::VectorXd>& /*x*/,
                                 const Eigen::Ref<const Eigen::VectorXd>& /*u*/) {
  CostDataContactForce* d = static_cast<CostDataContactForce*>(data.get());

  // We transform the force to the contact frame
  data->r = (d->contact->jMf.actInv(d->contact->f) - fref_.oFf).toVector();

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelContactForce::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                     const Eigen::Ref<const Eigen::VectorXd>& x,
                                     const Eigen::Ref<const Eigen::VectorXd>& u, const bool&) {

  CostDataContactForce* d = static_cast<CostDataContactForce*>(data.get());

  const Eigen::MatrixXd& df_dx = d->contact->df_dx;
  const Eigen::MatrixXd& df_du = d->contact->df_du;

  activation_->calcDiff(data->activation, data->r);
  data->Rx = df_dx;
  data->Ru = df_du;
  data->Lx.noalias() = df_dx.transpose() * data->activation->Ar;
  data->Lu.noalias() = df_du.transpose() * data->activation->Ar;

  d->Arr_Ru.noalias() = data->activation->Arr * df_du;

  data->Lxx.noalias() = df_dx.transpose() * data->activation->Arr * df_dx;
  data->Lxu.noalias() = df_dx.transpose() * d->Arr_Ru;
  data->Luu.noalias() = df_du.transpose() * d->Arr_Ru;
}

boost::shared_ptr<CostDataAbstract> CostModelContactForce::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataContactForce>(this, data);
}

const FrameForce& CostModelContactForce::get_fref() const { return fref_; }

void CostModelContactForce::set_fref(const FrameForce& fref_in) { fref_ = fref_in; }

}  // namespace crocoddyl
