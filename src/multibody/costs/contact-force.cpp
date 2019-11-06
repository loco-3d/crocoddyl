///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/contact-force.hpp"

namespace crocoddyl {

CostModelContactForce::CostModelContactForce(boost::shared_ptr<StateMultibody> state,
                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                             boost::shared_ptr<ContactModelAbstract> contact,
                                             const Eigen::VectorXd& fref, const std::size_t& nu)
    : CostModelAbstract(state, activation, nu), contact_(contact), fref_(fref) {
  assert(activation_->get_nr() == contact->get_nc() && "nr is not equals to 6");
  assert(fref.size() == contact->get_nc() && "fref size does not match contact");
}

CostModelContactForce::CostModelContactForce(boost::shared_ptr<StateMultibody> state,
                                             boost::shared_ptr<ActivationModelAbstract> activation,
                                             boost::shared_ptr<ContactModelAbstract> contact,
                                             const Eigen::VectorXd& fref)
    : CostModelAbstract(state, activation), contact_(contact), fref_(fref) {
  assert(activation_->get_nr() == contact->get_nc() && "nr is not equals to 6");
  assert(fref.size() == contact->get_nc() && "fref size does not match contact");
}

CostModelContactForce::CostModelContactForce(boost::shared_ptr<StateMultibody> state,
                                             boost::shared_ptr<ContactModelAbstract> contact,
                                             const Eigen::VectorXd& fref, const std::size_t& nu)
    : CostModelAbstract(state, contact->get_nc(), nu), contact_(contact), fref_(fref) {
  assert(fref.size() == contact->get_nc() && "fref size does not match contact");
}

CostModelContactForce::CostModelContactForce(boost::shared_ptr<StateMultibody> state,
                                             boost::shared_ptr<ContactModelAbstract> contact,
                                             const Eigen::VectorXd& fref)
    : CostModelAbstract(state, contact->get_nc()), contact_(contact), fref_(fref) {
  assert(fref.size() == contact->get_nc() && "fref size does not match contact");
}

CostModelContactForce::~CostModelContactForce() {}

void CostModelContactForce::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                 const Eigen::Ref<const Eigen::VectorXd>& /*x*/,
                                 const Eigen::Ref<const Eigen::VectorXd>& /*u*/) {
  CostDataContactForce* d = static_cast<CostDataContactForce*>(data.get());

  data->r = d->contact_->frame_force - fref_;

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelContactForce::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                     const Eigen::Ref<const Eigen::VectorXd>& x,
                                     const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }

  CostDataContactForce* d = static_cast<CostDataContactForce*>(data.get());

  const Eigen::MatrixXd& df_dx = d->contact_->df_dx;
  const Eigen::MatrixXd& df_du = d->contact_->df_du;
  const std::size_t& nv = state_->get_nv();

  activation_->calcDiff(data->activation, data->r, recalc);
  data->Rx = df_dx;
  data->Ru = df_du;
  data->Lx.noalias() = df_dx.transpose() * data->activation->Ar;
  data->Lu.noalias() = df_du.transpose() * data->activation->Ar;

  d->Arr_Ru.noalias() = data->activation->Arr * df_du;

  data->Lxx.noalias() = df_dx.transpose() * data->activation->Arr * df_dx;
  data->Lxu.noalias() = df_dx.transpose() * d->Arr_Ru;
  data->Luu.noalias() = df_du.transpose() * d->Arr_Ru;
}

boost::shared_ptr<CostDataAbstract> CostModelContactForce::createData(pinocchio::Data* const data) {
  return boost::make_shared<CostDataContactForce>(this, data);
}

}  // namespace crocoddyl
