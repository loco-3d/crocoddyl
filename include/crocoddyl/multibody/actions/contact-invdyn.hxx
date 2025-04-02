///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelContactInvDynamicsTpl<Scalar>::
    DifferentialActionModelContactInvDynamicsTpl(
        std::shared_ptr<StateMultibody> state,
        std::shared_ptr<ActuationModelAbstract> actuation,
        std::shared_ptr<ContactModelMultiple> contacts,
        std::shared_ptr<CostModelSum> costs)
    : Base(state, state->get_nv() + contacts->get_nc_total(), costs->get_nr(),
           0, state->get_nv() - actuation->get_nu() + contacts->get_nc_total()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      constraints_(std::make_shared<ConstraintModelManager>(
          state, state->get_nv() + contacts->get_nc_total())),
      pinocchio_(state->get_pinocchio().get()) {
  init(state);
}

template <typename Scalar>
DifferentialActionModelContactInvDynamicsTpl<Scalar>::
    DifferentialActionModelContactInvDynamicsTpl(
        std::shared_ptr<StateMultibody> state,
        std::shared_ptr<ActuationModelAbstract> actuation,
        std::shared_ptr<ContactModelMultiple> contacts,
        std::shared_ptr<CostModelSum> costs,
        std::shared_ptr<ConstraintModelManager> constraints)
    : Base(state, state->get_nv() + contacts->get_nc_total(), costs->get_nr(),
           constraints->get_ng(),
           state->get_nv() - actuation->get_nu() + contacts->get_nc_total() +
               constraints->get_nh(),
           constraints->get_ng_T(), constraints->get_nh_T()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      constraints_(constraints),
      pinocchio_(state->get_pinocchio().get()) {
  init(state);
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::init(
    const std::shared_ptr<StateMultibody>& state) {
  if (contacts_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: "
        << "Contacts doesn't have the same control dimension (it should be " +
               std::to_string(nu_) + ")");
  }
  if (costs_->get_nu() != nu_) {
    throw_pretty(
        "Invalid argument: "
        << "Costs doesn't have the same control dimension (it should be " +
               std::to_string(nu_) + ")");
  }
  const std::size_t nu = actuation_->get_nu();
  const std::size_t nc = contacts_->get_nc_total();
  VectorXs lb =
      VectorXs::Constant(nu_, -std::numeric_limits<Scalar>::infinity());
  VectorXs ub =
      VectorXs::Constant(nu_, std::numeric_limits<Scalar>::infinity());
  Base::set_u_lb(lb);
  Base::set_u_ub(ub);
  contacts_->setComputeAllContacts(true);

  if (state_->get_nv() - actuation_->get_nu() > 0) {
    constraints_->addConstraint(
        "tau", std::make_shared<ConstraintModelResidualTpl<Scalar>>(
                   state_,
                   std::make_shared<
                       typename DifferentialActionModelContactInvDynamicsTpl<
                           Scalar>::ResidualModelActuation>(state, nu, nc),
                   false));
  }
  if (contacts_->get_nc_total() != 0) {
    typename ContactModelMultiple::ContactModelContainer contact_list;
    contact_list = contacts_->get_contacts();
    typename ContactModelMultiple::ContactModelContainer::iterator it_m, end_m;
    for (it_m = contact_list.begin(), end_m = contact_list.end(); it_m != end_m;
         ++it_m) {
      const std::shared_ptr<ContactItem>& contact = it_m->second;
      const std::string name = contact->name;
      const pinocchio::FrameIndex id = contact->contact->get_id();
      const std::size_t nc_i = contact->contact->get_nc();
      const bool active = contact->active;
      constraints_->addConstraint(
          name + "_acc",
          std::make_shared<ConstraintModelResidualTpl<Scalar>>(
              state_,
              std::make_shared<
                  typename DifferentialActionModelContactInvDynamicsTpl<
                      Scalar>::ResidualModelContact>(state, id, nc_i, nc),
              false),
          active);
      constraints_->addConstraint(
          name + "_force",
          std::make_shared<ConstraintModelResidualTpl<Scalar>>(
              state_,
              std::make_shared<ResidualModelContactForceTpl<Scalar>>(
                  state, id, pinocchio::ForceTpl<Scalar>::Zero(), nc_i, nu_,
                  false),
              false),
          !active);
    }
  }
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = contacts_->get_nc_total();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a =
      u.head(nv);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic>
      f_ext = u.tail(nc);

  d->xout = a;
  pinocchio::forwardKinematics(*pinocchio_, d->pinocchio, q, v, a);
  pinocchio::computeJointJacobians(*pinocchio_, d->pinocchio);
  contacts_->calc(d->multibody.contacts, x);
  contacts_->updateForce(d->multibody.contacts, f_ext);
  pinocchio::rnea(*pinocchio_, d->pinocchio, q, v, a,
                  d->multibody.contacts->fext);
  pinocchio::updateGlobalPlacements(*pinocchio_, d->pinocchio);
  pinocchio::centerOfMass(*pinocchio_, d->pinocchio, q, v, a);
  actuation_->commands(d->multibody.actuation, x, d->pinocchio.tau);
  d->multibody.joint->a = a;
  d->multibody.joint->tau = d->multibody.actuation->u;
  actuation_->calc(d->multibody.actuation, x, d->multibody.joint->tau);
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
  for (std::string name : contacts_->get_active_set()) {
    constraints_->changeConstraintStatus(name + "_acc", true);
    constraints_->changeConstraintStatus(name + "_force", false);
  }
  for (std::string name : contacts_->get_inactive_set()) {
    constraints_->changeConstraintStatus(name + "_acc", false);
    constraints_->changeConstraintStatus(name + "_force", true);
  }
  d->constraints->resize(this, d);
  constraints_->calc(d->constraints, x, u);
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }

  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());

  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q, v);
  pinocchio::computeCentroidalMomentum(*pinocchio_, d->pinocchio);
  costs_->calc(d->costs, x);
  d->cost = d->costs->cost;
  d->constraints->resize(this, d, false);
  constraints_->calc(d->constraints, x);
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = contacts_->get_nc_total();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(nv);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a =
      u.head(nv);

  pinocchio::computeRNEADerivatives(*pinocchio_, d->pinocchio, q, v, a,
                                    d->multibody.contacts->fext);
  contacts_->updateRneaDiff(d->multibody.contacts, d->pinocchio);
  d->pinocchio.M.template triangularView<Eigen::StrictlyLower>() =
      d->pinocchio.M.template triangularView<Eigen::StrictlyUpper>()
          .transpose();
  pinocchio::jacobianCenterOfMass(*pinocchio_, d->pinocchio, false);
  actuation_->calcDiff(d->multibody.actuation, x, d->multibody.joint->tau);
  actuation_->torqueTransform(d->multibody.actuation, x,
                              d->multibody.joint->tau);
  d->multibody.joint->dtau_dx.leftCols(nv).noalias() =
      d->multibody.actuation->Mtau * d->pinocchio.dtau_dq;
  d->multibody.joint->dtau_dx.rightCols(nv).noalias() =
      d->multibody.actuation->Mtau * d->pinocchio.dtau_dv;
  d->multibody.joint->dtau_du.leftCols(nv).noalias() =
      d->multibody.actuation->Mtau * d->pinocchio.M;
  d->multibody.joint->dtau_du.rightCols(nc).noalias() =
      -d->multibody.actuation->Mtau *
      d->multibody.contacts->Jc.topRows(nc).transpose();
  contacts_->calcDiff(d->multibody.contacts, x);
  costs_->calcDiff(d->costs, x, u);
  for (std::string name : contacts_->get_active_set()) {
    constraints_->changeConstraintStatus(name + "_acc", true);
    constraints_->changeConstraintStatus(name + "_force", false);
  }
  for (std::string name : contacts_->get_inactive_set()) {
    constraints_->changeConstraintStatus(name + "_acc", false);
    constraints_->changeConstraintStatus(name + "_force", true);
  }
  d->constraints->resize(this, d);
  constraints_->calcDiff(d->constraints, x, u);
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  costs_->calcDiff(d->costs, x);
  if (constraints_ != nullptr) {
    constraints_->calcDiff(d->constraints, x);
  }
}

template <typename Scalar>
std::shared_ptr<DifferentialActionDataAbstractTpl<Scalar>>
DifferentialActionModelContactInvDynamicsTpl<Scalar>::createData() {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::quasiStatic(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x, std::size_t,
    Scalar) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nu = actuation_->get_nu();
  std::size_t nc = contacts_->get_nc_total();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(nq);
  d->tmp_xstatic.head(nq) = q;
  d->tmp_xstatic.tail(nv).setZero();
  u.setZero();

  pinocchio::computeAllTerms(*pinocchio_, d->pinocchio, q,
                             d->tmp_xstatic.tail(nv));
  pinocchio::computeJointJacobians(*pinocchio_, d->pinocchio, q);
  pinocchio::rnea(*pinocchio_, d->pinocchio, q, d->tmp_xstatic.tail(nv),
                  d->tmp_xstatic.tail(nv));
  actuation_->calc(d->multibody.actuation, d->tmp_xstatic,
                   d->tmp_xstatic.tail(nu));
  actuation_->calcDiff(d->multibody.actuation, d->tmp_xstatic,
                       d->tmp_xstatic.tail(nu));
  contacts_->setComputeAllContacts(false);
  contacts_->calc(d->multibody.contacts, d->tmp_xstatic);
  contacts_->setComputeAllContacts(true);

  d->tmp_Jstatic.conservativeResize(nv, nu + nc);
  d->tmp_Jstatic.leftCols(nu) = d->multibody.actuation->dtau_du;
  d->tmp_Jstatic.rightCols(nc) =
      d->multibody.contacts->Jc.topRows(nc).transpose();
  d->tmp_rstatic.noalias() = pseudoInverse(d->tmp_Jstatic) * d->pinocchio.tau;
  if (nc != 0) {
    nc = 0;
    std::size_t nc_r = 0;
    for (typename ContactModelMultiple::ContactModelContainer::const_iterator
             it_m = contacts_->get_contacts().begin();
         it_m != contacts_->get_contacts().end(); ++it_m) {
      const std::shared_ptr<ContactItem>& m_i = it_m->second;
      const std::size_t nc_i = m_i->contact->get_nc();
      if (m_i->active) {
        u.segment(nv + nc, nc_i) = d->tmp_rstatic.segment(nu + nc_r, nc_i);
        nc_r += nc_i;
      } else {
        u.segment(nv + nc, nc_i).setZero();
      }
      nc += nc_i;
    }
  }
  d->pinocchio.tau.setZero();
}

template <typename Scalar>
template <typename NewScalar>
DifferentialActionModelContactInvDynamicsTpl<NewScalar>
DifferentialActionModelContactInvDynamicsTpl<Scalar>::cast() const {
  typedef DifferentialActionModelContactInvDynamicsTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  typedef ContactModelMultipleTpl<NewScalar> ContactType;
  typedef CostModelSumTpl<NewScalar> CostType;
  typedef ConstraintModelManagerTpl<NewScalar> ConstraintType;
  if (constraints_) {
    const std::shared_ptr<ConstraintType>& constraints =
        std::make_shared<ConstraintType>(
            constraints_->template cast<NewScalar>());
    if (state_->get_nv() - actuation_->get_nu() > 0) {
      constraints->removeConstraint("tau");
    }
    if (contacts_->get_nc_total() != 0) {
      typename ContactModelMultiple::ContactModelContainer contact_list;
      contact_list = contacts_->get_contacts();
      typename ContactModelMultiple::ContactModelContainer::iterator it_m,
          end_m;
      for (it_m = contact_list.begin(), end_m = contact_list.end();
           it_m != end_m; ++it_m) {
        const std::string name = it_m->second->name;
        constraints->removeConstraint(name + "_acc");
        constraints->removeConstraint(name + "_force");
      }
    }
    ReturnType ret(
        std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
        actuation_->template cast<NewScalar>(),
        std::make_shared<ContactType>(contacts_->template cast<NewScalar>()),
        std::make_shared<CostType>(costs_->template cast<NewScalar>()),
        constraints);
    return ret;
  } else {
    ReturnType ret(
        std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
        actuation_->template cast<NewScalar>(),
        std::make_shared<ContactType>(contacts_->template cast<NewScalar>()),
        std::make_shared<CostType>(costs_->template cast<NewScalar>()));
    return ret;
  }
}

template <typename Scalar>
bool DifferentialActionModelContactInvDynamicsTpl<Scalar>::checkData(
    const std::shared_ptr<DifferentialActionDataAbstract>& data) {
  std::shared_ptr<Data> d = std::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::print(
    std::ostream& os) const {
  os << "DifferentialActionModelContactInvDynamics {nx=" << state_->get_nx()
     << ", ndx=" << state_->get_ndx() << ", nu=" << nu_
     << ", nc=" << contacts_->get_nc_total() << "}";
}

template <typename Scalar>
std::size_t DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_ng()
    const {
  if (constraints_ != nullptr) {
    return constraints_->get_ng();
  } else {
    return Base::get_ng();
  }
}

template <typename Scalar>
std::size_t DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_nh()
    const {
  if (constraints_ != nullptr) {
    return constraints_->get_nh();
  } else {
    return Base::get_nh();
  }
}

template <typename Scalar>
std::size_t DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_ng_T()
    const {
  if (constraints_ != nullptr) {
    return constraints_->get_ng_T();
  } else {
    return Base::get_ng_T();
  }
}

template <typename Scalar>
std::size_t DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_nh_T()
    const {
  if (constraints_ != nullptr) {
    return constraints_->get_nh_T();
  } else {
    return Base::get_nh_T();
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_g_lb() const {
  if (constraints_ != nullptr) {
    return constraints_->get_lb();
  } else {
    return g_lb_;
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_g_ub() const {
  if (constraints_ != nullptr) {
    return constraints_->get_ub();
  } else {
    return g_lb_;
  }
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>&
DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_pinocchio() const {
  return *pinocchio_;
}

template <typename Scalar>
const std::shared_ptr<ActuationModelAbstractTpl<Scalar>>&
DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_actuation() const {
  return actuation_;
}

template <typename Scalar>
const std::shared_ptr<ContactModelMultipleTpl<Scalar>>&
DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_contacts() const {
  return contacts_;
}

template <typename Scalar>
const std::shared_ptr<CostModelSumTpl<Scalar>>&
DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_costs() const {
  return costs_;
}

template <typename Scalar>
const std::shared_ptr<ConstraintModelManagerTpl<Scalar>>&
DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_constraints() const {
  return constraints_;
}

}  // namespace crocoddyl
