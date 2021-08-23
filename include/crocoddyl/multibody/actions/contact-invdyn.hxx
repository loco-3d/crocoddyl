///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Pisa
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/math.hpp"
#include "crocoddyl/multibody/actions/contact-invdyn.hpp"

#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelContactInvDynamicsTpl<Scalar>::DifferentialActionModelContactFwdDynamicsTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelAbstract> actuation,
    boost::shared_ptr<ContactModelMultiple> contacts, boost::shared_ptr<CostModelSum> costs)
    : Base(state, state->get_nv() + actuation->get_nu() + contacts.get_nc(), costs->get_nr(), 0, state->get_nv() + contacts->get_nc()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      constraints_(boost::make_shared<ConstraintModelManager>(state, state->get_nv() + actuation->get_nu()+ contacts.get_nc())),
      pinocchio_(*state->get_pinocchio().get()){
  if (contacts_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Contacts doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (costs_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Costs doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }
    const std::size_t nu =  actuation_->get_nu();
    const std::size_t nh = state_->get_nv() + contacts_->get_nc();
  VectorXs lb = VectorXs::Constant(nu_, -std::numeric_limits<Scalar>::infinity());
  VectorXs ub = VectorXs::Constant(nu_, std::numeric_limits<Scalar>::infinity());
  lb.tail(nu) = Scalar(-1.) * pinocchio_.effortLimit.tail(nu);
  ub.tail(nu) = Scalar(1.) * pinocchio_.effortLimit.tail(nu);
  Base::set_u_lb(lb);
  Base::set_u_ub(ub);

  constraints_->addConstraint(
      "rnea",
      boost::make_shared<ConstraintModelResidual>(
          state_, boost::make_shared<typename DifferentialActionModelContactInvDynamicsTpl<Scalar>::ResidualModelRnea>(
                      state, nc, nu)));
  typename ContactModelContainer::iterator it_m, end_m;
  for (it_m = contacts_.begin(), end_m = contacts_.end();
            it_m != end_m; ++it_m) {
            constraints_->addConstraint(it_m->second->name, 
            boost::make_shared<ConstraintModelResidual>(
                state_, boost::make_shared<typename DifferentialActionModelContactInvDynamicsTpl<Scalar>::ResidualModelContact>(
                      state, nc, nu))
            );            
            }
}

template <typename Scalar>
DifferentialActionModelContactInvDynamicsTpl<Scalar>::DifferentialActionModelContactInvDynamicsTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelAbstract> actuation,
    boost::shared_ptr<ContactModelMultiple> contacts, boost::shared_ptr<CostModelSum> costs,
    boost::shared_ptr<ConstraintModelManager> constraints)
    : Base(state, state->get_nv() + actuation->get_nu() + contacts.get_nc(), costs->get_nr(), constraints->get_ng(), state->get_nv() + contacts->get_nc() + constraints->get_nh()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      constraints_(constraints),
      pinocchio_(*state->get_pinocchio().get())
  if (contacts_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Contacts doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (costs_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Costs doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }
  VectorXs lb = VectorXs::Constant(nu_, -std::numeric_limits<Scalar>::infinity());
  VectorXs ub = VectorXs::Constant(nu_, std::numeric_limits<Scalar>::infinity());
  lb.tail(nu) = Scalar(-1.) * pinocchio_.effortLimit.tail(nu);
  ub.tail(nu) = Scalar(1.) * pinocchio_.effortLimit.tail(nu);
  Base::set_u_lb(lb);
  Base::set_u_ub(ub);
    
    constraints_->addConstraint(
      "rnea",
      boost::make_shared<ConstraintModelResidual>(
          state_, boost::make_shared<typename DifferentialActionModelContactInvDynamicsTpl<Scalar>::ResidualModelRnea>(
                      state, nc, nu)));
  typename ContactModelContainer::iterator it_m, end_m;
  for (it_m = contacts_.begin(), end_m = contacts_.end();
            it_m != end_m; ++it_m) {
            constraints_->addConstraint(it_m->second->name, 
            boost::make_shared<ConstraintModelResidual>(
                state_, boost::make_shared<typename DifferentialActionModelContactInvDynamicsTpl<Scalar>::ResidualModelContact>(
                      state, nc, nu))
            );            
            }
}

template <typename Scalar>
DifferentialActionModelContactInvDynamicsTpl<Scalar>::~DifferentialActionModelContactInvDynamicsTpl() {}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  const std::size_t nc = contacts_->get_nc();
  const std::size_t nv = state_->get_nv();
  const std::size_t nu = actuation_->get_nu();
  
  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(state_->get_nv());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a = u.head(nv);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> tau = u.segment(nv, nv+nu);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> f_ext = u.tail(nc);
  
  d->xout = a;
  contacts_->updateForce(d->multibody.contacts, f_ext);
  pinocchio::rnea(pinocchio_, d->pinocchio, q, v, a, d->contacts->fext);
  pinocchio::updateGlobalPlacements(pinocchio_, d->pinocchio);
  pinocchio::centerOfMass(pinocchio_, d->pinocchio, q, v, a)
  pinocchio::computeJointJacobians(pinocchio_, d->pinocchio, q);
  pinocchio::jacobianCenterOfMass(pinocchio_, d->pinocchio, q);
  
  actuation_->calc(d->multibody.actuation, x, tau);
  contacts_->calc(d->multibody.contacts, x);

  // Computing the cost value and residuals
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
  constraints_->calc(d->constraints, x, u);
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  const std::size_t nc = contacts_->get_nc();
  const std::size_t nv = state_->get_nv();
  const std::size_t nu = actuation_->get_nu();
  
  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(state_->get_nv());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a = u.head(nv);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> tau = u.segment(nv, nv+nu);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> f_ext = u.tail(nc);

  pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, q, v, a, d->contacts->fext);

  actuation_->calcDiff(d->multibody.actuation, x, tau);
  contacts_->calcDiff(d->multibody.contacts, x);

  costs_->calcDiff(d->costs, x, u);
  constraints_->calcDiff(d->constraints, x, u);

}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelContactInvDynamicsTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::quasiStatic(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
    const Eigen::Ref<const VectorXs>& x, std::size_t, Scalar) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  // Static casting the data
  pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, d->tmp_xstatic.tail(nv));
  pinocchio::computeJointJacobians(pinocchio_, d->pinocchio, q);
  pinocchio::rnea(pinocchio_, d->pinocchio, q, d->tmp_xstatic.tail(nv), d->tmp_xstatic.tail(nv));
  
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nc = contacts_->get_nc();
  const std::size_t nu = actuation_->get_nu();
  
  if (nc!=0){
    Data* d = static_cast<Data*>(data.get());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());

  d->tmp_xstatic.head(nq) = q;
  d->tmp_xstatic.tail(nv) *= 0;
  d->tmp_ustatic.setZero();
  // Check the velocity input is zero
  assert_pretty(x.tail(nv).isZero(), "The velocity input should be zero for quasi-static to work.");


  actuation_->calc(d->multibody.actuation, d->tmp_xstatic, d->tmp_ustatic.head(nu));
  actuation_->calcDiff(d->multibody.actuation, d->tmp_xstatic, d->tmp_ustatic.head(nu));
  contacts_->calc(d->multibody.contacts, d->tmp_xstatic);

  // Allocates memory
  d->tmp_Jstatic.resize(nv, nu + nc);
  d->tmp_Jstatic << d->multibody.actuation->dtau_du, d->multibody.contacts->Jc.topRows(nc).transpose();

  u.segment(nv,nv+nu).noalias() = pseudoInverse(d->tmp_Jstatic.leftCols(nu+nc)) * d->pinocchio.tau.head(nu);
  u.tail(nc).noalias() = pseudoInverse(d->tmp_Jstatic.leftCols(nu+nc)) * (d->pinocchio.tau-);
  d->pinocchio.tau.setZero();
  
  }
  else{
    u.segment(nv,nv+nu).noalias() =  d->pinocchio.tau.tail(nu);
    d->pinocchio.tau.setZero();
  }

}

template <typename Scalar>
bool DifferentialActionModelContactInvDynamicsTpl<Scalar>::checkData(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::print(std::ostream& os) const {
  os << "DifferentialActionModelContactInvDynamics {nx=" << state_->get_nx() << ", ndx=" << state_->get_ndx()
     << ", nu=" << nu_ << ", nc=" << contacts_->get_nc() << "}";
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>& DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_pinocchio() const {
  return pinocchio_;
}

template <typename Scalar>
const boost::shared_ptr<ActuationModelAbstractTpl<Scalar> >&
DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_actuation() const {
  return actuation_;
}

template <typename Scalar>
const boost::shared_ptr<ContactModelMultipleTpl<Scalar> >&
DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_contacts() const {
  return contacts_;
}

template <typename Scalar>
const boost::shared_ptr<CostModelSumTpl<Scalar> >& DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_costs()
    const {
  return costs_;
}

template <typename Scalar>
const boost::shared_ptr<ConstraintModelManagerTpl<Scalar> >&
DifferentialActionModelContactInvDynamicsTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::ResidualModelContactTpl<Scalar>::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) {

      const boost::shared_ptr<typename Data::ResidualDataRnea>& d =
          boost::static_pointer_cast<typename Data::ResidualDataRnea>(data);
        d->r = d->contact->a0;
    }

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::ResidualModelContactTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) {

      const boost::shared_ptr<typename Data::ResidualDataRnea>& d =
          boost::static_pointer_cast<typename Data::ResidualDataRnea>(data);
        d->Rx = d->contact->da0_dx;
        d->Ru.rightCols(state_->get_nv()) = d->contact->Jc;
    }

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar>>
DifferentialActionModelContactInvDynamicsTpl<Scalar>::ResidualModelContactTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}


}  // namespace crocoddyl
