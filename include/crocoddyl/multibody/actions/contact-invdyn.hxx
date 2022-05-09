///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2022, Heriot-Watt University, University of Edinburgh,
//                          University of Pisa
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/math.hpp"
#include "crocoddyl/multibody/actions/contact-invdyn.hpp"
#include "crocoddyl/core/constraints/residual.hpp"

#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelContactInvDynamicsTpl<Scalar>::DifferentialActionModelContactInvDynamicsTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelAbstract> actuation,
    boost::shared_ptr<ContactModelMultiple> contacts, boost::shared_ptr<CostModelSum> costs)
    : Base(state, state->get_nv() + actuation->get_nu() + contacts->get_nc(), costs->get_nr(), 0,
           state->get_nv() + contacts->get_nc()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      constraints_(boost::make_shared<ConstraintModelManager>(
          state, state->get_nv() + actuation->get_nu() + contacts->get_nc())),
      pinocchio_(*state->get_pinocchio().get()) {
  init(state);
}

template <typename Scalar>
DifferentialActionModelContactInvDynamicsTpl<Scalar>::DifferentialActionModelContactInvDynamicsTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelAbstract> actuation,
    boost::shared_ptr<ContactModelMultiple> contacts, boost::shared_ptr<CostModelSum> costs,
    boost::shared_ptr<ConstraintModelManager> constraints)
    : Base(state, state->get_nv() + actuation->get_nu() + contacts->get_nc(), costs->get_nr(), constraints->get_ng(),
           state->get_nv() + contacts->get_nc() + constraints->get_nh()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      constraints_(constraints),
      pinocchio_(*state->get_pinocchio().get()) {
  init(state);
}

template <typename Scalar>
DifferentialActionModelContactInvDynamicsTpl<Scalar>::~DifferentialActionModelContactInvDynamicsTpl() {}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::init(const boost::shared_ptr<StateMultibody>& state) {
  if (contacts_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Contacts doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (costs_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Costs doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }
  const std::size_t nv = state_->get_nv();
  const std::size_t nu = actuation_->get_nu();
  const std::size_t nc = contacts_->get_nc();
  VectorXs lb = VectorXs::Constant(nu_, -std::numeric_limits<Scalar>::infinity());
  VectorXs ub = VectorXs::Constant(nu_, std::numeric_limits<Scalar>::infinity());
  lb.segment(nv, nu) = Scalar(-1.) * pinocchio_.effortLimit.tail(nu);
  ub.segment(nv, nu) = Scalar(1.) * pinocchio_.effortLimit.tail(nu);
  Base::set_u_lb(lb);
  Base::set_u_ub(ub);

  constraints_->addConstraint(
      "rnea",
      boost::make_shared<ConstraintModelResidual>(
          state_, boost::make_shared<typename DifferentialActionModelContactInvDynamicsTpl<Scalar>::ResidualModelRnea>(
                      state, nc, nu)));
  if (contacts_->get_nc() != 0) {
    typename ContactModelMultiple::ContactModelContainer contact_list;
    contact_list = contacts_->get_contacts();
    typename ContactModelMultiple::ContactModelContainer::iterator it_m, end_m;
    for (it_m = contact_list.begin(), end_m = contact_list.end(); it_m != end_m; ++it_m) {
      const boost::shared_ptr<ContactItem>& contact = it_m->second;
      constraints_->addConstraint(
          contact->name,
          boost::make_shared<ConstraintModelResidual>(
              state_,
              boost::make_shared<typename DifferentialActionModelContactInvDynamicsTpl<Scalar>::ResidualModelContact>(
                  state, contact->contact->get_id(), contact->contact->get_nc(), nc, nu)));
    }
  }
  constraints_->shareDimensions(this);
}

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
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nc = contacts_->get_nc();
  const std::size_t nv = state_->get_nv();
  const std::size_t nu = actuation_->get_nu();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(state_->get_nv());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a = u.head(nv);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> tau = u.segment(nv, nu);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> f_ext = u.tail(nc);

  d->xout = a;
  contacts_->updateForce(d->multibody.contacts, f_ext);
  pinocchio::rnea(pinocchio_, d->pinocchio, q, v, a, d->multibody.contacts->fext);
  pinocchio::updateGlobalPlacements(pinocchio_, d->pinocchio);
  pinocchio::centerOfMass(pinocchio_, d->pinocchio, q, v, a);
  pinocchio::computeJointJacobians(pinocchio_, d->pinocchio);

  actuation_->calc(d->multibody.actuation, x, tau);
  contacts_->calc(d->multibody.contacts, x);
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
  d->constraints->resize(this, d);
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
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nv = state_->get_nv();
  const std::size_t nu = actuation_->get_nu();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(state_->get_nv());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a = u.head(nv);
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> tau = u.segment(nv, nu);

  pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, q, v, a, d->multibody.contacts->fext);
  d->pinocchio.M.template triangularView<Eigen::StrictlyLower>() =
      d->pinocchio.M.template triangularView<Eigen::StrictlyUpper>().transpose();
  pinocchio::jacobianCenterOfMass(pinocchio_, d->pinocchio, false);

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
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nq = state_->get_nq();
  const std::size_t nv = state_->get_nv();
  const std::size_t nu = actuation_->get_nu();
  const std::size_t nc = contacts_->get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(nq);
  d->tmp_xstatic.head(nq) = q;
  d->tmp_xstatic.tail(nv).setZero();
  u.setZero();

  pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, d->tmp_xstatic.tail(nv));
  pinocchio::computeJointJacobians(pinocchio_, d->pinocchio, q);
  pinocchio::rnea(pinocchio_, d->pinocchio, q, d->tmp_xstatic.tail(nv), d->tmp_xstatic.tail(nv));
  actuation_->calc(d->multibody.actuation, d->tmp_xstatic, u.segment(nv, nu));
  actuation_->calcDiff(d->multibody.actuation, d->tmp_xstatic, u.segment(nv, nu));
  contacts_->calc(d->multibody.contacts, d->tmp_xstatic);

  d->tmp_Jstatic.resize(nv, nu + nc);
  d->tmp_Jstatic.leftCols(nu) = d->multibody.actuation->dtau_du;
  d->tmp_Jstatic.rightCols(nc) = d->multibody.contacts->Jc.topRows(nc).transpose();
  u.segment(nv, nu) = (pseudoInverse(d->tmp_Jstatic) * d->pinocchio.tau).head(nu);
  if (nc != 0) {
    d->tmp_Jcstatic.resize(nv, nc);
    d->tmp_Jcstatic = d->tmp_Jstatic.rightCols(nc);
    d->pinocchio.tau -= d->multibody.actuation->tau;
    u.tail(nc).noalias() = pseudoInverse(d->tmp_Jcstatic) * d->pinocchio.tau;
  }
  d->pinocchio.tau.setZero();
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::multiplyByFu(const Eigen::Ref<const MatrixXs>& Fu,
                                                                        const Eigen::Ref<const MatrixXs>& A,
                                                                        Eigen::Ref<MatrixXs> out,
                                                                        const AssignmentOp op) const {
  assert_pretty(is_a_AssignmentOp(op), ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));
  if (static_cast<std::size_t>(A.cols()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "number of columns of A is wrong, it should be " + std::to_string(state_->get_nv()) +
                        " instead of " + std::to_string(A.cols()));
  }
  if (A.rows() != out.rows()) {
    throw_pretty("Invalid argument: "
                 << "A and out have different number of rows: " + std::to_string(A.rows()) + " and " +
                        std::to_string(out.rows()));
  }
  if (static_cast<std::size_t>(out.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "number of columns of out is wrong, it should be " + std::to_string(nu_) + " instead of " +
                        std::to_string(out.cols()));
  }
  const std::size_t nv = state_->get_nv();
  switch (op) {
    case setto:
      out.leftCols(nv).noalias() = A * Fu.leftCols(nv);
      break;
    case addto:
      out.leftCols(nv).noalias() += A * Fu.leftCols(nv);
      break;
    case rmfrom:
      out.leftCols(nv).noalias() -= A * Fu.leftCols(nv);
      break;
    default:
      throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
  }
}

template <typename Scalar>
void DifferentialActionModelContactInvDynamicsTpl<Scalar>::multiplyFuTransposeBy(const Eigen::Ref<const MatrixXs>& Fu,
                                                                                 const Eigen::Ref<const MatrixXs>& A,
                                                                                 Eigen::Ref<MatrixXdRowMajor> out,
                                                                                 const AssignmentOp op) const {
  assert_pretty(is_a_AssignmentOp(op), ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));
  if (static_cast<std::size_t>(A.rows()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "number of rows of A is wrong, it should be " + std::to_string(state_->get_nv()) + " instead of " +
                        std::to_string(A.rows()));
  }
  if (A.cols() != out.cols()) {
    throw_pretty("Invalid argument: "
                 << "A and out have different number of columns: " + std::to_string(A.cols()) + " and " +
                        std::to_string(out.cols()));
  }
  if (static_cast<std::size_t>(out.rows()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "number of rows of out is wrong, it should be " + std::to_string(nu_) + " instead of " +
                        std::to_string(out.cols()));
  }
  const std::size_t nv = state_->get_nv();
  switch (op) {
    case setto:
      out.topRows(nv).noalias() = Fu.transpose().topRows(nv) * A;
      break;
    case addto:
      out.topRows(nv).noalias() += Fu.transpose().topRows(nv) * A;
      break;
    case rmfrom:
      out.topRows(nv).noalias() -= Fu.transpose().topRows(nv) * A;
      break;
    default:
      throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
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

}  // namespace crocoddyl
