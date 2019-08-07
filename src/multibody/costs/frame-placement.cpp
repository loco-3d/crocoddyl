#include "crocoddyl/multibody/costs/frame-placement.hpp"
#include "pinocchio/algorithm/frames.hpp"

namespace crocoddyl {

CostModelFramePlacement::CostModelFramePlacement(StateMultibody& state, ActivationModelAbstract& activation,
                                                 const FramePlacement& Mref, const unsigned int& nu)
    : CostModelAbstract(state, activation, nu), Mref_(Mref), oMf_inv_(Mref.oMf.inverse()), nv_(state.get_nv()) {
  assert(activation_.get_nr() == 6 && "CostModelFramePlacement: activation::nr is not equals to 6");
}

CostModelFramePlacement::CostModelFramePlacement(StateMultibody& state, ActivationModelAbstract& activation,
                                                 const FramePlacement& Mref)
    : CostModelAbstract(state, activation), Mref_(Mref), oMf_inv_(Mref.oMf.inverse()), nv_(state.get_nv()) {
  assert(activation_.get_nr() == 6 && "CostModelFramePlacement: activation::nr is not equals to 6");
}

CostModelFramePlacement::CostModelFramePlacement(StateMultibody& state, const FramePlacement& Mref,
                                                 const unsigned int& nu)
    : CostModelAbstract(state, 6, nu), Mref_(Mref), oMf_inv_(Mref.oMf.inverse()), nv_(state.get_nv()) {}

CostModelFramePlacement::CostModelFramePlacement(StateMultibody& state, const FramePlacement& Mref)
    : CostModelAbstract(state, 6), Mref_(Mref), oMf_inv_(Mref.oMf.inverse()), nv_(state.get_nv()) {}

CostModelFramePlacement::~CostModelFramePlacement() {}

void CostModelFramePlacement::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                   const Eigen::Ref<const Eigen::VectorXd>&,
                                   const Eigen::Ref<const Eigen::VectorXd>&) {
  CostDataFramePlacement* d = static_cast<CostDataFramePlacement*>(data.get());

  // Compute the frame placement w.r.t. the reference frame
  d->rMf = oMf_inv_ * d->pinocchio->oMf[Mref_.frame];
  d->r = pinocchio::log6(d->rMf);
  data->r = d->r;  // this is needed because we overwrite it

  // Compute the cost
  activation_.calc(d->activation, d->r);
  d->cost = d->activation->a_value;
}

void CostModelFramePlacement::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                       const Eigen::Ref<const Eigen::VectorXd>& x,
                                       const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }
  // Update the frame placements
  CostDataFramePlacement* d = static_cast<CostDataFramePlacement*>(data.get());
  pinocchio::updateFramePlacements(state_.get_pinocchio(), *d->pinocchio);

  // Compute the frame Jacobian at the error point
  pinocchio::Jlog6(d->rMf, d->rJf);
  pinocchio::getFrameJacobian(state_.get_pinocchio(), *d->pinocchio, Mref_.frame, pinocchio::LOCAL, d->fJf);
  d->J = d->rJf * d->fJf;

  // Compute the derivatives of the frame placement
  activation_.calcDiff(d->activation, d->r, recalc);
  d->Rx.topLeftCorner(6, nv_) = d->J;
  d->Lx.head(nv_) = d->J.transpose() * d->activation->Ar;
  d->Lxx.topLeftCorner(nv_, nv_) = d->J.transpose() * d->activation->Arr * d->J;
}

boost::shared_ptr<CostDataAbstract> CostModelFramePlacement::createData(pinocchio::Data* const data) {
  return boost::make_shared<CostDataFramePlacement>(this, data);
}

const FramePlacement& CostModelFramePlacement::get_Mref() const { return Mref_; }

}  // namespace crocoddyl