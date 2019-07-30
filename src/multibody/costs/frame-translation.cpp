#include "crocoddyl/multibody/costs/frame-translation.hpp"
#include "pinocchio/algorithm/frames.hpp"

namespace crocoddyl {

CostModelFrameTranslation::CostModelFrameTranslation(pinocchio::Model* const model,
                                                     ActivationModelAbstract* const activation,
                                                     const FrameTranslation& xref, const unsigned int& nu)
    : CostModelAbstract(model, activation, nu), xref_(xref) {
  assert(activation->get_nr() == 3 && "CostModelFrameTranslation: activation::nr is not equals to 3");
}

CostModelFrameTranslation::CostModelFrameTranslation(pinocchio::Model* const model,
                                                     ActivationModelAbstract* const activation,
                                                     const FrameTranslation& xref)
    : CostModelAbstract(model, activation), xref_(xref) {
  assert(activation->get_nr() == 3 && "CostModelFramePlacement: activation::nr is not equals to 3");
}

CostModelFrameTranslation::CostModelFrameTranslation(pinocchio::Model* const model, const FrameTranslation& xref,
                                                     const unsigned int& nu)
    : CostModelAbstract(model, 3, nu), xref_(xref) {}

CostModelFrameTranslation::CostModelFrameTranslation(pinocchio::Model* const model, const FrameTranslation& xref)
    : CostModelAbstract(model, 3), xref_(xref) {}

CostModelFrameTranslation::~CostModelFrameTranslation() {}

void CostModelFrameTranslation::calc(boost::shared_ptr<CostDataAbstract>& data,
                                     const Eigen::Ref<const Eigen::VectorXd>&,
                                     const Eigen::Ref<const Eigen::VectorXd>&) {
  // Compute the frame translation w.r.t. the reference frame
  data->r = data->pinocchio->oMf[xref_.frame].translation() - xref_.oxf;

  // Compute the cost
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

void CostModelFrameTranslation::calcDiff(boost::shared_ptr<CostDataAbstract>& data,
                                         const Eigen::Ref<const Eigen::VectorXd>& x,
                                         const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }
  // Update the frame placements
  CostDataFrameTranslation* d = static_cast<CostDataFrameTranslation*>(data.get());
  pinocchio::updateFramePlacements(*pinocchio_, *d->pinocchio);

  // Compute the frame Jacobian at the error point
  pinocchio::getFrameJacobian(*pinocchio_, *d->pinocchio, xref_.frame, pinocchio::LOCAL, d->fJf);
  d->J = d->pinocchio->oMf[xref_.frame].rotation() * d->fJf.topRows<3>();

  // Compute the derivatives of the frame placement
  activation_->calcDiff(d->activation, d->r, recalc);
  d->Rx.topLeftCorner(3, nv_) = d->J;
  d->Lx.head(nv_) = d->J.transpose() * d->activation->Ar;
  d->Lxx.topLeftCorner(nv_, nv_) = d->J.transpose() * d->activation->Arr * d->J;
}

boost::shared_ptr<CostDataAbstract> CostModelFrameTranslation::createData(pinocchio::Data* const data) {
  return boost::make_shared<CostDataFrameTranslation>(this, data);
}

const FrameTranslation& CostModelFrameTranslation::get_xref() const { return xref_; }

}  // namespace crocoddyl