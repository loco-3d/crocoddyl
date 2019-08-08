#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include <pinocchio/algorithm/frames.hpp>

namespace crocoddyl {

ContactModel3D::ContactModel3D(StateMultibody& state, const FrameTranslation& xref, const Eigen::Vector2d& gains)
    : ContactModelAbstract(state, 3), xref_(xref), gains_(gains) {}

ContactModel3D::~ContactModel3D() {}

void ContactModel3D::calc(const boost::shared_ptr<ContactDataAbstract>& data,
                          const Eigen::Ref<const Eigen::VectorXd>& x) {
  // ContactData3D* d = static_cast<ContactDataAbstract*>(data.get());
  // d->v = pinocchio::getFrameVelocity(state_.get_pinocchio(),*d->pinocchio, xref_.frame);
}

void ContactModel3D::calcDiff(const boost::shared_ptr<ContactDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& x, const bool& recalc) {
  // ContactData3D* d = static_cast<ContactDataAbstract*>(data.get());
}

boost::shared_ptr<ContactDataAbstract> ContactModel3D::createData(pinocchio::Data* const data) {
  return boost::make_shared<ContactData3D>(this, data);
}

}  // namespace crocoddyl