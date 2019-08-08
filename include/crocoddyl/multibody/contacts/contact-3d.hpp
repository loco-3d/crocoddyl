///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACTS_CONTACT_3D_HPP_
#define CROCODDYL_MULTIBODY_CONTACTS_CONTACT_3D_HPP_

#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/multibody/data.hpp>

namespace crocoddyl {

class ContactModel3D : public ContactModelAbstract {
 public:
  ContactModel3D(StateMultibody& state, const FrameTranslation& xref, const Eigen::Vector2d& gains);
  ~ContactModel3D();

  void calc(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const bool& recalc = true);
  boost::shared_ptr<ContactDataAbstract> createData(pinocchio::Data* const data);

 private:
  FrameTranslation xref_;
  Eigen::Vector2d gains_;
};

struct ContactData3D : public ContactDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ContactData3D(Model* const model, pinocchio::Data* const data)
      : ContactDataAbstract(model, data), jMf(pinocchio::SE3::Identity()), Jw(3, model->get_state().get_ndx()) {
    fXj.fill(0);
    vv.fill(0);
    vw.fill(0);
  }

  pinocchio::SE3 jMf;
  pinocchio::SE3::ActionMatrixType fXj;
  pinocchio::Motion v;
  Eigen::Vector3d vv;
  Eigen::Vector3d vw;
  pinocchio::Data::Matrix3x Jw;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_CONTACTS_CONTACT_3D_HPP_
