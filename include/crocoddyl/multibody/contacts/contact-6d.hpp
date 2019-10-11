///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_HPP_
#define CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_HPP_

#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/multibody/data.hpp>

namespace crocoddyl {

class ContactModel6D : public ContactModelAbstract {
 public:
  ContactModel6D(boost::shared_ptr<StateMultibody> state, const FramePlacement& xref, const std::size_t& nu,
                 const Eigen::Vector2d& gains = Eigen::Vector2d::Zero());
  ContactModel6D(boost::shared_ptr<StateMultibody> state, const FramePlacement& xref,
                 const Eigen::Vector2d& gains = Eigen::Vector2d::Zero());
  ~ContactModel6D();

  void calc(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const bool& recalc = true);
  void updateForce(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::VectorXd& force);
  boost::shared_ptr<ContactDataAbstract> createData(pinocchio::Data* const data);

  const FramePlacement& get_Mref() const;
  const Eigen::Vector2d& get_gains() const;

 private:
  FramePlacement Mref_;
  Eigen::Vector2d gains_;
};

struct ContactData6D : public ContactDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ContactData6D(Model* const model, pinocchio::Data* const data)
      : ContactDataAbstract(model, data),
        jMf(model->get_state()->get_pinocchio().frames[model->get_Mref().frame].placement),
        rMf(pinocchio::SE3::Identity()),
        fXj(jMf.inverse().toActionMatrix()),
        v_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dq(6, model->get_state()->get_nv()),
        a_partial_dv(6, model->get_state()->get_nv()),
        a_partial_da(6, model->get_state()->get_nv()) {
    joint = model->get_state()->get_pinocchio().frames[model->get_Mref().frame].parent;
    v_partial_dq.fill(0);
    a_partial_dq.fill(0);
    a_partial_dv.fill(0);
    a_partial_da.fill(0);
    rMf_Jlog6.fill(0);
  }

  pinocchio::SE3 jMf;
  pinocchio::SE3 rMf;
  pinocchio::SE3::ActionMatrixType fXj;
  pinocchio::Motion v;
  pinocchio::Motion a;
  pinocchio::Data::Matrix6x v_partial_dq;
  pinocchio::Data::Matrix6x a_partial_dq;
  pinocchio::Data::Matrix6x a_partial_dv;
  pinocchio::Data::Matrix6x a_partial_da;
  pinocchio::SE3::Matrix6 rMf_Jlog6;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_HPP_
