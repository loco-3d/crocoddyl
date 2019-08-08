///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
#define CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_

#include "crocoddyl/multibody/states/multibody.hpp"
#include <pinocchio/multibody/data.hpp>

namespace crocoddyl {

struct ContactDataAbstract;  // forward declaration

class ContactModelAbstract {
 public:
  ContactModelAbstract(StateMultibody& state, const unsigned int& nc);
  ~ContactModelAbstract();

  void calc(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const bool& recalc = true);
  boost::shared_ptr<ContactDataAbstract> createData(pinocchio::Data* const data);

  StateMultibody& get_state() const;
  const unsigned int& get_nc() const;

 protected:
  StateMultibody& state_;
  unsigned int nc_;

#ifdef PYTHON_BINDINGS
 public:
  void calc_wrap(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::VectorXd& x) { calc(data, x); }

  void calcDiff_wrap(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::VectorXd& x,
                     const bool& recalc = true) {
    calcDiff(data, x, recalc);
  }
#endif
};

struct ContactDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ContactDataAbstract(Model* const model, pinocchio::Data* const data)
      : pinocchio(data),
        Jc(model->get_nc(), model->get_state().get_nv()),
        a0(model->get_nc()),
        Ax(model->get_nc(), model->get_state().get_ndx()) {
    Jc.fill(0);
    a0.fill(0);
    Ax.fill(0);
  }

  pinocchio::Data* pinocchio;
  Eigen::MatrixXd Jc;
  Eigen::VectorXd a0;
  Eigen::MatrixXd Ax;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
