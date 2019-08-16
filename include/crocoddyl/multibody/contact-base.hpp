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
#include <pinocchio/spatial/force.hpp>

namespace crocoddyl {

struct ContactDataAbstract;  // forward declaration

class ContactModelAbstract {
 public:
  ContactModelAbstract(StateMultibody& state, unsigned int const& nc);
  ~ContactModelAbstract();

  virtual void calc(const boost::shared_ptr<ContactDataAbstract>& data,
                    const Eigen::Ref<const Eigen::VectorXd>& x) = 0;
  virtual void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                        const bool& recalc = true) = 0;
  virtual void updateLagrangian(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::VectorXd& lambda) = 0;
  virtual boost::shared_ptr<ContactDataAbstract> createData(pinocchio::Data* const data);

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
        Ax(model->get_nc(), model->get_state().get_ndx()),
        fext(model->get_state().get_pinocchio().njoints, pinocchio::Force::Zero()) {
    Jc.fill(0);
    a0.fill(0);
    Ax.fill(0);
  }

  pinocchio::Data* pinocchio;
  Eigen::MatrixXd Jc;
  Eigen::VectorXd a0;
  Eigen::MatrixXd Ax;
  pinocchio::container::aligned_vector<pinocchio::Force> fext;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
