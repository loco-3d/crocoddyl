///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
#define CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_

#include "crocoddyl/multibody/states/multibody.hpp"
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/force.hpp>

namespace crocoddyl {

struct ImpulseDataAbstract;  // forward declaration

class ImpulseModelAbstract {
 public:
  ImpulseModelAbstract(StateMultibody& state, unsigned int const& ni);
  ~ImpulseModelAbstract();

  virtual void calc(const boost::shared_ptr<ImpulseDataAbstract>& data,
                    const Eigen::Ref<const Eigen::VectorXd>& x) = 0;
  virtual void calcDiff(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                        const bool& recalc = true) = 0;
  virtual void updateForce(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::VectorXd& force) = 0;
  void updateVelocity(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::VectorXd& vnext) const;

  void updateVelocityDiff(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::MatrixXd& dvnext_dx) const;

  virtual boost::shared_ptr<ImpulseDataAbstract> createData(pinocchio::Data* const data);

  StateMultibody& get_state() const;
  unsigned int const& get_ni() const;

 protected:
  StateMultibody& state_;
  unsigned int ni_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::VectorXd& x) { calc(data, x); }

  void calcDiff_wrap(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::VectorXd& x,
                     const bool& recalc = true) {
    calcDiff(data, x, recalc);
  }

#endif
};

struct ImpulseDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ImpulseDataAbstract(Model* const model, pinocchio::Data* const data)
      : pinocchio(data),
        joint(0),
        vnext(model->get_state().get_nv()),
        dvnext_dx(model->get_state().get_nv(), model->get_state().get_ndx()),
        Jc(model->get_ni(), model->get_state().get_nv()),
        dv_dq(model->get_ni(), model->get_state().get_nv()),
        f(pinocchio::Force::Zero()) {
    vnext.fill(0);
    dvnext_dx.fill(0);
    Jc.fill(0);
    dv_dq.fill(0);
  }

  pinocchio::Data* pinocchio;
  pinocchio::JointIndex joint;
  Eigen::VectorXd vnext;
  Eigen::MatrixXd dvnext_dx;
  Eigen::MatrixXd Jc;
  Eigen::MatrixXd dv_dq;
  pinocchio::Force f;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
