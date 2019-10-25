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
  ImpulseModelAbstract(boost::shared_ptr<StateMultibody> state, const std::size_t& ni);
  ~ImpulseModelAbstract();

  virtual void calc(const boost::shared_ptr<ImpulseDataAbstract>& data,
                    const Eigen::Ref<const Eigen::VectorXd>& x) = 0;
  virtual void calcDiff(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                        const bool& recalc = true) = 0;

  virtual void updateForce(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::VectorXd& force) = 0;
  void updateForceDiff(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::MatrixXd& df_dq) const;

  virtual boost::shared_ptr<ImpulseDataAbstract> createData(pinocchio::Data* const data);

  const boost::shared_ptr<StateMultibody>& get_state() const;
  const std::size_t& get_ni() const;

 protected:
  boost::shared_ptr<StateMultibody> state_;
  std::size_t ni_;

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
        Jc(model->get_ni(), model->get_state()->get_nv()),
        dv0_dq(model->get_ni(), model->get_state()->get_nv()),
        df_dq(model->get_ni(), model->get_state()->get_nv()),
        f(pinocchio::Force::Zero()) {
    Jc.fill(0);
    dv0_dq.fill(0);
    df_dq.fill(0);
  }
  virtual ~ImpulseDataAbstract() {}

  pinocchio::Data* pinocchio;
  pinocchio::JointIndex joint;
  Eigen::MatrixXd Jc;
  Eigen::MatrixXd dv0_dq;
  Eigen::MatrixXd df_dq;
  pinocchio::Force f;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
