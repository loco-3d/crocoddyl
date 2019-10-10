///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_IMPULSES_IMPULSE_6D_HPP_
#define CROCODDYL_MULTIBODY_IMPULSES_IMPULSE_6D_HPP_

#include "crocoddyl/multibody/impulse-base.hpp"
#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/multibody/data.hpp>

namespace crocoddyl {

class ImpulseModel6D : public ImpulseModelAbstract {
 public:
  ImpulseModel6D(boost::shared_ptr<StateMultibody> state, const std::size_t& frame);
  ~ImpulseModel6D();

  void calc(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const bool& recalc = true);
  void updateForce(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::VectorXd& force);
  boost::shared_ptr<ImpulseDataAbstract> createData(pinocchio::Data* const data);

  const std::size_t& get_frame() const;

 private:
  std::size_t frame_;
};

struct ImpulseData6D : public ImpulseDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ImpulseData6D(Model* const model, pinocchio::Data* const data)
      : ImpulseDataAbstract(model, data),
        jMf(model->get_state()->get_pinocchio().frames[model->get_frame()].placement),
        fXj(jMf.inverse().toActionMatrix()),
        fJf(6, model->get_state()->get_nv()),
        v_partial_dq(6, model->get_state()->get_nv()),
        v_partial_dv(6, model->get_state()->get_nv()) {
    joint = model->get_state()->get_pinocchio().frames[model->get_frame()].parent;
    fJf.fill(0);
    v_partial_dq.fill(0);
    v_partial_dv.fill(0);
  }

  pinocchio::SE3 jMf;
  pinocchio::SE3::ActionMatrixType fXj;
  pinocchio::Data::Matrix6x fJf;
  pinocchio::Data::Matrix6x v_partial_dq;
  pinocchio::Data::Matrix6x v_partial_dv;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_IMPULSES_IMPULSE_6D_HPP_
