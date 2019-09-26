///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_FRAME_ROTATION_HPP_
#define CROCODDYL_MULTIBODY_COSTS_FRAME_ROTATION_HPP_

#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/frames.hpp"

namespace crocoddyl {

class CostModelFrameRotation : public CostModelAbstract {
 public:
  CostModelFrameRotation(StateMultibody& state, ActivationModelAbstract& activation, const FrameRotation& Fref,
                         unsigned int const& nu);
  CostModelFrameRotation(StateMultibody& state, ActivationModelAbstract& activation, const FrameRotation& Fref);
  CostModelFrameRotation(StateMultibody& state, const FrameRotation& Fref, unsigned int const& nu);
  CostModelFrameRotation(StateMultibody& state, const FrameRotation& Fref);
  ~CostModelFrameRotation();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataAbstract> createData(pinocchio::Data* const data);

  const FrameRotation& get_Rref() const;

 private:
  FrameRotation Rref_;
  Eigen::Matrix3d oRf_inv_;
};

struct CostDataFrameRotation : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataFrameRotation(Model* const model, pinocchio::Data* const data)
      : CostDataAbstract(model, data),
        J(3, model->get_state().get_nv()),
        rJf(3, 3),
        fJf(6, model->get_state().get_nv()),
        Arr_J(3, model->get_state().get_nv()) {
    r.fill(0);
    rRf.setIdentity();
    J.fill(0);
    rJf.fill(0);
    fJf.fill(0);
    Arr_J.fill(0);
  }

  Eigen::Vector3d r;
  Eigen::Matrix3d rRf;
  pinocchio::Data::Matrix3x J;
  Eigen::Matrix3d rJf;
  pinocchio::Data::Matrix6x fJf;
  pinocchio::Data::Matrix3x Arr_J;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_ROTATION_HPP_
