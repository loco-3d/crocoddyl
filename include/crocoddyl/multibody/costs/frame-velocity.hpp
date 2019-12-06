///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_FRAME_VELOCITY_HPP_
#define CROCODDYL_MULTIBODY_COSTS_FRAME_VELOCITY_HPP_

#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/frames.hpp"

namespace crocoddyl {

class CostModelFrameVelocity : public CostModelAbstract {
 public:
  CostModelFrameVelocity(boost::shared_ptr<StateMultibody> state,
                         boost::shared_ptr<ActivationModelAbstract> activation, const FrameMotion& Fref,
                         const std::size_t& nu);
  CostModelFrameVelocity(boost::shared_ptr<StateMultibody> state,
                         boost::shared_ptr<ActivationModelAbstract> activation, const FrameMotion& Fref);
  CostModelFrameVelocity(boost::shared_ptr<StateMultibody> state, const FrameMotion& vref, const std::size_t& nu);
  CostModelFrameVelocity(boost::shared_ptr<StateMultibody> state, const FrameMotion& vref);
  ~CostModelFrameVelocity();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  const FrameMotion& get_vref() const;
  void set_vref(const FrameMotion& vref_in);

 private:
  FrameMotion vref_;
};

struct CostDataFrameVelocity : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataFrameVelocity(Model* const model, DataCollectorAbstract* const data)
      : CostDataAbstract(model, data),
        joint(model->get_state()->get_pinocchio().frames[model->get_vref().frame].parent),
        vr(pinocchio::Motion::Zero()),
        fXj(model->get_state()->get_pinocchio().frames[model->get_vref().frame].placement.inverse().toActionMatrix()),
        dv_dq(6, model->get_state()->get_nv()),
        dv_dv(6, model->get_state()->get_nv()),
        Arr_Rx(6, model->get_state()->get_nv()) {
    dv_dq.fill(0);
    dv_dv.fill(0);
    Arr_Rx.fill(0);
    // Check that proper shared data has been passed
    DataCollectorMultibody* d = dynamic_cast<DataCollectorMultibody*>(shared);
    if (d == NULL) {
      throw std::invalid_argument("the shared data should be derived from DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::Data* pinocchio;
  pinocchio::JointIndex joint;
  pinocchio::Motion vr;
  pinocchio::SE3::ActionMatrixType fXj;
  pinocchio::Data::Matrix6x dv_dq;
  pinocchio::Data::Matrix6x dv_dv;
  pinocchio::Data::Matrix6x Arr_Rx;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_VELOCITY_HPP_
