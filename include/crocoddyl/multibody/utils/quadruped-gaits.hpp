///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_UTILS_QUADRUPED_GAITS_HPP_
#define CROCODDYL_MULTIBODY_UTILS_QUADRUPED_GAITS_HPP_

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/frame.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/se3.hpp>

#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/multibody/actions/impulse-fwddyn.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/residuals/frame-velocity.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"

namespace crocoddyl {

class SimpleQuadrupedGaitProblem {
 public:
  SimpleQuadrupedGaitProblem(const pinocchio::Model& rmodel,
                             const std::string& lf_foot,
                             const std::string& rf_foot,
                             const std::string& lh_foot,
                             const std::string& rh_foot);
  ~SimpleQuadrupedGaitProblem();

  std::shared_ptr<crocoddyl::ShootingProblem> createWalkingProblem(
      const Eigen::VectorXd& x0, const double stepLength,
      const double stepHeight, const double timeStep,
      const std::size_t stepKnots, const std::size_t supportKnots);

  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract> >
  createFootStepModels(const double timeStep, Eigen::Vector3d& comPos0,
                       std::vector<Eigen::Vector3d>& feetPos0,
                       const double stepLength, const double stepHeight,
                       const std::size_t numKnots,
                       const std::vector<pinocchio::FrameIndex>& supportFootIds,
                       const std::vector<pinocchio::FrameIndex>& swingFootIds);

  std::shared_ptr<ActionModelAbstract> createSwingFootModel(
      const double timeStep,
      const std::vector<pinocchio::FrameIndex>& supportFootIds,
      const Eigen::Vector3d& comTask =
          Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity()),
      const std::vector<pinocchio::FrameIndex>& swingFootIds =
          std::vector<pinocchio::FrameIndex>(),
      const std::vector<pinocchio::SE3>& swingFootTask =
          std::vector<pinocchio::SE3>());

  std::shared_ptr<ActionModelAbstract> createFootSwitchModel(
      const std::vector<pinocchio::FrameIndex>& supportFootIds,
      const std::vector<pinocchio::FrameIndex>& swingFootIds,
      const std::vector<pinocchio::SE3>& swingFootTask,
      const bool pseudoImpulse = false);

  std::shared_ptr<ActionModelAbstract> createPseudoImpulseModel(
      const std::vector<pinocchio::FrameIndex>& supportFootIds,
      const std::vector<pinocchio::FrameIndex>& swingFootIds,
      const std::vector<pinocchio::SE3>& swingFootTask);

  std::shared_ptr<ActionModelAbstract> createImpulseModel(
      const std::vector<pinocchio::FrameIndex>& supportFootIds,
      const std::vector<pinocchio::FrameIndex>& swingFootIds,
      const std::vector<pinocchio::SE3>& ref_swingFootTask);

  const Eigen::VectorXd& get_defaultState() const;

 protected:
  pinocchio::Model rmodel_;
  pinocchio::Data rdata_;
  pinocchio::FrameIndex lf_foot_id_, rf_foot_id_, lh_foot_id_, rh_foot_id_;
  std::shared_ptr<StateMultibody> state_;
  std::shared_ptr<ActuationModelFloatingBase> actuation_;
  bool firtstep_;
  Eigen::VectorXd defaultstate_;
};
}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_UTILS_QUADRUPED_GAITS_HPP_
