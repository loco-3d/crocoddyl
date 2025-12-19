#ifndef CROCODDYL_MULTIBODY_UTILS_BIPED_GAITS_HPP_
#define CROCODDYL_MULTIBODY_UTILS_BIPED_GAITS_HPP_

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/frame.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/se3.hpp>

#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/residuals/joint-effort.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/multibody/actions/contact-invdyn.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/residuals/frame-velocity.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/multibody/wrench-cone.hpp"

namespace crocoddyl {
class SimpleBipedGaitProblem {
 public:
  struct CostWeight {
    Eigen::VectorXd x_weights = Eigen::VectorXd::Zero(0);
    double state_weight = 1e1;
    double control_weight = 1e-1;
    double com_track_weight = 1e6;
    double foot_track_weight = 1e6;
    double contact_wrench_weight = 1e1;
    double impulse_foot_vel_weight = 1e6;
  };

  SimpleBipedGaitProblem(pinocchio::Model& rmodel, std::string left_foot,
                         std::string right_foot, int num_steps,
                         const CostWeight& cost_weight);

  SimpleBipedGaitProblem(pinocchio::Model& rmodel, std::string left_foot,
                         std::string right_foot, int num_steps);

  ~SimpleBipedGaitProblem() {};

  std::shared_ptr<crocoddyl::ShootingProblem> createWalkingProblem(
      const Eigen::VectorXd& x0, const double stepLength,
      const double stepHeight, const double timeStep,
      const std::size_t stepKnots, const std::size_t supportKnots,
      bool fwddyn = true);

  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>
  createFootStepModels(
      double timeStep, Eigen::Vector3d& comPos0,
      std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>>& feetPos0,
      const double stepLength, const double stepHeight,
      const std::size_t numKnots,
      const std::vector<pinocchio::FrameIndex>& supportFootIds,
      const std::vector<pinocchio::FrameIndex>& swingFootIds);

  std::shared_ptr<ActionModelAbstract> createSwingFootModel(
      double timeStep, const std::vector<pinocchio::FrameIndex>& supportFootIds,
      const Eigen::Vector3d& comTask = Eigen::Vector3d::Zero(),
      const std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>>&
          swingFootTask =
              std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>>());

  std::shared_ptr<ActionModelAbstract> createFootSwitchModel(
      const std::vector<pinocchio::FrameIndex>& supportFootIds,
      const std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>>&
          swingFootTask);

 private:
  pinocchio::Model rmodel_;
  pinocchio::Data rdata_;
  pinocchio::FrameIndex left_foot_id_, right_foot_id_;
  std::shared_ptr<StateMultibody> state_;
  std::shared_ptr<ActuationModelFloatingBase> actuation_;
  bool fwddyn_;
  Eigen::VectorXd q0_;
  int num_steps_;
  bool first_step_ = true;
  CostWeight cost_weight_;
};
}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_UTILS_BIPED_GAITS_HPP_
