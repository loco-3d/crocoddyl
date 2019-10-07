#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <cmath>
#include <limits>

#include <crocoddyl/multibody/actions/contact-fwddyn.hpp>
#include <crocoddyl/multibody/actions/impulse-fwddyn.hpp>
#include <crocoddyl/multibody/actuations/floating-base.hpp>
#include <crocoddyl/multibody/impulses/impulse-3d.hpp>
#include <crocoddyl/multibody/contacts/contact-3d.hpp>
#include <crocoddyl/multibody/costs/frame-translation.hpp>
#include <crocoddyl/multibody/costs/frame-velocity.hpp>
#include <crocoddyl/multibody/costs/com-position.hpp>
#include <crocoddyl/multibody/costs/state.hpp>
#include <crocoddyl/multibody/costs/control.hpp>
#include <crocoddyl/core/activations/weighted-quadratic.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/optctrl/shooting.hpp>

#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/frame.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace crocoddyl {

class SimpleQuadrupedGaitProblem {
 public:
  SimpleQuadrupedGaitProblem(const pinocchio::Model& rmodel, const std::string& lfFoot, const std::string& rfFoot,
                             const std::string& lhFoot, const std::string& rhFoot);
  ~SimpleQuadrupedGaitProblem();

  ShootingProblem createWalkingProblem(const Eigen::VectorXd& x0, const double stepLength, const double stepHeight,
                                       const double timeStep, const std::size_t stepKnots,
                                       const std::size_t supportKnots);

  void createFootStepModels(double timeStep, Eigen::Vector3d& comPos0, std::vector<Eigen::Vector3d>& feetPos0,
                            double stepLength, double stepHeight, std::size_t numKnots,
                            const std::vector<pinocchio::FrameIndex>& supportFootIds,
                            const std::vector<pinocchio::FrameIndex>& swingFootIds,
                            std::vector<ActionModelAbstract*>& actionModelList);

  void createSwingFootModel(double timeStep, const std::vector<pinocchio::FrameIndex>& supportFootIds,
                            const Eigen::Vector3d& comTask, const std::vector<FramePlacement>& swingFootTask,
                            std::vector<ActionModelAbstract*>& actionModelList);

  void createFootSwitchModel(const std::vector<pinocchio::FrameIndex>& supportFootIds,
                             const std::vector<FramePlacement>& swingFootTask, bool pseudoImpulse,
                             std::vector<ActionModelAbstract*>& actionModelList);

  void createPseudoImpulseModel(const std::vector<pinocchio::FrameIndex>& supportFootIds,
                                const std::vector<FramePlacement>& swingFootTask,
                                std::vector<ActionModelAbstract*>& actionModelList);

  void createImpulseModel(const std::vector<pinocchio::FrameIndex>& supportFootIds,
                          const std::vector<FramePlacement>& swingFootTask,
                          std::vector<ActionModelAbstract*>& actionModelList);

  const Eigen::VectorXd& get_defaultState() const { return defaultState_; };

 protected:
  pinocchio::Model rmodel_;
  pinocchio::Data rdata_;
  pinocchio::FrameIndex lfFootId_, rfFootId_, lhFootId_, rhFootId_;
  StateMultibody state_;
  ActuationModelFloatingBase actuation_;
  bool firstStep_;
  Eigen::VectorXd defaultState_;
};
}  // namespace crocoddyl
