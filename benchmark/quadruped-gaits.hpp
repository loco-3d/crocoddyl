#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <cmath>
#include <limits>
#include <Eigen/Core>

#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/activations/weighted-quadratic.hpp>
#include <crocoddyl/core/integrator/euler.hpp>

#include <crocoddyl/multibody/actuations/floating-base.hpp>
#include <crocoddyl/multibody/frames.hpp>
#include <crocoddyl/multibody/actions/contact-fwddyn.hpp>
#include <crocoddyl/multibody/impulses/multiple-impulses.hpp>
#include <crocoddyl/multibody/impulses/multiple-impulses.hpp>
#include <crocoddyl/multibody/impulses/impulse-3d.hpp>
#include <crocoddyl/multibody/actions/impulse-fwddyn.hpp>

#include <crocoddyl/multibody/costs/frame-placement.hpp>
#include <crocoddyl/multibody/costs/frame-translation.hpp>
#include <crocoddyl/multibody/costs/frame-velocity.hpp>
#include <crocoddyl/multibody/costs/com-position.hpp>
#include <crocoddyl/multibody/costs/state.hpp>
#include <crocoddyl/multibody/costs/control.hpp>
#include <crocoddyl/multibody/costs/cost-sum.hpp>
#include <crocoddyl/multibody/contacts/multiple-contacts.hpp>
#include <crocoddyl/multibody/contacts/contact-3d.hpp>

#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/frame.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace crocoddyl {
  class SimpleQuadrupedGaitProblem {
    
  public:
    SimpleQuadrupedGaitProblem(const pinocchio::Model& rmodel_, const std::string& lfFoot_,
                               const std::string& rfFoot_, const std::string& lhFoot_,
                               const std::string& rhFoot_);
    
    void createFootStepModels(double timeStep,
                              Eigen::Vector3d& comPos0,
                              std::vector<Eigen::Vector3d>& feetPos0,
                              double stepLength, double stepHeight,
                              unsigned int numKnots,
                              const std::vector<pinocchio::FrameIndex>& supportFootIds,
                              const std::vector<pinocchio::FrameIndex>& swingFootIds,
                              std::vector<ActionModelAbstract*>& actionModelList);
    
    void createSwingFootModel(double timeStep,
                              const std::vector<pinocchio::FrameIndex>& supportFootIds,
                              const Eigen::Vector3d& comTask,
                              const std::vector<crocoddyl::FramePlacement>& swingFootTask,
                              std::vector<ActionModelAbstract*>& actionModelList);
    
    void createFootSwitchModel(const std::vector<pinocchio::FrameIndex>& supportFootIds,
                               const std::vector<crocoddyl::FramePlacement>& swingFootTask,
                               bool pseudoImpulse,
                               std::vector<ActionModelAbstract*>& actionModelList);
    
    void createPseudoImpulseModel(const std::vector<pinocchio::FrameIndex>& supportFootIds,
                                  const std::vector<crocoddyl::FramePlacement>&
                                  swingFootTask,
                                  std::vector<ActionModelAbstract*>& actionModelList);
    
    void createImpulseModel(const std::vector<pinocchio::FrameIndex>& supportFootIds,
                            const std::vector<crocoddyl::FramePlacement>& swingFootTask,
                            std::vector<ActionModelAbstract*>& actionModelList);
    
    ShootingProblem createWalkingProblem(const Eigen::VectorXd& x0,
                                         const double stepLength, const double stepHeight,
                                         const double timeStep, const unsigned int stepKnots,
                                         const unsigned int supportKnots);

    Eigen::VectorXd& get_defaultState() const { return defaultState; };
    
  protected:
    pinocchio::Model rmodel;
    pinocchio::Data rdata;
    pinocchio::FrameIndex lfFootId, rfFootId, lhFootId, rhFootId;
    crocoddyl::StateMultibody state;
    crocoddyl::ActuationModelFloatingBase actuation;
    std::vector<ActionModelAbstract*> runningModels;
    ActionModelAbstract* terminalModel;
    bool firstStep;
    Eigen::VectorXd defaultState;
  };
}
