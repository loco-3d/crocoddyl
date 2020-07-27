#include <pinocchio/gepetto/viewer.hpp>
#include "factory/biped.hpp"
#include <example-robot-data/path.hpp>

template <typename Scalar>
void build_biped_action_models(boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> >& runningModel,
                               boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> >& terminalModel) {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector2s Vector2s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector3s Vector3s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Matrix3s Matrix3s;
  typedef typename crocoddyl::FramePlacementTpl<Scalar> FramePlacement;
  typedef typename crocoddyl::FrameTranslationTpl<Scalar> FrameTranslation;
  typedef typename crocoddyl::DifferentialActionModelContactFwdDynamicsTpl<Scalar>
      DifferentialActionModelContactFwdDynamics;
  typedef typename crocoddyl::IntegratedActionModelEulerTpl<Scalar> IntegratedActionModelEuler;
  typedef typename crocoddyl::ActuationModelFloatingBaseTpl<Scalar> ActuationModelFloatingBase;
  typedef typename crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef typename crocoddyl::CostModelAbstractTpl<Scalar> CostModelAbstract;
  typedef typename crocoddyl::CostModelFramePlacementTpl<Scalar> CostModelFramePlacement;
  typedef typename crocoddyl::CostModelCoMPositionTpl<Scalar> CostModelCoMPosition;
  typedef typename crocoddyl::CostModelStateTpl<Scalar> CostModelState;
  typedef typename crocoddyl::CostModelControlTpl<Scalar> CostModelControl;

  const std::string RF = "leg_right_6_joint";
  const std::string LF = "leg_left_6_joint";

  pinocchio::ModelTpl<double> modeld;
  pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_reduced.urdf",
                              pinocchio::JointModelFreeFlyer(), modeld);
  modeld.lowerPositionLimit.head<7>().array() = -1;
  modeld.upperPositionLimit.head<7>().array() = 1.;
  pinocchio::srdf::loadReferenceConfigurations(modeld, EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf",
                                               false);

  pinocchio::ModelTpl<Scalar> model(modeld.cast<Scalar>());
  boost::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > state =
      boost::make_shared<crocoddyl::StateMultibodyTpl<Scalar> >(
          boost::make_shared<pinocchio::ModelTpl<Scalar> >(model));

  boost::shared_ptr<ActuationModelFloatingBase> actuation = boost::make_shared<ActuationModelFloatingBase>(state);

  FramePlacement Mref(model.getFrameId("arm_right_7_joint"),
                      pinocchio::SE3Tpl<Scalar>(Matrix3s::Identity(), Vector3s(Scalar(.0), Scalar(.0), Scalar(.4))));

  boost::shared_ptr<CostModelAbstract> comCost =
      boost::make_shared<CostModelCoMPosition>(state, Vector3s::Zero(), actuation->get_nu());
  boost::shared_ptr<CostModelAbstract> goalTrackingCost =
      boost::make_shared<CostModelFramePlacement>(state, Mref, actuation->get_nu());
  boost::shared_ptr<CostModelAbstract> xRegCost = boost::make_shared<CostModelState>(state, actuation->get_nu());
  boost::shared_ptr<CostModelAbstract> uRegCost = boost::make_shared<CostModelControl>(state, actuation->get_nu());

  // Create a cost model per the running and terminal action model.
  boost::shared_ptr<CostModelSum> runningCostModel = boost::make_shared<CostModelSum>(state, actuation->get_nu());
  boost::shared_ptr<CostModelSum> terminalCostModel = boost::make_shared<CostModelSum>(state, actuation->get_nu());

  // Then let's added the running and terminal cost functions
  runningCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));
  //   runningCostModel->addCost("comPos", comCost, Scalar(1e-7));
  runningCostModel->addCost("xReg", xRegCost, Scalar(1e-4));
  runningCostModel->addCost("uReg", uRegCost, Scalar(1e-4));
  terminalCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));


  //////////////////////////////////////////////////////////////////////////////////
  typedef pinocchio::RigidContactModelTpl<Scalar,0> RigidContactModel;
  typedef pinocchio::RigidContactDataTpl<Scalar,0> RigidContactData;
  
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactModel) contact_models;
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactData) contact_data;

  RigidContactModel ci_RF(pinocchio::CONTACT_6D,model.getFrameId(RF),pinocchio::LOCAL);
  RigidContactModel ci_LF(pinocchio::CONTACT_3D,model.getFrameId(RF),pinocchio::LOCAL);

  contact_models.push_back(ci_RF); contact_data.push_back(RigidContactData(ci_RF));
  contact_models.push_back(ci_LF); contact_data.push_back(RigidContactData(ci_LF));

  ///////////////////////////////////////////////////////////////////////////////

  // Next, we need to create an action model for running and terminal nodes
  boost::shared_ptr<DifferentialActionModelContactFwdDynamics> runningDAM =
      boost::make_shared<DifferentialActionModelContactFwdDynamics>(state, actuation, contact_models,
                                                                    runningCostModel);
  boost::shared_ptr<DifferentialActionModelContactFwdDynamics> terminalDAM =
      boost::make_shared<DifferentialActionModelContactFwdDynamics>(state, actuation, contact_models,
                                                                    terminalCostModel);

  runningModel = boost::make_shared<IntegratedActionModelEuler>(runningDAM, Scalar(1e-3));
  terminalModel = boost::make_shared<IntegratedActionModelEuler>(terminalDAM, Scalar(1e-3));
}


boost::shared_ptr<crocoddyl::SolverDDP>
HandTrackingProblemFull::create_tracking_problem(const Eigen::VectorXd& x_init,
                                                 const std::size_t& horizon_length,
                                                 const Eigen::VectorXd& running_costs,
                                                 const Eigen::VectorXd& terminal_costs,
                                                 const Eigen::VectorXd& state_pos_costs,
                                                 const Eigen::VectorXd& state_vel_costs,
                                                 const std::string reference_configuration,
                                                 const double& time_step) {
  if (x_init.size() != pin_model_.nq+pin_model_.nv){
    throw_pretty("size of x_init should be "+ std::to_string(pin_model_.nq+pin_model_.nv));
  }
  if(state_pos_costs.size() != pin_model_.nv) {
    throw_pretty("state_pos_costs must be of size nv.");
  }
  if(state_vel_costs.size() != pin_model_.nv) {
    throw_pretty("state_vel_costs  must be of size nv.");
  }
  
  boost::shared_ptr<StateMultibody> state =
    boost::make_shared<StateMultibody>(boost::make_shared<pinocchio::Model>(pin_model_));
  boost::shared_ptr<ActuationModel> actuation =
      boost::make_shared<ActuationModel>(state);
      
      
  Eigen::VectorXd xlb(2 * pin_model_.nv), xub(2 * pin_model_.nv);
  
  xlb << Eigen::VectorXd::Ones(6) * (-std::numeric_limits<double>::max()),
         pin_model_.lowerPositionLimit.tail(pin_model_.nq - 7),
         Eigen::VectorXd::Ones(pin_model_.nv) * (-std::numeric_limits<double>::max());
  xub << Eigen::VectorXd::Ones(6) * (std::numeric_limits<double>::max()),
         pin_model_.upperPositionLimit.tail(pin_model_.nq - 7),
         Eigen::VectorXd::Ones(pin_model_.nv) * (std::numeric_limits<double>::max());
  crocoddyl::ActivationBounds bounds = crocoddyl::ActivationBounds(xlb, xub);
  
  
  //Pinocchio Reference configuration taken to be "half_sitting"
  const Eigen::Vector3d& comref = pinocchio::centerOfMass(pin_model_, pin_data_,
                                                          pin_model_.referenceConfigurations[reference_configuration], false);
  pinocchio::framesForwardKinematics(pin_model_, pin_data_, pin_model_.referenceConfigurations[reference_configuration]);
  
  //Create Contact models
  boost::shared_ptr<ContactModelMultiple> contact_model =
    boost::make_shared<ContactModelMultiple>(state, actuation->get_nu());
  for (ContactFrames::const_iterator it = contact_ids_.begin();
       it != contact_ids_.end(); ++it) {
    const pinocchio::SE3& contact_frame_pos = pin_data_.oMf[*it];
    FramePlacement xref(*it, contact_frame_pos);
    boost::shared_ptr<ContactModelAbstract> single_contact_model =
      boost::make_shared<ContactModel6D>(state, xref, actuation->get_nu(), Eigen::Vector2d(0., 4.));
    contact_model->addContact(pin_model_.frames[*it].name + "_contact",
                              single_contact_model);
  }
  
  //Create cost models
  goalTrackingCost_ =
    boost::make_shared<CostModelFrameTranslation>(state, crocoddyl::FrameTranslation(frame_ids_, target_pos_), actuation->get_nu());
  
  //xWeights.head(state->get_nq()).array() = 0.01 * 0.01;
  //xWeights.tail(state->get_nv()).array() = 10 * 10;
  Eigen::VectorXd xWeights(state->get_ndx());
  xWeights << state_pos_costs,state_vel_costs;
  boost::shared_ptr<ActivationModelWeightedQuad> actxWeights = 
    boost::make_shared<ActivationModelWeightedQuad>(xWeights);
  Eigen::VectorXd default_state(pin_model_.nq+pin_model_.nv);
  
  default_state << pin_model_.referenceConfigurations[reference_configuration], Eigen::VectorXd::Zero(pin_model_.nv);

  boost::shared_ptr<CostModelAbstract> xRegCost =
    boost::make_shared<CostModelState>(state, actxWeights,default_state, actuation->get_nu());
  boost::shared_ptr<CostModelAbstract> uRegCost =
    boost::make_shared<CostModelControl>(state, actuation->get_nu());

  boost::shared_ptr<ActivationModelQuadraticBarrier> activation_bounded = 
    boost::make_shared<ActivationModelQuadraticBarrier>(bounds);

  boost::shared_ptr<CostModelAbstract> jointLimitCost =
    boost::make_shared<CostModelState>(state, activation_bounded, actuation->get_nu());

  boost::shared_ptr<CostModelAbstract> comTrack =
    boost::make_shared<CostModelCoMPosition>(state, comref, actuation->get_nu());  

  boost::shared_ptr<CostModelSum> runningCostModel =
    boost::make_shared<CostModelSum>(state, actuation->get_nu());
  boost::shared_ptr<CostModelSum> terminalCostModel =
    boost::make_shared<CostModelSum>(state, actuation->get_nu());
  
  //Add costs
  runningCostModel.get()->addCost("gripperPose", goalTrackingCost_, running_costs(0));
  runningCostModel.get()->addCost("stateReg", xRegCost, running_costs(1));
  runningCostModel.get()->addCost("ctrlReg", uRegCost, running_costs(2));
  runningCostModel.get()->addCost("limitCost", jointLimitCost, running_costs(3));
  
  terminalCostModel.get()->addCost("gripperPose", goalTrackingCost_, terminal_costs(0));
  terminalCostModel.get()->addCost("stateReg", xRegCost, terminal_costs(1));
  terminalCostModel.get()->addCost("limitCost", jointLimitCost, terminal_costs(2));

  //Create Contact Models:
  
  
  //Create Running models
  boost::shared_ptr<DAModel> runningDAM =
    boost::make_shared<DAModel>(state, actuation, contact_model, runningCostModel);
  boost::shared_ptr<ActionModelAbstract> runningModel =
    boost::make_shared<IAModel>(runningDAM, time_step);
  
  //Create Terminal Model
  boost::shared_ptr<DAModel> terminalDAM =
    boost::make_shared<DAModel>(state, actuation, contact_model, terminalCostModel);
  boost::shared_ptr<ActionModelAbstract> terminalModel =
    boost::make_shared<IAModel>(terminalDAM, time_step);
  
  //Create Shooting Problem
  std::vector<boost::shared_ptr<ActionModelAbstract> >
    runningModels(horizon_length, runningModel);
  shooting_problem =
    boost::make_shared<ShootingProblem>(x_init, runningModels, terminalModel);
  
  //Create DDP problem
  ddp = boost::make_shared<SolverDDP>(shooting_problem);
  return ddp;
}

void HandTrackingProblemCG::reset_ddp_data(const Eigen::VectorXd& x0){
  shooting_problem->set_x0(x0);
  ddp->allocateData();
  return;
}


int main(int argc, char* argv[]) {

  unsigned int N = 100;  // number of nodes
  unsigned int T = 1e3;  // number of trials
  if (argc > 1) {
    T = atoi(argv[1]);
  }


  //Create runningModel
  boost::shared_ptr<crocoddyl::ActionModelAbstract> runningModel, terminalModel;
  crocoddyl::benchmark::build_biped_action_models(runningModel, terminalModel);
