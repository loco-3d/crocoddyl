#include "crocoddyl/multibody/utils/biped-gaits.hpp"

crocoddyl::SimpleBipedGaitProblem::SimpleBipedGaitProblem(
    pinocchio::Model& rmodel, std::string left_foot, std::string right_foot,
    int num_steps, const CostWeight& cost_weight)
    : rmodel_(rmodel),
      rdata_(rmodel_),
      left_foot_id_(rmodel_.getFrameId(
          left_foot,
          (pinocchio::FrameType)(pinocchio::JOINT | pinocchio::FIXED_JOINT |
                                 pinocchio::BODY))),
      right_foot_id_(rmodel_.getFrameId(
          right_foot,
          (pinocchio::FrameType)(pinocchio::JOINT | pinocchio::FIXED_JOINT |
                                 pinocchio::BODY))),
      state_(std::make_shared<crocoddyl::StateMultibody>(
          std::make_shared<pinocchio::Model>(rmodel_))),
      actuation_(
          std::make_shared<crocoddyl::ActuationModelFloatingBase>(state_)),
      num_steps_(num_steps),
      cost_weight_(cost_weight) {
  if (cost_weight_.x_weights.size() != 2 * rmodel_.nv) {
    cost_weight_.x_weights = Eigen::VectorXd::Ones(2 * rmodel_.nv);
  }
}

crocoddyl::SimpleBipedGaitProblem::SimpleBipedGaitProblem(
    pinocchio::Model& rmodel, std::string left_foot, std::string right_foot,
    int num_steps)
    : SimpleBipedGaitProblem(rmodel, left_foot, right_foot, num_steps,
                             CostWeight{}) {}

std::shared_ptr<crocoddyl::ShootingProblem>
crocoddyl::SimpleBipedGaitProblem::createWalkingProblem(
    const Eigen::VectorXd& x0, const double stepLength, const double stepHeight,
    const double timeStep, const std::size_t stepKnots,
    const std::size_t supportKnots, bool fwddyn) {
  Eigen::VectorXd q0 = x0.head(rmodel_.nq);
  q0_ = q0;

  this->fwddyn_ = fwddyn;

  pinocchio::forwardKinematics(rmodel_, rdata_, q0);
  pinocchio::updateFramePlacements(rmodel_, rdata_);

  Eigen::Vector3d rfPos0 = rdata_.oMf[right_foot_id_].translation();
  Eigen::Vector3d lfPos0 = rdata_.oMf[left_foot_id_].translation();
  std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>> feetPos0;
  feetPos0.push_back(std::make_pair(right_foot_id_, rfPos0));
  feetPos0.push_back(std::make_pair(left_foot_id_, lfPos0));
  Eigen::Vector3d comRef = (rfPos0 + lfPos0) / 2.0;
  comRef[2] = pinocchio::centerOfMass(rmodel_, rdata_, q0)[2];

  std::vector<pinocchio::FrameIndex> lf_ids;
  lf_ids.push_back(left_foot_id_);
  std::vector<pinocchio::FrameIndex> rf_ids;
  rf_ids.push_back(right_foot_id_);

  std::vector<pinocchio::FrameIndex> rf_lf_ids;
  rf_lf_ids.push_back(right_foot_id_);
  rf_lf_ids.push_back(left_foot_id_);

  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> loco3dModel;

  // initial double support phase
  for (size_t i = 0; i < supportKnots; i++)
    loco3dModel.push_back(createSwingFootModel(timeStep, rf_lf_ids, comRef));

  // walking steps
  for (int i = 1; i <= num_steps_; i++) {
    if (i % 4 == 1)  // right step
    {
      std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>
          rightStepModels;
      if (first_step_) {
        rightStepModels =
            createFootStepModels(timeStep, comRef, feetPos0, 0.5 * stepLength,
                                 stepHeight, stepKnots, lf_ids, rf_ids);
        first_step_ = false;
      } else {
        rightStepModels =
            createFootStepModels(timeStep, comRef, feetPos0, stepLength,
                                 stepHeight, stepKnots, lf_ids, rf_ids);
      }
      loco3dModel.insert(loco3dModel.end(), rightStepModels.begin(),
                         rightStepModels.end());
    } else if (i % 4 == 3)  // left step
    {
      std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>
          leftStepModels =
              createFootStepModels(timeStep, comRef, feetPos0, stepLength,
                                   stepHeight, stepKnots, rf_ids, lf_ids);
      loco3dModel.insert(loco3dModel.end(), leftStepModels.begin(),
                         leftStepModels.end());
    } else if (i % 2 == 0)  // double support
    {
      for (size_t j = 0; j < supportKnots; j++) {
        loco3dModel.push_back(
            createSwingFootModel(timeStep, rf_lf_ids, comRef, feetPos0));
      }
    }
  }

  // final double support phase
  if (num_steps_ % 2 == 1) {
    for (size_t i = 0; i < supportKnots; i++)
      loco3dModel.push_back(
          createSwingFootModel(timeStep, rf_lf_ids, comRef, feetPos0));
  }

  // terminal state
  std::shared_ptr<crocoddyl::ActionModelAbstract> terminalModel =
      loco3dModel.back();
  loco3dModel.pop_back();

  return std::make_shared<crocoddyl::ShootingProblem>(x0, loco3dModel,
                                                      terminalModel);
}

std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>
crocoddyl::SimpleBipedGaitProblem::createFootStepModels(
    double timeStep, Eigen::Vector3d& comPos0,
    std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>>& feetPos0,
    const double stepLength, const double stepHeight,
    const std::size_t numKnots,
    const std::vector<pinocchio::FrameIndex>& supportFootIds,
    const std::vector<pinocchio::FrameIndex>& swingFootIds) {
  int num_legs = (int)supportFootIds.size() + (int)swingFootIds.size();
  double com_percentage = (double)swingFootIds.size() / (double)num_legs;

  // action models for the foot swing
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> footStepModels;
  std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>> footTask;
  for (size_t k = 0; k < numKnots; k++)  // assume numKnots is odd
  {
    footTask.clear();
    for (size_t i = 0; i < swingFootIds.size(); i++)  // i: swing foot index
    {
      Eigen::Vector3d dp = Eigen::Vector3d::Zero();
      if (k < (numKnots - 1) / 2) {
        dp = Eigen::Vector3d(
            stepLength * ((double)k + 1) / (double)numKnots, 0.0,
            stepHeight * (double)k / ((double)(numKnots - 1) / 2.0));
      } else if (k == (numKnots - 1) / 2) {
        dp = Eigen::Vector3d(stepLength * ((double)k + 1) / (double)numKnots,
                             0.0, stepHeight);
      } else {
        dp = Eigen::Vector3d(
            stepLength * ((double)k + 1) / (double)numKnots, 0.0,
            stepHeight * (1.0 - ((double)k - (double)(numKnots - 1) / 2.0) /
                                    ((double)(numKnots - 1) / 2.0)));
      }

      Eigen::Vector3d footPos = Eigen::Vector3d::Zero();
      for (size_t j = 0; j < feetPos0.size(); ++j) {
        if (feetPos0[j].first == swingFootIds[i])  // get only swing foot
        {
          footPos = feetPos0[j].second;
          break;
        }
      }
      footTask.push_back(std::make_pair(swingFootIds[i], footPos + dp));
    }
    Eigen::Vector3d comTask =
        Eigen::Vector3d(stepLength * ((double)(k + 1) / (double)numKnots), 0.0,
                        0.0) *
            com_percentage +
        comPos0;
    footStepModels.push_back(
        createSwingFootModel(timeStep, supportFootIds, comTask, footTask));
  }

  // action model for the foot switch
  footStepModels.push_back(createFootSwitchModel(supportFootIds, footTask));

  // updating the current foot position for next step
  comPos0 += Eigen::Vector3d(stepLength * com_percentage, 0., 0.);
  for (size_t i = 0; i < swingFootIds.size(); i++)
    for (size_t j = 0; j < feetPos0.size(); ++j) {
      if (feetPos0[j].first == swingFootIds[i]) {
        feetPos0[j].second += Eigen::Vector3d(stepLength, 0., 0.);
        break;
      }
    }

  return footStepModels;
}

std::shared_ptr<crocoddyl::ActionModelAbstract>
crocoddyl::SimpleBipedGaitProblem::createSwingFootModel(
    double timeStep, const std::vector<pinocchio::FrameIndex>& supportFootIds,
    const Eigen::Vector3d& comTask,
    const std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>>&
        swingFootTask) {
  size_t nu;
  if (this->fwddyn_)
    nu = actuation_->get_nu();
  else
    nu = state_->get_nv() + 6 * supportFootIds.size();

  // Creating a 6D multi-contact model at the supporting foot
  std::shared_ptr<crocoddyl::ContactModelMultiple> contactModel =
      std::make_shared<crocoddyl::ContactModelMultiple>(state_, nu);
  for (size_t i = 0; i < supportFootIds.size(); i++) {
    std::shared_ptr<crocoddyl::ContactModelAbstract> supportContactModel =
        std::make_shared<crocoddyl::ContactModel6D>(
            state_, supportFootIds[i], pinocchio::SE3::Identity(),
            pinocchio::LOCAL_WORLD_ALIGNED, nu, Eigen::Vector2d(0., 50.));
    contactModel->addContact(
        rmodel_.frames[supportFootIds[i]].name + "_contact",
        supportContactModel);
  }

  // creating the cost model for a contact phase
  std::shared_ptr<crocoddyl::CostModelSum> costModel =
      std::make_shared<crocoddyl::CostModelSum>(state_, nu);

  // Com tracking cost
  if (comTask != Eigen::Vector3d::Zero()) {
    std::shared_ptr<crocoddyl::ResidualModelAbstract> comResidual =
        std::make_shared<crocoddyl::ResidualModelCoMPosition>(state_, comTask,
                                                              nu);
    std::shared_ptr<crocoddyl::CostModelAbstract> comTrack =
        std::make_shared<crocoddyl::CostModelResidual>(state_, comResidual);
    costModel->addCost("comTrack", comTrack, cost_weight_.com_track_weight);
  }

  // supporting foot contact wrench cone cost
  for (size_t i = 0; i < supportFootIds.size(); ++i) {
    Eigen::Matrix3d Rsurf = Eigen::Matrix3d::Identity();
    crocoddyl::WrenchCone cone(Rsurf, 0.7, Eigen::Vector2d(0.1, 0.05));

    std::shared_ptr<crocoddyl::ResidualModelAbstract> wrench_residual =
        std::make_shared<crocoddyl::ResidualModelContactWrenchCone>(
            state_, supportFootIds[i], cone, nu, fwddyn_);

    std::shared_ptr<crocoddyl::ActivationModelAbstract> wrench_activation =
        std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(
            crocoddyl::ActivationBounds(cone.get_lb(), cone.get_ub()));

    std::shared_ptr<crocoddyl::CostModelAbstract> wrench_cone =
        std::make_shared<crocoddyl::CostModelResidual>(
            state_, wrench_activation, wrench_residual);

    costModel->addCost(rmodel_.frames[supportFootIds[i]].name + "_wrenchCone",
                       wrench_cone, cost_weight_.contact_wrench_weight);
  }

  // swing foot tracking cost
  if (!swingFootTask.empty()) {
    for (size_t i = 0; i < swingFootTask.size(); i++) {
      pinocchio::FrameIndex id = swingFootTask[i].first;
      Eigen::Vector3d posTask = swingFootTask[i].second;

      std::shared_ptr<crocoddyl::ResidualModelAbstract>
          foot_placement_residual =
              std::make_shared<crocoddyl::ResidualModelFramePlacement>(
                  state_, id,
                  pinocchio::SE3(Eigen::Matrix3d::Identity(), posTask), nu);

      std::shared_ptr<crocoddyl::CostModelAbstract> foot_track =
          std::make_shared<crocoddyl::CostModelResidual>(
              state_, foot_placement_residual);

      costModel->addCost(rmodel_.frames[id].name + "_footTrack", foot_track,
                         cost_weight_.foot_track_weight);
    }
  }

  // state regularization cost
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(rmodel_.nq + rmodel_.nv);
  x0 << q0_, Eigen::VectorXd::Zero(rmodel_.nv);
  std::shared_ptr<crocoddyl::ResidualModelAbstract> stateResidual =
      std::make_shared<crocoddyl::ResidualModelState>(state_, x0, nu);
  std::shared_ptr<crocoddyl::ActivationModelAbstract> stateActivation =
      std::make_shared<crocoddyl::ActivationModelWeightedQuad>(
          cost_weight_.x_weights);
  std::shared_ptr<crocoddyl::CostModelAbstract> stateReg =
      std::make_shared<crocoddyl::CostModelResidual>(state_, stateActivation,
                                                     stateResidual);

  // control regularization cost
  std::shared_ptr<crocoddyl::CostModelAbstract> ctrlReg;
  if (this->fwddyn_) {
    std::shared_ptr<crocoddyl::ResidualModelAbstract> ctrlResidual =
        std::make_shared<crocoddyl::ResidualModelControl>(state_, nu);
    ctrlReg =
        std::make_shared<crocoddyl::CostModelResidual>(state_, ctrlResidual);
  } else {
    std::shared_ptr<crocoddyl::ResidualModelAbstract> ctrlResidual =
        std::make_shared<crocoddyl::ResidualModelJointEffort>(state_,
                                                              actuation_, nu);
    ctrlReg =
        std::make_shared<crocoddyl::CostModelResidual>(state_, ctrlResidual);
  }

  costModel->addCost("stateReg", stateReg, cost_weight_.state_weight);
  costModel->addCost("ctrlReg", ctrlReg, cost_weight_.control_weight);

  // Creating the action model for the KKT dynamics with symplectic Euler
  // integration scheme
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> dmodel;
  if (this->fwddyn_) {
    dmodel =
        std::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(
            state_, actuation_, contactModel, costModel, 0.0, true);
  } else {
    dmodel =
        std::make_shared<crocoddyl::DifferentialActionModelContactInvDynamics>(
            state_, actuation_, contactModel, costModel);
  }

  return std::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodel,
                                                                 timeStep);
}

std::shared_ptr<crocoddyl::ActionModelAbstract>
crocoddyl::SimpleBipedGaitProblem::createFootSwitchModel(
    const std::vector<pinocchio::FrameIndex>& supportFootIds,
    const std::vector<std::pair<pinocchio::FrameIndex, Eigen::Vector3d>>&
        swingFootTask) {
  size_t nu;
  if (this->fwddyn_) {
    nu = actuation_->get_nu();
  } else {
    nu = state_->get_nv() + 6 * supportFootIds.size();
  }

  // Creating a 6D multi-contact model at the supporting foot
  std::shared_ptr<crocoddyl::ContactModelMultiple> contactModel =
      std::make_shared<crocoddyl::ContactModelMultiple>(state_, nu);
  for (size_t i = 0; i < supportFootIds.size(); i++) {
    std::shared_ptr<crocoddyl::ContactModelAbstract> supportContactModel =
        std::make_shared<crocoddyl::ContactModel6D>(
            state_, supportFootIds[i], pinocchio::SE3::Identity(),
            pinocchio::LOCAL_WORLD_ALIGNED, nu, Eigen::Vector2d(0., 50.));
    contactModel->addContact(
        rmodel_.frames[supportFootIds[i]].name + "_contact",
        supportContactModel);
  }

  // Creating the cost model for a contact phase
  std::shared_ptr<crocoddyl::CostModelSum> costModel =
      std::make_shared<crocoddyl::CostModelSum>(state_, nu);
  for (size_t i = 0; i < supportFootIds.size(); ++i) {
    Eigen::Matrix3d Rsurf = Eigen::Matrix3d::Identity();
    crocoddyl::WrenchCone cone(Rsurf, 0.7, Eigen::Vector2d(0.1, 0.05));

    std::shared_ptr<crocoddyl::ResidualModelAbstract> wrench_residual =
        std::make_shared<crocoddyl::ResidualModelContactWrenchCone>(
            state_, supportFootIds[i], cone, nu, fwddyn_);

    std::shared_ptr<crocoddyl::ActivationModelAbstract> wrench_activation =
        std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(
            crocoddyl::ActivationBounds(cone.get_lb(), cone.get_ub()));

    std::shared_ptr<crocoddyl::CostModelAbstract> wrench_cone =
        std::make_shared<crocoddyl::CostModelResidual>(
            state_, wrench_activation, wrench_residual);

    costModel->addCost(rmodel_.frames[supportFootIds[i]].name + "_wrenchCone",
                       wrench_cone, cost_weight_.contact_wrench_weight);
  }

  // swing foot tracking cost
  for (size_t i = 0; i < swingFootTask.size(); i++) {
    pinocchio::FrameIndex id = swingFootTask[i].first;
    Eigen::Vector3d posTask = swingFootTask[i].second;

    std::shared_ptr<crocoddyl::ResidualModelAbstract> foot_placement_residual =
        std::make_shared<crocoddyl::ResidualModelFramePlacement>(
            state_, id, pinocchio::SE3(Eigen::Matrix3d::Identity(), posTask),
            nu);

    std::shared_ptr<crocoddyl::CostModelAbstract> foot_track =
        std::make_shared<crocoddyl::CostModelResidual>(state_,
                                                       foot_placement_residual);

    costModel->addCost(rmodel_.frames[id].name + "_footTrack", foot_track,
                       cost_weight_.foot_track_weight);

    std::shared_ptr<crocoddyl::ResidualModelAbstract>
        impulse_foot_vel_residual =
            std::make_shared<crocoddyl::ResidualModelFrameVelocity>(
                state_, id, pinocchio::Motion::Zero(),
                pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, nu);
    std::shared_ptr<CostModelAbstract> impulseFootVelCost =
        std::make_shared<crocoddyl::CostModelResidual>(
            state_, impulse_foot_vel_residual);

    costModel->addCost(rmodel_.frames[id].name + "_impulseFootVel",
                       impulseFootVelCost,
                       cost_weight_.impulse_foot_vel_weight);
  }

  // state regularization cost
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(rmodel_.nq + rmodel_.nv);
  x0 << q0_, Eigen::VectorXd::Zero(rmodel_.nv);
  std::shared_ptr<crocoddyl::ResidualModelAbstract> stateResidual =
      std::make_shared<crocoddyl::ResidualModelState>(state_, x0, nu);
  std::shared_ptr<crocoddyl::ActivationModelAbstract> stateActivation =
      std::make_shared<crocoddyl::ActivationModelWeightedQuad>(
          cost_weight_.x_weights);
  std::shared_ptr<crocoddyl::CostModelAbstract> stateReg =
      std::make_shared<crocoddyl::CostModelResidual>(state_, stateActivation,
                                                     stateResidual);

  // control regularization cost
  std::shared_ptr<crocoddyl::CostModelAbstract> ctrlReg;
  if (this->fwddyn_) {
    std::shared_ptr<crocoddyl::ResidualModelAbstract> ctrlResidual =
        std::make_shared<crocoddyl::ResidualModelControl>(state_, nu);
    ctrlReg =
        std::make_shared<crocoddyl::CostModelResidual>(state_, ctrlResidual);
  } else {
    std::shared_ptr<crocoddyl::ResidualModelAbstract> ctrlResidual =
        std::make_shared<crocoddyl::ResidualModelJointEffort>(state_,
                                                              actuation_, nu);
    ctrlReg =
        std::make_shared<crocoddyl::CostModelResidual>(state_, ctrlResidual);
  }

  costModel->addCost("stateReg", stateReg, cost_weight_.state_weight);
  costModel->addCost("ctrlReg", ctrlReg, cost_weight_.control_weight);

  // Creating the action model for the KKT dynamics with symplectic Euler
  // integration scheme
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> dmodel;
  if (this->fwddyn_) {
    dmodel =
        std::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(
            state_, actuation_, contactModel, costModel, 0.0, true);
  } else {
    dmodel =
        std::make_shared<crocoddyl::DifferentialActionModelContactInvDynamics>(
            state_, actuation_, contactModel, costModel);
  }

  return std::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodel, 0.0);
}
