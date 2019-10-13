#include "quadruped-gaits.hpp"

namespace crocoddyl {

SimpleQuadrupedGaitProblem::SimpleQuadrupedGaitProblem(const pinocchio::Model& rmodel, const std::string& lfFoot,
                                                       const std::string& rfFoot, const std::string& lhFoot,
                                                       const std::string& rhFoot)
    : rmodel_(rmodel),
      rdata_(rmodel_),
      lfFootId_(rmodel_.getFrameId(
          lfFoot, (pinocchio::FrameType)(pinocchio::JOINT | pinocchio::FIXED_JOINT | pinocchio::BODY))),
      rfFootId_(rmodel_.getFrameId(
          rfFoot, (pinocchio::FrameType)(pinocchio::JOINT | pinocchio::FIXED_JOINT | pinocchio::BODY))),
      lhFootId_(rmodel_.getFrameId(
          lhFoot, (pinocchio::FrameType)(pinocchio::JOINT | pinocchio::FIXED_JOINT | pinocchio::BODY))),
      rhFootId_(rmodel_.getFrameId(
          rhFoot, (pinocchio::FrameType)(pinocchio::JOINT | pinocchio::FIXED_JOINT | pinocchio::BODY))),
      state_(boost::make_shared<crocoddyl::StateMultibody>(boost::ref(rmodel_))),
      actuation_(boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state_)),
      firstStep_(true),
      defaultState_(rmodel_.nq + rmodel_.nv) {
  defaultState_.head(rmodel_.nq) = rmodel_.referenceConfigurations["half_sitting"];
  defaultState_.tail(rmodel_.nv).setZero();
}

SimpleQuadrupedGaitProblem::~SimpleQuadrupedGaitProblem() {}

boost::shared_ptr<crocoddyl::ShootingProblem> SimpleQuadrupedGaitProblem::createWalkingProblem(
    const Eigen::VectorXd& x0, const double stepLength, const double stepHeight, const double timeStep,
    const std::size_t stepKnots, const std::size_t supportKnots) {
  int nq = rmodel_.nq;

  // Initial Condition
  const Eigen::VectorBlock<const Eigen::VectorXd> q0 = x0.head(nq);
  pinocchio::forwardKinematics(rmodel_, rdata_, q0);
  pinocchio::centerOfMass(rmodel_, rdata_, q0);
  pinocchio::updateFramePlacements(rmodel_, rdata_);

  const pinocchio::SE3::Vector3& rfFootPos0 = rdata_.oMf[rfFootId_].translation();
  const pinocchio::SE3::Vector3& rhFootPos0 = rdata_.oMf[rhFootId_].translation();
  const pinocchio::SE3::Vector3& lfFootPos0 = rdata_.oMf[lfFootId_].translation();
  const pinocchio::SE3::Vector3& lhFootPos0 = rdata_.oMf[lhFootId_].translation();

  pinocchio::SE3::Vector3 comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4;
  comRef[2] = rdata_.com[0][2];

  // Defining the action models along the time instances
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > loco3dModel;
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > rhStep, rfStep, lhStep, lfStep;

  // doublesupport
  std::vector<pinocchio::FrameIndex> supportFeet;
  supportFeet.push_back(lfFootId_);
  supportFeet.push_back(rfFootId_);
  supportFeet.push_back(lhFootId_);
  supportFeet.push_back(rhFootId_);
  Eigen::Vector3d nullCoM = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  const std::vector<crocoddyl::FramePlacement> emptyVector;
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > doubleSupport(
      supportKnots, createSwingFootModel(timeStep, supportFeet, nullCoM, emptyVector));

  const pinocchio::FrameIndex arr_lf_rf_lh[] = {lfFootId_, rfFootId_, lhFootId_};
  const pinocchio::FrameIndex arr_lf_lh_rh[] = {lfFootId_, lhFootId_, rhFootId_};
  const pinocchio::FrameIndex arr_lf_rf_rh[] = {lfFootId_, rfFootId_, rhFootId_};
  const pinocchio::FrameIndex arr_rf_lh_rh[] = {rfFootId_, lhFootId_, rhFootId_};

  std::vector<pinocchio::FrameIndex> legs_lf_rf_lh(arr_lf_rf_lh,
                                                   arr_lf_rf_lh + sizeof(arr_lf_rf_lh) / sizeof(arr_lf_rf_lh[0]));
  std::vector<pinocchio::FrameIndex> legs_lf_lh_rh(arr_lf_lh_rh,
                                                   arr_lf_lh_rh + sizeof(arr_lf_lh_rh) / sizeof(arr_lf_lh_rh[0]));
  std::vector<pinocchio::FrameIndex> legs_lf_rf_rh(arr_lf_rf_rh,
                                                   arr_lf_rf_rh + sizeof(arr_lf_rf_rh) / sizeof(arr_lf_rf_rh[0]));
  std::vector<pinocchio::FrameIndex> legs_rf_lh_rh(arr_rf_lh_rh,
                                                   arr_rf_lh_rh + sizeof(arr_rf_lh_rh) / sizeof(arr_rf_lh_rh[0]));

  std::vector<pinocchio::FrameIndex> rhFoot(1, rhFootId_);
  std::vector<pinocchio::FrameIndex> rfFoot(1, rfFootId_);
  std::vector<pinocchio::FrameIndex> lfFoot(1, lfFootId_);
  std::vector<pinocchio::FrameIndex> lhFoot(1, lhFootId_);

  std::vector<Eigen::Vector3d> rhFootPos0vec(1, rhFootPos0);
  std::vector<Eigen::Vector3d> lhFootPos0vec(1, lhFootPos0);
  std::vector<Eigen::Vector3d> rfFootPos0vec(1, rfFootPos0);
  std::vector<Eigen::Vector3d> lfFootPos0vec(1, lfFootPos0);
  if (firstStep_) {
    rhStep = createFootStepModels(timeStep, comRef, rhFootPos0vec, 0.5 * stepLength, stepHeight, stepKnots,
                                  legs_lf_rf_lh, rhFoot);
    rfStep = createFootStepModels(timeStep, comRef, rfFootPos0vec, 0.5 * stepLength, stepHeight, stepKnots,
                                  legs_lf_lh_rh, rfFoot);
    firstStep_ = false;
  } else {
    rhStep = createFootStepModels(timeStep, comRef, rhFootPos0vec, stepLength, stepHeight, stepKnots, legs_lf_rf_lh,
                                  rhFoot);
    rfStep = createFootStepModels(timeStep, comRef, rfFootPos0vec, stepLength, stepHeight, stepKnots, legs_lf_lh_rh,
                                  rfFoot);
  }
  lhStep = createFootStepModels(timeStep, comRef, lhFootPos0vec, stepLength, stepHeight, stepKnots, legs_lf_rf_rh,
                                lhFoot);
  lfStep = createFootStepModels(timeStep, comRef, lfFootPos0vec, stepLength, stepHeight, stepKnots, legs_rf_lh_rh,
                                lfFoot);

  loco3dModel.insert(loco3dModel.end(), doubleSupport.begin(), doubleSupport.end());
  loco3dModel.insert(loco3dModel.end(), rhStep.begin(), rhStep.end());
  loco3dModel.insert(loco3dModel.end(), rfStep.begin(), rfStep.end());
  loco3dModel.insert(loco3dModel.end(), doubleSupport.begin(), doubleSupport.end());
  loco3dModel.insert(loco3dModel.end(), lhStep.begin(), lhStep.end());
  loco3dModel.insert(loco3dModel.end(), lfStep.begin(), lfStep.end());

  return boost::make_shared<crocoddyl::ShootingProblem>(x0, loco3dModel, loco3dModel.back());
}

std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > SimpleQuadrupedGaitProblem::createFootStepModels(
    double timeStep, Eigen::Vector3d& comPos0, std::vector<Eigen::Vector3d>& feetPos0, double stepLength,
    double stepHeight, std::size_t numKnots, const std::vector<pinocchio::FrameIndex>& supportFootIds,
    const std::vector<pinocchio::FrameIndex>& swingFootIds) {
  std::size_t numLegs = static_cast<std::size_t>(supportFootIds.size() + swingFootIds.size());
  double comPercentage = static_cast<double>(swingFootIds.size()) / static_cast<double>(numLegs);

  // Action models for the foot swing
  std::vector<boost::shared_ptr<ActionModelAbstract> > footSwingModel;
  std::vector<crocoddyl::FramePlacement> swingFootTask;
  for (std::size_t k = 0; k < numKnots; ++k) {
    double _kp1_n = 0;
    Eigen::Vector3d dp = Eigen::Vector3d::Zero();
    swingFootTask.clear();
    for (std::size_t i = 0; i < swingFootIds.size(); ++i) {
      // Defining a foot swing task given the step length resKnot = numKnots % 2
      std::size_t phKnots = numKnots >> 1;  // bitwise divide.
      _kp1_n = static_cast<double>(k + 1) / static_cast<double>(numKnots);
      double _k = static_cast<double>(k);
      double _phKnots = static_cast<double>(phKnots);
      if (k < phKnots)
        dp << stepLength * _kp1_n, 0., stepHeight * _k / _phKnots;
      else if (k == phKnots)
        dp << stepLength * _kp1_n, 0., stepHeight;
      else
        dp << stepLength * _kp1_n, 0., stepHeight * (1 - (_k - _phKnots) / _phKnots);
      Eigen::Vector3d tref = feetPos0[i] + dp;

      swingFootTask.push_back(
          crocoddyl::FramePlacement(swingFootIds[i], pinocchio::SE3(Eigen::Matrix3d::Identity(), tref)));
    }

    // Action model for the foot switch
    Eigen::Vector3d comTask = Eigen::Vector3d(stepLength * _kp1_n, 0., 0.) * comPercentage + comPos0;
    footSwingModel.push_back(createSwingFootModel(timeStep, supportFootIds, comTask, swingFootTask));
  }
  // Action model for the foot switch
  footSwingModel.push_back(createFootSwitchModel(supportFootIds, swingFootTask));

  // Updating the current foot position for next step
  comPos0 += Eigen::Vector3d(stepLength * comPercentage, 0., 0.);
  for (std::size_t i = 0; i < feetPos0.size(); ++i) {
    feetPos0[i] += Eigen::Vector3d(stepLength, 0., 0.);
  }
  return footSwingModel;
}

boost::shared_ptr<crocoddyl::ActionModelAbstract> SimpleQuadrupedGaitProblem::createSwingFootModel(
    double timeStep, const std::vector<pinocchio::FrameIndex>& supportFootIds, const Eigen::Vector3d& comTask,
    const std::vector<crocoddyl::FramePlacement>& swingFootTask) {
  // Creating a 3D multi-contact model, and then including the supporting foot
  boost::shared_ptr<crocoddyl::ContactModelMultiple> contactModel =
      boost::make_shared<crocoddyl::ContactModelMultiple>(state_, actuation_->get_nu());
  for (std::vector<pinocchio::FrameIndex>::const_iterator it = supportFootIds.begin(); it != supportFootIds.end();
       ++it) {
    crocoddyl::FrameTranslation xref(*it, Eigen::Vector3d::Zero());
    boost::shared_ptr<crocoddyl::ContactModelAbstract> supportContactModel =
        boost::make_shared<crocoddyl::ContactModel3D>(state_, xref, actuation_->get_nu(), Eigen::Vector2d(0., 50.));
    contactModel->addContact(rmodel_.frames[*it].name + "_contact", supportContactModel);
  }

  // Creating the cost model for a contact phase
  boost::shared_ptr<crocoddyl::CostModelSum> costModel =
      boost::make_shared<crocoddyl::CostModelSum>(state_, actuation_->get_nu());
  if (comTask.array().allFinite()) {
    boost::shared_ptr<crocoddyl::CostModelAbstract> comTrack =
        boost::make_shared<crocoddyl::CostModelCoMPosition>(state_, comTask, actuation_->get_nu());
    costModel->addCost("comTrack", comTrack, 1e4);
  }
  if (!swingFootTask.empty()) {
    for (std::vector<crocoddyl::FramePlacement>::const_iterator it = swingFootTask.begin(); it != swingFootTask.end();
         ++it) {
      crocoddyl::FrameTranslation xref(it->frame, it->oMf.translation());
      boost::shared_ptr<crocoddyl::CostModelAbstract> footTrack =
          boost::make_shared<crocoddyl::CostModelFrameTranslation>(state_, xref, actuation_->get_nu());
      costModel->addCost(rmodel_.frames[it->frame].name + "_footTrack", footTrack, 1e4);
    }
  }
  Eigen::VectorXd stateWeights(2 * rmodel_.nv);
  stateWeights.head<3>().fill(0.);
  stateWeights.segment<3>(3).fill(pow(500., 2));
  stateWeights.segment(6, rmodel_.nv - 6).fill(pow(0.01, 2));
  stateWeights.segment(rmodel_.nv, rmodel_.nv).fill(pow(10., 2));
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> stateActivation =
      boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(stateWeights);
  boost::shared_ptr<crocoddyl::CostModelAbstract> stateReg =
      boost::make_shared<crocoddyl::CostModelState>(state_, stateActivation, defaultState_, actuation_->get_nu());
  boost::shared_ptr<crocoddyl::CostModelAbstract> ctrlReg =
      boost::make_shared<crocoddyl::CostModelControl>(state_, actuation_->get_nu());
  costModel->addCost("stateReg", stateReg, 1e-1);
  costModel->addCost("ctrlReg", ctrlReg, 1e-4);

  // Creating the action model for the KKT dynamics with simpletic Euler integration scheme
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> dmodel =
      boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state_, actuation_, contactModel,
                                                                               costModel);
  return boost::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodel, timeStep);
}

boost::shared_ptr<ActionModelAbstract> SimpleQuadrupedGaitProblem::createFootSwitchModel(
    const std::vector<pinocchio::FrameIndex>& supportFootIds,
    const std::vector<crocoddyl::FramePlacement>& swingFootTask, bool pseudoImpulse) {
  if (pseudoImpulse) {
    return createPseudoImpulseModel(supportFootIds, swingFootTask);
  } else {
    return createImpulseModel(supportFootIds, swingFootTask);
  }
}

boost::shared_ptr<crocoddyl::ActionModelAbstract> SimpleQuadrupedGaitProblem::createPseudoImpulseModel(
    const std::vector<pinocchio::FrameIndex>& supportFootIds,
    const std::vector<crocoddyl::FramePlacement>& swingFootTask) {
  // Creating a 3D multi-contact model, and then including the supporting foot
  boost::shared_ptr<crocoddyl::ContactModelMultiple> contactModel =
      boost::make_shared<crocoddyl::ContactModelMultiple>(state_, actuation_->get_nu());
  for (std::vector<pinocchio::FrameIndex>::const_iterator it = supportFootIds.begin(); it != supportFootIds.end();
       ++it) {
    crocoddyl::FrameTranslation xref(*it, Eigen::Vector3d::Zero());
    boost::shared_ptr<crocoddyl::ContactModelAbstract> supportContactModel =
        boost::make_shared<crocoddyl::ContactModel3D>(state_, xref, actuation_->get_nu(), Eigen::Vector2d(0., 50.));
    contactModel->addContact(rmodel_.frames[*it].name + "_contact", supportContactModel);
  }

  // Creating the cost model for a contact phase
  boost::shared_ptr<crocoddyl::CostModelSum> costModel =
      boost::make_shared<crocoddyl::CostModelSum>(state_, actuation_->get_nu());
  if (!swingFootTask.empty()) {
    for (std::vector<crocoddyl::FramePlacement>::const_iterator it = swingFootTask.begin(); it != swingFootTask.end();
         ++it) {
      crocoddyl::FrameTranslation xref(it->frame, it->oMf.translation());
      crocoddyl::FrameMotion vref(it->frame, pinocchio::Motion::Zero());
      boost::shared_ptr<crocoddyl::CostModelAbstract> footTrack =
          boost::make_shared<crocoddyl::CostModelFrameTranslation>(state_, xref, actuation_->get_nu());
      boost::shared_ptr<crocoddyl::CostModelAbstract> impulseFootVelCost =
          boost::make_shared<crocoddyl::CostModelFrameVelocity>(state_, vref, actuation_->get_nu());
      costModel->addCost(rmodel_.frames[it->frame].name + "_footTrack", footTrack, 1e7);
      costModel->addCost(rmodel_.frames[it->frame].name + "_impulseVel", impulseFootVelCost, 1e6);
    }
  }
  Eigen::VectorXd stateWeights(2 * rmodel_.nv);
  stateWeights.head<3>().fill(0.);
  stateWeights.segment<3>(3).fill(pow(500., 2));
  stateWeights.segment(6, rmodel_.nv - 6).fill(pow(0.01, 2));
  stateWeights.segment(rmodel_.nv, rmodel_.nv).fill(pow(10., 2));
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> stateActivation =
      boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(stateWeights);
  boost::shared_ptr<crocoddyl::CostModelAbstract> stateReg =
      boost::make_shared<crocoddyl::CostModelState>(state_, stateActivation, defaultState_, actuation_->get_nu());
  boost::shared_ptr<crocoddyl::CostModelAbstract> ctrlReg =
      boost::make_shared<crocoddyl::CostModelControl>(state_, actuation_->get_nu());
  costModel->addCost("stateReg", stateReg, 1e1);
  costModel->addCost("ctrlReg", ctrlReg, 1e-3);

  // Creating the action model for the KKT dynamics with simpletic Euler integration scheme
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> dmodel =
      boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state_, actuation_, contactModel,
                                                                               costModel);
  return boost::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodel, 0.);
}

boost::shared_ptr<ActionModelAbstract> SimpleQuadrupedGaitProblem::createImpulseModel(
    const std::vector<pinocchio::FrameIndex>& supportFootIds, const std::vector<FramePlacement>& swingFootTask) {
  // Creating a 3D multi-contact model, and then including the supporting foot
  boost::shared_ptr<crocoddyl::ImpulseModelMultiple> impulseModel =
      boost::make_shared<crocoddyl::ImpulseModelMultiple>(state_);
  for (std::vector<pinocchio::FrameIndex>::const_iterator it = supportFootIds.begin(); it != supportFootIds.end();
       ++it) {
    boost::shared_ptr<crocoddyl::ImpulseModelAbstract> supportContactModel =
        boost::make_shared<crocoddyl::ImpulseModel3D>(state_, *it);
    impulseModel->addImpulse(rmodel_.frames[*it].name + "_impulse", supportContactModel);
  }

  // Creating the cost model for a contact phase
  boost::shared_ptr<crocoddyl::CostModelSum> costModel = boost::make_shared<crocoddyl::CostModelSum>(state_, 0, true);
  if (!swingFootTask.empty()) {
    for (std::vector<crocoddyl::FramePlacement>::const_iterator it = swingFootTask.begin(); it != swingFootTask.end();
         ++it) {
      crocoddyl::FrameTranslation xref(it->frame, it->oMf.translation());
      boost::shared_ptr<crocoddyl::CostModelAbstract> footTrack =
          boost::make_shared<crocoddyl::CostModelFrameTranslation>(state_, xref, 0);
      costModel->addCost(rmodel_.frames[it->frame].name + "_footTrack", footTrack, 1e7);
    }
  }
  Eigen::VectorXd stateWeights(2 * rmodel_.nv);
  stateWeights.head<6>().fill(1.);
  stateWeights.segment(6, rmodel_.nv - 6).fill(pow(10., 2));
  stateWeights.segment(rmodel_.nv, rmodel_.nv).fill(pow(10., 2));
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> stateActivation =
      boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(stateWeights);
  boost::shared_ptr<crocoddyl::CostModelAbstract> stateReg =
      boost::make_shared<crocoddyl::CostModelState>(state_, stateActivation, defaultState_, 0);
  costModel->addCost("stateReg", stateReg, 1e1);

  // Creating the action model for the KKT dynamics with simpletic Euler integration scheme
  return boost::make_shared<crocoddyl::ActionModelImpulseFwdDynamics>(state_, impulseModel, costModel);
}

}  // namespace crocoddyl
