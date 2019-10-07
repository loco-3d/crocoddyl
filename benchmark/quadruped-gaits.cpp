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
      state_(rmodel_),
      actuation_(state_),
      firstStep_(true),
      defaultState_(rmodel_.nq + rmodel_.nv) {
  defaultState_.head(rmodel_.nq) = rmodel_.referenceConfigurations["half_sitting"];
  defaultState_.tail(rmodel_.nv).setZero();
}

SimpleQuadrupedGaitProblem::~SimpleQuadrupedGaitProblem() {}

ShootingProblem SimpleQuadrupedGaitProblem::createWalkingProblem(const Eigen::VectorXd& x0, const double stepLength,
                                                                 const double stepHeight, const double timeStep,
                                                                 const std::size_t stepKnots,
                                                                 const std::size_t supportKnots) {
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
  std::vector<ActionModelAbstract*> loco3dModel;
  std::vector<ActionModelAbstract*> doubleSupport1, doubleSupport2, rhStep, rfStep, lhStep, lfStep;

  // doublesupport
  std::vector<pinocchio::FrameIndex> supportFeet;
  supportFeet.push_back(lfFootId_);
  supportFeet.push_back(rfFootId_);
  supportFeet.push_back(lhFootId_);
  supportFeet.push_back(rhFootId_);
  Eigen::Vector3d nullCom;
  const std::vector<FramePlacement> emptyVector;
  nullCom.fill(std::numeric_limits<double>::infinity());
  for (std::size_t k = 0; k < supportKnots; k++) {
    createSwingFootModel(timeStep, supportFeet, nullCom, emptyVector, doubleSupport1);
    createSwingFootModel(timeStep, supportFeet, nullCom, emptyVector, doubleSupport2);
  }

  std::vector<Eigen::Vector3d> rhFootPos0vec;
  rhFootPos0vec.push_back(rhFootPos0);
  std::vector<Eigen::Vector3d> lhFootPos0vec;
  lhFootPos0vec.push_back(lhFootPos0);
  std::vector<Eigen::Vector3d> rfFootPos0vec;
  rfFootPos0vec.push_back(rfFootPos0);
  std::vector<Eigen::Vector3d> lfFootPos0vec;
  lfFootPos0vec.push_back(lfFootPos0);

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

  std::vector<pinocchio::FrameIndex> legs_rh, legs_rf, legs_lf, legs_lh;
  legs_rh.push_back(rhFootId_);
  legs_lh.push_back(lhFootId_);
  legs_rf.push_back(rfFootId_);
  legs_lf.push_back(lfFootId_);

  if (firstStep_) {
    createFootStepModels(timeStep, comRef, rhFootPos0vec, 0.5 * stepLength, stepHeight, stepKnots, legs_lf_rf_lh,
                         legs_rh, rhStep);
    createFootStepModels(timeStep, comRef, rhFootPos0vec, 0.5 * stepLength, stepHeight, stepKnots, legs_lf_lh_rh,
                         legs_rf, rfStep);
    firstStep_ = false;
  } else {
    createFootStepModels(timeStep, comRef, rhFootPos0vec, stepLength, stepHeight, stepKnots, legs_lf_rf_lh, legs_rh,
                         rhStep);
    createFootStepModels(timeStep, comRef, rfFootPos0vec, stepLength, stepHeight, stepKnots, legs_lf_lh_rh, legs_rf,
                         rfStep);
  }
  createFootStepModels(timeStep, comRef, lhFootPos0vec, stepLength, stepHeight, stepKnots, legs_lf_rf_rh, legs_lh,
                       lhStep);
  createFootStepModels(timeStep, comRef, lfFootPos0vec, stepLength, stepHeight, stepKnots, legs_rf_lh_rh, legs_lf,
                       lfStep);

  for (std::size_t i = 0; i < doubleSupport1.size(); i++) loco3dModel.push_back(doubleSupport1[i]);
  for (std::size_t i = 0; i < rhStep.size(); i++) loco3dModel.push_back(rhStep[i]);
  for (std::size_t i = 0; i < rfStep.size(); i++) loco3dModel.push_back(rfStep[i]);
  for (std::size_t i = 0; i < doubleSupport2.size(); i++) loco3dModel.push_back(doubleSupport2[i]);
  for (std::size_t i = 0; i < lhStep.size(); i++) loco3dModel.push_back(lhStep[i]);
  for (std::size_t i = 0; i < lfStep.size(); i++) loco3dModel.push_back(lfStep[i]);

  ShootingProblem problem(x0, loco3dModel, loco3dModel.back());
  return problem;
}

void SimpleQuadrupedGaitProblem::createFootStepModels(double timeStep, Eigen::Vector3d& comPos0,
                                                      std::vector<Eigen::Vector3d>& feetPos0, double stepLength,
                                                      double stepHeight, std::size_t numKnots,
                                                      const std::vector<pinocchio::FrameIndex>& supportFootIds,
                                                      const std::vector<pinocchio::FrameIndex>& swingFootIds,
                                                      std::vector<ActionModelAbstract*>& actionModelList) {
  typedef std::vector<FramePlacement> FramePlacementVector;
  std::size_t numLegs = static_cast<std::size_t>(supportFootIds.size() + swingFootIds.size());
  double comPercentage = static_cast<double>(swingFootIds.size() / numLegs);
  Eigen::Vector3d dp = Eigen::Vector3d::Zero();
  std::vector<FramePlacementVector> swingFootTaskVector;
  double _kp1_n = 0;
  for (std::size_t k = 0; k < numKnots; k++) {
    FramePlacementVector swingFootTask;
    for (std::size_t i = 0; i < swingFootIds.size(); ++i) {
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
      swingFootTask.push_back(
          FramePlacement(swingFootIds[i], pinocchio::SE3(Eigen::Matrix3d::Identity(), feetPos0[i] + dp)));
    }
    swingFootTaskVector.push_back(swingFootTask);
    Eigen::Vector3d comTask = Eigen::Vector3d(stepLength * _kp1_n, 0., 0.) * comPercentage + comPos0;
    createSwingFootModel(timeStep, supportFootIds, comTask, swingFootTaskVector.back(), actionModelList);
  }
  createFootSwitchModel(supportFootIds, swingFootTaskVector.back(), true, actionModelList);
  comPos0 += Eigen::Vector3d(stepLength * comPercentage, 0., 0.);
  for (std::size_t i = 0; i < feetPos0.size(); i++) {
    feetPos0[i] += Eigen::Vector3d(stepLength, 0., 0.);
  }
  return;
}

void SimpleQuadrupedGaitProblem::createSwingFootModel(double timeStep,
                                                      const std::vector<pinocchio::FrameIndex>& supportFootIds,
                                                      const Eigen::Vector3d& comTask,
                                                      const std::vector<FramePlacement>& swingFootTask,
                                                      std::vector<ActionModelAbstract*>& actionModelList) {
  // Creating a 3D multi-contact model, and then including the supporting
  // foot
  ContactModelMultiple contactModel(state_, actuation_.get_nu());
  std::vector<FrameTranslation> xrefVector;

  std::list<ContactModel3D> contacts3d;  //[supportFootIds.size()] = ;

  for (std::size_t i = 0; i < supportFootIds.size(); i++) {
    xrefVector.push_back(FrameTranslation(supportFootIds[i], Eigen::Vector3d::Zero()));
    contacts3d.push_back(ContactModel3D(state_, xrefVector.back(), actuation_.get_nu(), Eigen::Vector2d(0., 50.)));
    contactModel.addContact("contact_" + rmodel_.frames[supportFootIds[i]].name, &(contacts3d.back()));
  }
  // Creating the cost model for a contact phase
  CostModelSum costModel(state_, actuation_.get_nu());
  if (!comTask.array().allFinite()) {
    CostModelCoMPosition comTrack(state_, comTask, actuation_.get_nu());
    costModel.addCost("comTrack", &(comTrack), 1e4);
  }
  if (!swingFootTask.empty()) {
    for (std::size_t i = 0; i < swingFootTask.size(); i++) {
      FrameTranslation xref(swingFootTask[i].frame, swingFootTask[i].oMf.translation());
      CostModelFrameTranslation footTrack(state_, xref, actuation_.get_nu());
      costModel.addCost("footTrack_" + rmodel_.frames[swingFootTask[i].frame].name, &footTrack, 1e4);
    }
  }
  Eigen::VectorXd state_Weights(2 * rmodel_.nv);
  state_Weights.head<3>().fill(0.);
  state_Weights.segment<3>(3).fill(pow(500., 2));
  state_Weights.segment(6, rmodel_.nv - 6).fill(pow(0.01, 2));
  state_Weights.segment(rmodel_.nv, rmodel_.nv).fill(pow(10., 2));
  ActivationModelWeightedQuad activation_wt(state_Weights);
  CostModelState state_Reg(state_, activation_wt, defaultState_, actuation_.get_nu());
  CostModelControl ctrlReg(state_, actuation_.get_nu());
  costModel.addCost("state_Reg", &state_Reg, 1e-1);
  costModel.addCost("ctrlReg", &ctrlReg, 1e-4);

  // Creating the action model for the KKT dynamics with simpletic Euler
  // integration scheme
  DifferentialActionModelContactFwdDynamics dmodel(state_, actuation_, contactModel, costModel);
  IntegratedActionModelEuler model(&dmodel, timeStep);
  actionModelList.push_back(&model);
  return;
}

void SimpleQuadrupedGaitProblem::createFootSwitchModel(const std::vector<pinocchio::FrameIndex>& supportFootIds,
                                                       const std::vector<FramePlacement>& swingFootTask,
                                                       bool pseudoImpulse,
                                                       std::vector<ActionModelAbstract*>& actionModelList) {
  if (pseudoImpulse) {
    return createPseudoImpulseModel(supportFootIds, swingFootTask, actionModelList);
  } else {
    return createImpulseModel(supportFootIds, swingFootTask, actionModelList);
  }
}

void SimpleQuadrupedGaitProblem::createPseudoImpulseModel(const std::vector<pinocchio::FrameIndex>& supportFootIds,
                                                          const std::vector<FramePlacement>& swingFootTask,
                                                          std::vector<ActionModelAbstract*>& actionModelList) {
  // Creating a 3D multi-contact model, and then including the supporting
  // foot
  ContactModelMultiple contactModel(state_, actuation_.get_nu());
  for (std::size_t i = 0; i < supportFootIds.size(); i++) {
    FrameTranslation xref(supportFootIds[i], Eigen::Vector3d::Zero());
    ContactModel3D supportContactModel(state_, xref, actuation_.get_nu(), Eigen::Vector2d(0., 50.));
    contactModel.addContact("contact_" + rmodel_.frames[supportFootIds[i]].name, &supportContactModel);
  }

  // Creating the cost model for a contact phase
  CostModelSum costModel(state_, actuation_.get_nu());
  if (!swingFootTask.empty()) {
    for (std::size_t i = 0; i < swingFootTask.size(); i++) {
      const FramePlacement& task = swingFootTask[i];
      const FrameTranslation xref(task.frame, task.oMf.translation());
      const FrameMotion vref(task.frame, pinocchio::Motion::Zero());
      CostModelFrameTranslation footTrack(state_, xref, actuation_.get_nu());
      CostModelFrameVelocity impulseFootVelCost(state_, vref, actuation_.get_nu());
      costModel.addCost("footTrack_" + rmodel_.frames[swingFootTask[i].frame].name, &footTrack, 1e7);
      costModel.addCost("impulseVel_" + rmodel_.frames[swingFootTask[i].frame].name, &impulseFootVelCost, 1e6);
    }
  }
  Eigen::VectorXd state_Weights(2 * rmodel_.nv);
  state_Weights.head<3>().fill(0.);
  state_Weights.segment<3>(3).fill(pow(500., 2));
  state_Weights.segment(6, rmodel_.nv - 6).fill(pow(0.01, 2));
  state_Weights.segment(rmodel_.nv, rmodel_.nv).fill(pow(10., 2));
  ActivationModelWeightedQuad activation_swt(state_Weights);
  CostModelState state_Reg(state_, activation_swt, defaultState_, actuation_.get_nu());
  CostModelControl ctrlReg(state_, actuation_.get_nu());
  costModel.addCost("state_Reg", &state_Reg, 1e1);
  costModel.addCost("ctrlReg", &ctrlReg, 1e-3);

  // Creating the action model for the KKT dynamics with simpletic Euler
  // integration scheme
  DifferentialActionModelContactFwdDynamics dmodel(state_, actuation_, contactModel, costModel);
  IntegratedActionModelEuler model(&dmodel, 0.);
  actionModelList.push_back(&model);
  return;
}

void SimpleQuadrupedGaitProblem::createImpulseModel(const std::vector<pinocchio::FrameIndex>& supportFootIds,
                                                    const std::vector<FramePlacement>& swingFootTask,
                                                    std::vector<ActionModelAbstract*>& actionModelList) {
  // Creating a 3D multi-contact model, and then including the supporting foot
  ImpulseModelMultiple impulseModel(state_);
  for (std::size_t i = 0; i < supportFootIds.size(); ++i) {
    ImpulseModel3D supportContactModel(state_, supportFootIds[i]);
    impulseModel.addImpulse("impulse_" + rmodel_.frames[supportFootIds[i]].name, &supportContactModel);
  }
  // Creating the cost model for a contact phase
  CostModelSum costModel(state_, 0, true);
  if (!swingFootTask.empty()) {
    for (std::size_t i = 0; i < swingFootTask.size(); i++) {
      const FramePlacement task = swingFootTask[i];
      FrameTranslation xref(task.frame, task.oMf.translation());
      CostModelFrameTranslation footTrack(state_, xref, 0);
      costModel.addCost("footTrack_" + rmodel_.frames[task.frame].name, &footTrack, 1e7);
    }
  }
  Eigen::VectorXd state_Weights(2 * rmodel_.nv);
  state_Weights.head<6>().fill(1.);
  state_Weights.segment(6, rmodel_.nv - 6).fill(pow(10., 2));
  state_Weights.segment(rmodel_.nv, rmodel_.nv).fill(pow(10., 2));
  ActivationModelWeightedQuad activation_swt(state_Weights);
  CostModelState state_Reg(state_, activation_swt, defaultState_, 0);
  costModel.addCost("state_Reg", &state_Reg, 1e1);

  // Creating the action model for the KKT dynamics with simpletic Euler
  // integration scheme
  ActionModelImpulseFwdDynamics model(state_, impulseModel, costModel);
  actionModelList.push_back(&model);
  return;
}

}  // namespace crocoddyl
