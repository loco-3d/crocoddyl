#include "quadruped-gaits.hpp"

namespace crocoddyl {

  SimpleQuadrupedGaitProblem::SimpleQuadrupedGaitProblem(const pinocchio::Model& rmodel_,
                                                         const std::string& lfFoot_,
                                                         const std::string& rfFoot_,
                                                         const std::string& lhFoot_,
                                                         const std::string& rhFoot_):
    rmodel(rmodel_),
    rdata(rmodel),
    lfFootId(rmodel.getFrameId(lfFoot_,
                               (pinocchio::FrameType) (pinocchio::JOINT | pinocchio::FIXED_JOINT | pinocchio::BODY))),
    rfFootId(rmodel.getFrameId(rfFoot_,
                               (pinocchio::FrameType) (pinocchio::JOINT | pinocchio::FIXED_JOINT | pinocchio::BODY))),
    lhFootId(rmodel.getFrameId(lhFoot_,
                               (pinocchio::FrameType) (pinocchio::JOINT | pinocchio::FIXED_JOINT | pinocchio::BODY))),
    rhFootId(rmodel.getFrameId(rhFoot_,
                               (pinocchio::FrameType) (pinocchio::JOINT | pinocchio::FIXED_JOINT | pinocchio::BODY))),
    state(rmodel),
    actuation(state),
    firstStep(true),
    defaultState(rmodel.nq+rmodel.nv)
  {
    defaultState.head(rmodel.nq) = rmodel.referenceConfigurations["half_sitting"];
    defaultState.tail(rmodel.nv).setZero();
  }
  
  void
  SimpleQuadrupedGaitProblem::createFootStepModels(double timeStep, Eigen::Vector3d& comPos0,
                                                   std::vector<Eigen::Vector3d>& feetPos0,
                                                   double stepLength, double stepHeight,
                                                   unsigned int numKnots,
                                                   const std::vector<pinocchio::FrameIndex>& supportFootIds,
                                                   const std::vector<pinocchio::FrameIndex>& swingFootIds,
                                                   std::vector<ActionModelAbstract*>& actionModelList)
  {
    typedef std::vector<FramePlacement> FramePlacementVector;
    unsigned int numLegs = supportFootIds.size() + swingFootIds.size();
    double comPercentage = (swingFootIds.size())/numLegs;
    Eigen::Vector3d dp = Eigen::Vector3d::Zero();
    std::vector<FramePlacementVector> swingFootTaskVector;
    for(unsigned int k=0;k<numKnots;k++) {
      FramePlacementVector swingFootTask;
      for(unsigned int i=0;i<swingFootIds.size();i++){
        unsigned int phKnots = numKnots>>1; //bitwise divide.
        if(k<phKnots)
          dp << stepLength * (k+1)/numKnots, 0., stepHeight *k/phKnots;
        else if(k==phKnots)
          dp << stepLength * (k+1)/numKnots, 0., stepHeight;
        else
          dp << stepLength * (k+1)/numKnots, 0., stepHeight * (1-(k-phKnots)/phKnots);
        swingFootTask.push_back(FramePlacement(swingFootIds[i],
                                               pinocchio::SE3(Eigen::Matrix3d::Identity(),
                                                              feetPos0[i]+dp)));
      }
      swingFootTaskVector.push_back(swingFootTask);
      Eigen::Vector3d comTask = Eigen::Vector3d(stepLength*(k+1)/numKnots, 0., 0.)*comPercentage + comPos0;
      createSwingFootModel(timeStep, supportFootIds, comTask,
                           swingFootTaskVector.back(), actionModelList);
    }
    createFootSwitchModel(supportFootIds, swingFootTaskVector.back(), true, actionModelList);
    comPos0 += Eigen::Vector3d(stepLength * comPercentage, 0., 0.);
    for(unsigned int i=0;i<feetPos0.size();i++){
      feetPos0[i] += Eigen::Vector3d(stepLength, 0.,0.);
    }
    return;
  }
  
  void
  SimpleQuadrupedGaitProblem::createSwingFootModel(double timeStep,
                                                   const std::vector<pinocchio::FrameIndex>& supportFootIds,
                                                   const Eigen::Vector3d& comTask,
                                                   const std::vector<FramePlacement>& swingFootTask,
                                                   std::vector<ActionModelAbstract*>& actionModelList) {
    // Creating a 3D multi-contact model, and then including the supporting
    // foot
    ContactModelMultiple contactModel(state, actuation.get_nu());
    std::vector<FrameTranslation> xrefVector;
    
    std::list<ContactModel3D> contacts3d;//[supportFootIds.size()] = ;
    
    for(unsigned int i=0;i<supportFootIds.size();i++) {
      xrefVector.push_back(FrameTranslation(supportFootIds[i],
                                            Eigen::Vector3d::Zero()));
      contacts3d.push_back(ContactModel3D(state, xrefVector.back(),
                                          actuation.get_nu(),
                                          Eigen::Vector2d(0., 50.)));
      contactModel.addContact("contact_" + rmodel.frames[supportFootIds[i]].name,
                              &(contacts3d.back()));
    }
    // Creating the cost model for a contact phase
    CostModelSum costModel(state, actuation.get_nu());
    if(!comTask.array().allFinite()) {
      CostModelCoMPosition comTrack(state, comTask, actuation.get_nu());
      costModel.addCost("comTrack", &(comTrack), 1e4);
    }
    if(!swingFootTask.empty()) {
      for(unsigned int i=0;i<swingFootTask.size();i++) {
        FrameTranslation xref(swingFootTask[i].frame,
                              swingFootTask[i].oMf.translation());
        CostModelFrameTranslation footTrack(state, xref, actuation.get_nu());
        costModel.addCost("footTrack_" + rmodel.frames[swingFootTask[i].frame].name,
                          &footTrack, 1e4);
      }
    }
    Eigen::VectorXd stateWeights(2*rmodel.nv);
    stateWeights.head<3>().fill(0.);
    stateWeights.segment<3>(3).fill(pow(500., 2));
    stateWeights.segment(6, rmodel.nv-6).fill(pow(0.01, 2));
    stateWeights.segment(rmodel.nv, rmodel.nv).fill(pow(10., 2));
    ActivationModelWeightedQuad activation_wt(stateWeights);
    CostModelState stateReg(state,
                            activation_wt,
                            defaultState, actuation.get_nu());
    CostModelControl ctrlReg(state, actuation.get_nu());
    costModel.addCost("stateReg", &stateReg, 1e-1);
    costModel.addCost("ctrlReg", &ctrlReg, 1e-4);
    
    // Creating the action model for the KKT dynamics with simpletic Euler
    // integration scheme
    DifferentialActionModelContactFwdDynamics dmodel(state, actuation,
                                                     contactModel, costModel);
    IntegratedActionModelEuler model(&dmodel, timeStep);
    actionModelList.push_back(&model);
    return;
  }
  
  void
  SimpleQuadrupedGaitProblem::createFootSwitchModel(const std::vector<pinocchio::FrameIndex>& supportFootIds,
                                                    const std::vector<FramePlacement>& swingFootTask,
                                                    bool pseudoImpulse,
                                                    std::vector<ActionModelAbstract*>& actionModelList) {
    if(pseudoImpulse) {
      return createPseudoImpulseModel(supportFootIds, swingFootTask, actionModelList);
    }
    else {
      return createImpulseModel(supportFootIds, swingFootTask, actionModelList);
    }
  }
  
  void
  SimpleQuadrupedGaitProblem::createPseudoImpulseModel(const std::vector<pinocchio::FrameIndex>& supportFootIds,
                                                       const std::vector<FramePlacement>& swingFootTask,
                                                       std::vector<ActionModelAbstract*>& actionModelList){
    // Creating a 3D multi-contact model, and then including the supporting
    // foot
    ContactModelMultiple contactModel(state, actuation.get_nu());
    for(unsigned int i;i<supportFootIds.size();i++) {
      FrameTranslation xref(supportFootIds[i], Eigen::Vector3d::Zero());
      ContactModel3D supportContactModel(state, xref, actuation.get_nu(),
                                         Eigen::Vector2d(0., 50.));
      contactModel.addContact("contact_" + rmodel.frames[supportFootIds[i]].name,
                              &supportContactModel);
    }
    
    // Creating the cost model for a contact phase
    CostModelSum costModel(state, actuation.get_nu());
    if (!swingFootTask.empty()) {
      for(unsigned int i=0;i<swingFootTask.size();i++) {
        const FramePlacement& task = swingFootTask[i];
        const FrameTranslation xref(task.frame, task.oMf.translation());
        const FrameMotion vref(task.frame, pinocchio::Motion::Zero());
        CostModelFrameTranslation footTrack(state, xref, actuation.get_nu());
        CostModelFrameVelocity impulseFootVelCost(state, vref,
                                                  actuation.get_nu());
        costModel.addCost("footTrack_" + rmodel.frames[swingFootTask[i].frame].name,
                          &footTrack, 1e7);
        costModel.addCost("impulseVel_" + rmodel.frames[swingFootTask[i].frame].name,
                          &impulseFootVelCost, 1e6);
      }
    }
    Eigen::VectorXd stateWeights(2*rmodel.nv);
    stateWeights.head<3>().fill(0.);
    stateWeights.segment<3>(3).fill(pow(500., 2));
    stateWeights.segment(6, rmodel.nv-6).fill(pow(0.01, 2));
    stateWeights.segment(rmodel.nv, rmodel.nv).fill(pow(10., 2));
    ActivationModelWeightedQuad activation_swt(stateWeights);
    CostModelState stateReg(state,
                            activation_swt,
                            defaultState, actuation.get_nu());
    CostModelControl ctrlReg(state, actuation.get_nu());
    costModel.addCost("stateReg", &stateReg, 1e1);
    costModel.addCost("ctrlReg", &ctrlReg, 1e-3);
    
    // Creating the action model for the KKT dynamics with simpletic Euler
    // integration scheme
    DifferentialActionModelContactFwdDynamics dmodel(state, actuation,
                                                     contactModel, costModel);
    IntegratedActionModelEuler model(&dmodel, 0.);
    actionModelList.push_back(&model);
    return;
  }
  
  void
  SimpleQuadrupedGaitProblem::createImpulseModel(const std::vector<pinocchio::FrameIndex>& supportFootIds,
                                                 const std::vector<FramePlacement>& swingFootTask,
                                                 std::vector<ActionModelAbstract*>& actionModelList) {
    
    // Creating a 3D multi-contact model, and then including the supporting foot
    ImpulseModelMultiple impulseModel(state);
    for(unsigned int i;i<supportFootIds.size();i++) {
      ImpulseModel3D supportContactModel(state, supportFootIds[i]);
      impulseModel.addImpulse("impulse_" + rmodel.frames[supportFootIds[i]].name,
                              &supportContactModel);
    }
    // Creating the cost model for a contact phase
    CostModelSum costModel(state, 0, true);
    if (!swingFootTask.empty()) {
      for(unsigned int i;i<swingFootTask.size();i++) {
        const FramePlacement task = swingFootTask[i];
        FrameTranslation xref(task.frame, task.oMf.translation());
        CostModelFrameTranslation footTrack(state, xref, 0);
        costModel.addCost("footTrack_" + rmodel.frames[task.frame].name, &footTrack, 1e7);
      }
    }
    Eigen::VectorXd stateWeights(2*rmodel.nv);
    stateWeights.head<6>().fill(1.);
    stateWeights.segment(6, rmodel.nv-6).fill(pow(10., 2));
    stateWeights.segment(rmodel.nv, rmodel.nv).fill(pow(10., 2));
    ActivationModelWeightedQuad activation_swt(stateWeights);
    CostModelState stateReg(state,
                            activation_swt,
                            defaultState, 0);
    costModel.addCost("stateReg", &stateReg, 1e1);
    
    // Creating the action model for the KKT dynamics with simpletic Euler
    // integration scheme
    ActionModelImpulseFwdDynamics model(state, impulseModel,
                                        costModel);
    actionModelList.push_back(&model);
    return;
  }
  
  ShootingProblem
  SimpleQuadrupedGaitProblem::createWalkingProblem(const Eigen::VectorXd& x0,
                                                   const double stepLength,
                                                   const double stepHeight,
                                                   const double timeStep,
                                                   const unsigned int stepKnots,
                                                   const unsigned int supportKnots) {
    int nq = rmodel.nq;
    int nv = rmodel.nv;
    
    //Initial Condition
    const Eigen::VectorBlock<const Eigen::VectorXd> q0 = x0.head(nq);
    pinocchio::forwardKinematics(rmodel, rdata, q0);
    pinocchio::centerOfMass(rmodel, rdata, q0);
    pinocchio::updateFramePlacements(rmodel, rdata);
    
    const pinocchio::SE3::Vector3& rfFootPos0 = rdata.oMf[rfFootId].translation();
    const pinocchio::SE3::Vector3& rhFootPos0 = rdata.oMf[rhFootId].translation();
    const pinocchio::SE3::Vector3& lfFootPos0 = rdata.oMf[lfFootId].translation();
    const pinocchio::SE3::Vector3& lhFootPos0 = rdata.oMf[lhFootId].translation();
    
    pinocchio::SE3::Vector3 comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0)/4;
    comRef[2] = rdata.com[0][2,0];
    
    //Defining the action models along the time instances
    std::vector<ActionModelAbstract*> loco3dModel;
    std::vector<ActionModelAbstract*> doubleSupport1, doubleSupport2,
      rhStep, rfStep, lhStep, lfStep;
    
    //doublesupport
    std::vector<pinocchio::FrameIndex> supportFeet;
    supportFeet.push_back(lfFootId);
    supportFeet.push_back(rfFootId);
    supportFeet.push_back(lhFootId);
    supportFeet.push_back(rhFootId);
    Eigen::Vector3d nullCom;
    const std::vector<FramePlacement> emptyVector;
    nullCom.fill(std::numeric_limits<double>::infinity());
    for(unsigned int k=0;k<supportKnots;k++){
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
    
    const pinocchio::FrameIndex arr_lf_rf_lh[] = {lfFootId, rfFootId, lhFootId};
    const pinocchio::FrameIndex arr_lf_lh_rh[] = {lfFootId, lhFootId, rhFootId};
    const pinocchio::FrameIndex arr_lf_rf_rh[] = {lfFootId, rfFootId, rhFootId};
    const pinocchio::FrameIndex arr_rf_lh_rh[] = {rfFootId, lhFootId, rhFootId};
    
    std::vector<pinocchio::FrameIndex> legs_lf_rf_lh (arr_lf_rf_lh,
                                                      arr_lf_rf_lh + sizeof(arr_lf_rf_lh) /
                                                      sizeof(arr_lf_rf_lh[0]));
    std::vector<pinocchio::FrameIndex> legs_lf_lh_rh (arr_lf_lh_rh,
                                                      arr_lf_lh_rh + sizeof(arr_lf_lh_rh) /
                                                      sizeof(arr_lf_lh_rh[0]));
    std::vector<pinocchio::FrameIndex> legs_lf_rf_rh (arr_lf_rf_rh,
                                                      arr_lf_rf_rh + sizeof(arr_lf_rf_rh) /
                                                      sizeof(arr_lf_rf_rh[0]));
    std::vector<pinocchio::FrameIndex> legs_rf_lh_rh (arr_rf_lh_rh,
                                                      arr_rf_lh_rh + sizeof(arr_rf_lh_rh) /
                                                      sizeof(arr_rf_lh_rh[0]));
    
    std::vector<pinocchio::FrameIndex> legs_rh, legs_rf, legs_lf, legs_lh;
    legs_rh.push_back(rhFootId);
    legs_lh.push_back(lhFootId);
    legs_rf.push_back(rfFootId);
    legs_lf.push_back(lfFootId);
    
    
    if(firstStep){
      createFootStepModels(timeStep, comRef, rhFootPos0vec,
                           0.5 * stepLength, stepHeight, stepKnots,
                           legs_lf_rf_lh, legs_rh, rhStep);
      createFootStepModels(timeStep, comRef, rhFootPos0vec,
                           0.5 * stepLength, stepHeight,
                           stepKnots,
                           legs_lf_lh_rh,
                           legs_rf, rfStep);
      firstStep = false;
    }    
    else{
      createFootStepModels(timeStep, comRef, rhFootPos0vec,
                           stepLength, stepHeight, stepKnots,
                           legs_lf_rf_lh,
                           legs_rh, rhStep);
      createFootStepModels(timeStep, comRef, rfFootPos0vec,
                           stepLength, stepHeight, stepKnots,
                           legs_lf_lh_rh,
                           legs_rf, rfStep);
    }
    createFootStepModels(timeStep, comRef, lhFootPos0vec,
                         stepLength, stepHeight, stepKnots,
                         legs_lf_rf_rh,
                         legs_lh, lhStep);
    createFootStepModels(timeStep, comRef, lfFootPos0vec,
                         stepLength, stepHeight, stepKnots,
                         legs_rf_lh_rh,
                         legs_lf, lfStep);
    
    for (unsigned int i=0;i<doubleSupport1.size();i++) loco3dModel.push_back(doubleSupport1[i]);
    for (unsigned int i=0;i<rhStep.size();i++) loco3dModel.push_back(rhStep[i]);
    for (unsigned int i=0;i<rfStep.size();i++) loco3dModel.push_back(rfStep[i]);
    for (unsigned int i=0;i<doubleSupport2.size();i++) loco3dModel.push_back(doubleSupport2[i]);
    for (unsigned int i=0;i<lhStep.size();i++) loco3dModel.push_back(lhStep[i]);
    for (unsigned int i=0;i<lfStep.size();i++) loco3dModel.push_back(lfStep[i]);
    
    ShootingProblem problem(x0, loco3dModel, loco3dModel.back());
    return problem;
}
}
