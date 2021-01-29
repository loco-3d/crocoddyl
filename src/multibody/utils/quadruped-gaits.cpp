///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/utils/quadruped-gaits.hpp"

namespace crocoddyl {

SimpleQuadrupedGaitProblem::SimpleQuadrupedGaitProblem(const pinocchio::Model& rmodel, const std::string& lf_foot,
                                                       const std::string& rf_foot, const std::string& lh_foot,
                                                       const std::string& rh_foot)
    : rmodel_(rmodel),
      rdata_(rmodel_),
      lf_foot_id_(rmodel_.getFrameId(
          lf_foot, (pinocchio::FrameType)(pinocchio::JOINT | pinocchio::FIXED_JOINT | pinocchio::BODY))),
      rf_foot_id_(rmodel_.getFrameId(
          rf_foot, (pinocchio::FrameType)(pinocchio::JOINT | pinocchio::FIXED_JOINT | pinocchio::BODY))),
      lh_foot_id_(rmodel_.getFrameId(
          lh_foot, (pinocchio::FrameType)(pinocchio::JOINT | pinocchio::FIXED_JOINT | pinocchio::BODY))),
      rh_foot_id_(rmodel_.getFrameId(
          rh_foot, (pinocchio::FrameType)(pinocchio::JOINT | pinocchio::FIXED_JOINT | pinocchio::BODY))),
      state_(boost::make_shared<crocoddyl::StateMultibody>(boost::make_shared<pinocchio::Model>(rmodel_))),
      actuation_(boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state_)),
      firtstep_(true),
      defaultstate_(rmodel_.nq + rmodel_.nv) {
  defaultstate_.head(rmodel_.nq) = rmodel_.referenceConfigurations["standing"];
  defaultstate_.tail(rmodel_.nv).setZero();
}

SimpleQuadrupedGaitProblem::~SimpleQuadrupedGaitProblem() {}

boost::shared_ptr<crocoddyl::ShootingProblem> SimpleQuadrupedGaitProblem::createWalkingProblem(
    const Eigen::VectorXd& x0, const double steplength, const double stepheight, const double timestep,
    const std::size_t stepknots, const std::size_t supportknots) {
  int nq = rmodel_.nq;

  // Initial Condition
  const Eigen::VectorBlock<const Eigen::VectorXd> q0 = x0.head(nq);
  pinocchio::forwardKinematics(rmodel_, rdata_, q0);
  pinocchio::centerOfMass(rmodel_, rdata_, q0);
  pinocchio::updateFramePlacements(rmodel_, rdata_);

  const pinocchio::SE3::Vector3& rf_foot_pos0 = rdata_.oMf[rf_foot_id_].translation();
  const pinocchio::SE3::Vector3& rh_foot_pos0 = rdata_.oMf[rh_foot_id_].translation();
  const pinocchio::SE3::Vector3& lf_foot_pos0 = rdata_.oMf[lf_foot_id_].translation();
  const pinocchio::SE3::Vector3& lh_foot_pos0 = rdata_.oMf[lh_foot_id_].translation();

  pinocchio::SE3::Vector3 comRef = (rf_foot_pos0 + rh_foot_pos0 + lf_foot_pos0 + lh_foot_pos0) / 4;
  comRef[2] = rdata_.com[0][2];

  // Defining the action models along the time instances
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > loco3d_model;
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > rh_step, rf_step, lh_step, lf_step;

  // doublesupport
  std::vector<pinocchio::FrameIndex> support_feet;
  support_feet.push_back(lf_foot_id_);
  support_feet.push_back(rf_foot_id_);
  support_feet.push_back(lh_foot_id_);
  support_feet.push_back(rh_foot_id_);
  Eigen::Vector3d nullCoM = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  const std::vector<crocoddyl::FramePlacement> emptyVector;
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > doubleSupport(
      supportknots, createSwingFootModel(timestep, support_feet, nullCoM, emptyVector));

  const pinocchio::FrameIndex rh_s[] = {lf_foot_id_, rf_foot_id_, lh_foot_id_};
  const pinocchio::FrameIndex rf_s[] = {lf_foot_id_, lh_foot_id_, rh_foot_id_};
  const pinocchio::FrameIndex lh_s[] = {lf_foot_id_, rf_foot_id_, rh_foot_id_};
  const pinocchio::FrameIndex lf_s[] = {rf_foot_id_, lh_foot_id_, rh_foot_id_};

  std::vector<pinocchio::FrameIndex> rh_support(rh_s, rh_s + sizeof(rh_s) / sizeof(rh_s[0]));
  std::vector<pinocchio::FrameIndex> rf_support(rf_s, rf_s + sizeof(rf_s) / sizeof(rf_s[0]));
  std::vector<pinocchio::FrameIndex> lh_support(lh_s, lh_s + sizeof(lh_s) / sizeof(lh_s[0]));
  std::vector<pinocchio::FrameIndex> lf_support(lf_s, lf_s + sizeof(lf_s) / sizeof(lf_s[0]));

  std::vector<pinocchio::FrameIndex> rh_foot(1, rh_foot_id_);
  std::vector<pinocchio::FrameIndex> rf_foot(1, rf_foot_id_);
  std::vector<pinocchio::FrameIndex> lf_foot(1, lf_foot_id_);
  std::vector<pinocchio::FrameIndex> lh_foot(1, lh_foot_id_);

  std::vector<Eigen::Vector3d> rh_foot_pos0_v(1, rh_foot_pos0);
  std::vector<Eigen::Vector3d> lh_foot_pos0_v(1, lh_foot_pos0);
  std::vector<Eigen::Vector3d> rf_foot_pos0_v(1, rf_foot_pos0);
  std::vector<Eigen::Vector3d> lf_foot_pos0_v(1, lf_foot_pos0);
  if (firtstep_) {
    rh_step = createFootStepModels(timestep, comRef, rh_foot_pos0_v, 0.5 * steplength, stepheight, stepknots,
                                   rh_support, rh_foot);
    rf_step = createFootStepModels(timestep, comRef, rf_foot_pos0_v, 0.5 * steplength, stepheight, stepknots,
                                   rf_support, rf_foot);
    firtstep_ = false;
  } else {
    rh_step =
        createFootStepModels(timestep, comRef, rh_foot_pos0_v, steplength, stepheight, stepknots, rh_support, rh_foot);
    rf_step =
        createFootStepModels(timestep, comRef, rf_foot_pos0_v, steplength, stepheight, stepknots, rf_support, rf_foot);
  }
  lh_step =
      createFootStepModels(timestep, comRef, lh_foot_pos0_v, steplength, stepheight, stepknots, lh_support, lh_foot);
  lf_step =
      createFootStepModels(timestep, comRef, lf_foot_pos0_v, steplength, stepheight, stepknots, lf_support, lf_foot);

  loco3d_model.insert(loco3d_model.end(), doubleSupport.begin(), doubleSupport.end());
  loco3d_model.insert(loco3d_model.end(), rh_step.begin(), rh_step.end());
  loco3d_model.insert(loco3d_model.end(), rf_step.begin(), rf_step.end());
  loco3d_model.insert(loco3d_model.end(), doubleSupport.begin(), doubleSupport.end());
  loco3d_model.insert(loco3d_model.end(), lh_step.begin(), lh_step.end());
  loco3d_model.insert(loco3d_model.end(), lf_step.begin(), lf_step.end());

  return boost::make_shared<crocoddyl::ShootingProblem>(x0, loco3d_model, loco3d_model.back());
}

std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> > SimpleQuadrupedGaitProblem::createFootStepModels(
    double timestep, Eigen::Vector3d& com_pos0, std::vector<Eigen::Vector3d>& feet_pos0, double steplength,
    double stepheight, std::size_t n_knots, const std::vector<pinocchio::FrameIndex>& support_foot_ids,
    const std::vector<pinocchio::FrameIndex>& swingFootIds) {
  std::size_t n_legs = static_cast<std::size_t>(support_foot_ids.size() + swingFootIds.size());
  double com_percentage = static_cast<double>(swingFootIds.size()) / static_cast<double>(n_legs);

  // Action models for the foot swing
  std::vector<boost::shared_ptr<ActionModelAbstract> > foot_swing_model;
  std::vector<crocoddyl::FramePlacement> foot_swing_task;
  for (std::size_t k = 0; k < n_knots; ++k) {
    double _kp1_n = 0;
    Eigen::Vector3d dp = Eigen::Vector3d::Zero();
    foot_swing_task.clear();
    for (std::size_t i = 0; i < swingFootIds.size(); ++i) {
      // Defining a foot swing task given the step length resKnot = n_knots % 2
      std::size_t phaseknots = n_knots >> 1;  // bitwise divide.
      _kp1_n = static_cast<double>(k + 1) / static_cast<double>(n_knots);
      double _k = static_cast<double>(k);
      double _phaseknots = static_cast<double>(phaseknots);
      if (k < phaseknots)
        dp << steplength * _kp1_n, 0., stepheight * _k / _phaseknots;
      else if (k == phaseknots)
        dp << steplength * _kp1_n, 0., stepheight;
      else
        dp << steplength * _kp1_n, 0., stepheight * (1 - (_k - _phaseknots) / _phaseknots);
      Eigen::Vector3d tref = feet_pos0[i] + dp;

      foot_swing_task.push_back(
          crocoddyl::FramePlacement(swingFootIds[i], pinocchio::SE3(Eigen::Matrix3d::Identity(), tref)));
    }

    // Action model for the foot switch
    Eigen::Vector3d com_task = Eigen::Vector3d(steplength * _kp1_n, 0., 0.) * com_percentage + com_pos0;
    foot_swing_model.push_back(createSwingFootModel(timestep, support_foot_ids, com_task, foot_swing_task));
  }
  // Action model for the foot switch
  foot_swing_model.push_back(createFootSwitchModel(support_foot_ids, foot_swing_task));

  // Updating the current foot position for next step
  com_pos0 += Eigen::Vector3d(steplength * com_percentage, 0., 0.);
  for (std::size_t i = 0; i < feet_pos0.size(); ++i) {
    feet_pos0[i] += Eigen::Vector3d(steplength, 0., 0.);
  }
  return foot_swing_model;
}

boost::shared_ptr<crocoddyl::ActionModelAbstract> SimpleQuadrupedGaitProblem::createSwingFootModel(
    double timestep, const std::vector<pinocchio::FrameIndex>& support_foot_ids, const Eigen::Vector3d& com_task,
    const std::vector<crocoddyl::FramePlacement>& foot_swing_task) {
  // Creating a 3D multi-contact model, and then including the supporting foot
  boost::shared_ptr<crocoddyl::ContactModelMultiple> contact_model =
      boost::make_shared<crocoddyl::ContactModelMultiple>(state_, actuation_->get_nu());
  for (std::vector<pinocchio::FrameIndex>::const_iterator it = support_foot_ids.begin(); it != support_foot_ids.end();
       ++it) {
    crocoddyl::FrameTranslation xref(*it, Eigen::Vector3d::Zero());
    boost::shared_ptr<crocoddyl::ContactModelAbstract> support_contact_model =
        boost::make_shared<crocoddyl::ContactModel3D>(state_, xref, actuation_->get_nu(), Eigen::Vector2d(0., 50.));
    contact_model->addContact(rmodel_.frames[*it].name + "_contact", support_contact_model);
  }

  // Creating the cost model for a contact phase
  boost::shared_ptr<crocoddyl::CostModelSum> cost_model =
      boost::make_shared<crocoddyl::CostModelSum>(state_, actuation_->get_nu());
  if (com_task.array().allFinite()) {
    boost::shared_ptr<crocoddyl::CostModelAbstract> com_track =
        boost::make_shared<crocoddyl::CostModelCoMPosition>(state_, com_task, actuation_->get_nu());
    cost_model->addCost("comTrack", com_track, 1e6);
  }
  if (!foot_swing_task.empty()) {
    for (std::vector<crocoddyl::FramePlacement>::const_iterator it = foot_swing_task.begin();
         it != foot_swing_task.end(); ++it) {
      crocoddyl::FrameTranslation xref(it->id, it->placement.translation());
      boost::shared_ptr<crocoddyl::CostModelAbstract> foot_track =
          boost::make_shared<crocoddyl::CostModelFrameTranslation>(state_, xref, actuation_->get_nu());
      cost_model->addCost(rmodel_.frames[it->id].name + "_footTrack", foot_track, 1e6);
    }
  }
  Eigen::VectorXd state_weights(2 * rmodel_.nv);
  state_weights.head<3>().fill(0.);
  state_weights.segment<3>(3).fill(pow(500., 2));
  state_weights.segment(6, rmodel_.nv - 6).fill(pow(0.01, 2));
  state_weights.segment(rmodel_.nv, 6).fill(pow(10., 2));
  state_weights.segment(rmodel_.nv + 6, rmodel_.nv - 6).fill(pow(1., 2));
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> state_activation =
      boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(state_weights);
  boost::shared_ptr<crocoddyl::CostModelAbstract> state_reg =
      boost::make_shared<crocoddyl::CostModelState>(state_, state_activation, defaultstate_, actuation_->get_nu());
  boost::shared_ptr<crocoddyl::CostModelAbstract> ctrl_reg =
      boost::make_shared<crocoddyl::CostModelControl>(state_, actuation_->get_nu());
  cost_model->addCost("stateReg", state_reg, 1e1);
  cost_model->addCost("ctrlReg", ctrl_reg, 1e-1);

  // Creating the action model for the KKT dynamics with simpletic Euler integration scheme
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> dmodel =
      boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state_, actuation_, contact_model,
                                                                               cost_model);
  return boost::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodel, timestep);
}

boost::shared_ptr<ActionModelAbstract> SimpleQuadrupedGaitProblem::createFootSwitchModel(
    const std::vector<pinocchio::FrameIndex>& support_foot_ids,
    const std::vector<crocoddyl::FramePlacement>& foot_swing_task, bool pseudo_impulse) {
  if (pseudo_impulse) {
    return createPseudoImpulseModel(support_foot_ids, foot_swing_task);
  } else {
    return createImpulseModel(support_foot_ids, foot_swing_task);
  }
}

boost::shared_ptr<crocoddyl::ActionModelAbstract> SimpleQuadrupedGaitProblem::createPseudoImpulseModel(
    const std::vector<pinocchio::FrameIndex>& support_foot_ids,
    const std::vector<crocoddyl::FramePlacement>& foot_swing_task) {
  // Creating a 3D multi-contact model, and then including the supporting foot
  boost::shared_ptr<crocoddyl::ContactModelMultiple> contact_model =
      boost::make_shared<crocoddyl::ContactModelMultiple>(state_, actuation_->get_nu());
  for (std::vector<pinocchio::FrameIndex>::const_iterator it = support_foot_ids.begin(); it != support_foot_ids.end();
       ++it) {
    crocoddyl::FrameTranslation xref(*it, Eigen::Vector3d::Zero());
    boost::shared_ptr<crocoddyl::ContactModelAbstract> support_contact_model =
        boost::make_shared<crocoddyl::ContactModel3D>(state_, xref, actuation_->get_nu(), Eigen::Vector2d(0., 50.));
    contact_model->addContact(rmodel_.frames[*it].name + "_contact", support_contact_model);
  }

  // Creating the cost model for a contact phase
  boost::shared_ptr<crocoddyl::CostModelSum> cost_model =
      boost::make_shared<crocoddyl::CostModelSum>(state_, actuation_->get_nu());
  if (!foot_swing_task.empty()) {
    for (std::vector<crocoddyl::FramePlacement>::const_iterator it = foot_swing_task.begin();
         it != foot_swing_task.end(); ++it) {
      crocoddyl::FrameTranslation xref(it->id, it->placement.translation());
      crocoddyl::FrameMotion vref(it->id, pinocchio::Motion::Zero());
      boost::shared_ptr<crocoddyl::CostModelAbstract> foot_track =
          boost::make_shared<crocoddyl::CostModelFrameTranslation>(state_, xref, actuation_->get_nu());
      boost::shared_ptr<crocoddyl::CostModelAbstract> impulse_foot_vel =
          boost::make_shared<crocoddyl::CostModelFrameVelocity>(state_, vref, actuation_->get_nu());
      cost_model->addCost(rmodel_.frames[it->id].name + "_footTrack", foot_track, 1e7);
      cost_model->addCost(rmodel_.frames[it->id].name + "_impulseVel", impulse_foot_vel, 1e6);
    }
  }
  Eigen::VectorXd state_weights(2 * rmodel_.nv);
  state_weights.head<3>().fill(0.);
  state_weights.segment<3>(3).fill(pow(500., 2));
  state_weights.segment(6, rmodel_.nv - 6).fill(pow(0.01, 2));
  state_weights.segment(rmodel_.nv, rmodel_.nv).fill(pow(10., 2));
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> state_activation =
      boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(state_weights);
  boost::shared_ptr<crocoddyl::CostModelAbstract> state_reg =
      boost::make_shared<crocoddyl::CostModelState>(state_, state_activation, defaultstate_, actuation_->get_nu());
  boost::shared_ptr<crocoddyl::CostModelAbstract> ctrl_reg =
      boost::make_shared<crocoddyl::CostModelControl>(state_, actuation_->get_nu());
  cost_model->addCost("stateReg", state_reg, 1e1);
  cost_model->addCost("ctrlReg", ctrl_reg, 1e-3);

  // Creating the action model for the KKT dynamics with simpletic Euler integration scheme
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> dmodel =
      boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state_, actuation_, contact_model,
                                                                               cost_model);
  return boost::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodel, 0.);
}

boost::shared_ptr<ActionModelAbstract> SimpleQuadrupedGaitProblem::createImpulseModel(
    const std::vector<pinocchio::FrameIndex>& support_foot_ids, const std::vector<FramePlacement>& foot_swing_task) {
  // Creating a 3D multi-contact model, and then including the supporting foot
  boost::shared_ptr<crocoddyl::ImpulseModelMultiple> impulse_model =
      boost::make_shared<crocoddyl::ImpulseModelMultiple>(state_);
  for (std::vector<pinocchio::FrameIndex>::const_iterator it = support_foot_ids.begin(); it != support_foot_ids.end();
       ++it) {
    boost::shared_ptr<crocoddyl::ImpulseModelAbstract> support_contact_model =
        boost::make_shared<crocoddyl::ImpulseModel3D>(state_, *it);
    impulse_model->addImpulse(rmodel_.frames[*it].name + "_impulse", support_contact_model);
  }

  // Creating the cost model for a contact phase
  boost::shared_ptr<crocoddyl::CostModelSum> cost_model = boost::make_shared<crocoddyl::CostModelSum>(state_, 0);
  if (!foot_swing_task.empty()) {
    for (std::vector<crocoddyl::FramePlacement>::const_iterator it = foot_swing_task.begin();
         it != foot_swing_task.end(); ++it) {
      crocoddyl::FrameTranslation xref(it->id, it->placement.translation());
      boost::shared_ptr<crocoddyl::CostModelAbstract> foot_track =
          boost::make_shared<crocoddyl::CostModelFrameTranslation>(state_, xref, 0);
      cost_model->addCost(rmodel_.frames[it->id].name + "_footTrack", foot_track, 1e7);
    }
  }
  Eigen::VectorXd state_weights(2 * rmodel_.nv);
  state_weights.head<6>().fill(1.);
  state_weights.segment(6, rmodel_.nv - 6).fill(pow(10., 2));
  state_weights.segment(rmodel_.nv, rmodel_.nv).fill(pow(10., 2));
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> state_activation =
      boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(state_weights);
  boost::shared_ptr<crocoddyl::CostModelAbstract> state_reg =
      boost::make_shared<crocoddyl::CostModelState>(state_, state_activation, defaultstate_, 0);
  cost_model->addCost("stateReg", state_reg, 1e1);

  // Creating the action model for the KKT dynamics with simpletic Euler integration scheme
  return boost::make_shared<crocoddyl::ActionModelImpulseFwdDynamics>(state_, impulse_model, cost_model);
}

const Eigen::VectorXd& SimpleQuadrupedGaitProblem::get_defaultState() const { return defaultstate_; }

}  // namespace crocoddyl
