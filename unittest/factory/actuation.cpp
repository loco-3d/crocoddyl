///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "actuation.hpp"

#include "crocoddyl/core/actuation/actuation-squashing.hpp"
#include "crocoddyl/core/actuation/squashing-base.hpp"
#include "crocoddyl/core/actuation/squashing/smooth-sat.hpp"
#include "crocoddyl/multibody/actuations/floating-base-thrusters.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<ActuationModelTypes::Type> ActuationModelTypes::all(
    ActuationModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ActuationModelTypes::Type type) {
  switch (type) {
    case ActuationModelTypes::ActuationModelFull:
      os << "ActuationModelFull";
      break;
    case ActuationModelTypes::ActuationModelFloatingBase:
      os << "ActuationModelFloatingBase";
      break;
    case ActuationModelTypes::ActuationModelFloatingBaseThrusters:
      os << "ActuationModelFloatingBaseThrusters";
      break;
    case ActuationModelTypes::ActuationModelSquashingFull:
      os << "ActuationModelSquashingFull";
      break;
    case ActuationModelTypes::NbActuationModelTypes:
      os << "NbActuationModelTypes";
      break;
    default:
      break;
  }
  return os;
}

ActuationModelFactory::ActuationModelFactory() {}
ActuationModelFactory::~ActuationModelFactory() {}

std::shared_ptr<crocoddyl::ActuationModelAbstract>
ActuationModelFactory::create(ActuationModelTypes::Type actuation_type,
                              StateModelTypes::Type state_type) const {
  std::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
  StateModelFactory factory;
  std::shared_ptr<crocoddyl::StateAbstract> state = factory.create(state_type);
  std::shared_ptr<crocoddyl::StateMultibody> state_multibody;
  // Thruster objects
  std::vector<crocoddyl::Thruster> ps;
  const double d_cog = 0.1525;
  const double cf = 6.6e-5;
  const double cm = 1e-6;
  pinocchio::SE3 p1(Eigen::Matrix3d::Identity(), Eigen::Vector3d(d_cog, 0, 0));
  pinocchio::SE3 p2(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, d_cog, 0));
  pinocchio::SE3 p3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(-d_cog, 0, 0));
  pinocchio::SE3 p4(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, -d_cog, 0));
  ps.push_back(crocoddyl::Thruster(p1, cm / cf, crocoddyl::ThrusterType::CCW));
  ps.push_back(crocoddyl::Thruster(p2, cm / cf, crocoddyl::ThrusterType::CW));
  ps.push_back(crocoddyl::Thruster(p3, cm / cf, crocoddyl::ThrusterType::CW));
  ps.push_back(crocoddyl::Thruster(p4, cm / cf, crocoddyl::ThrusterType::CCW));
  // Actuation Squashing objects
  std::shared_ptr<crocoddyl::ActuationModelAbstract> act;
  std::shared_ptr<crocoddyl::SquashingModelSmoothSat> squash;
  Eigen::VectorXd lb;
  Eigen::VectorXd ub;
  switch (actuation_type) {
    case ActuationModelTypes::ActuationModelFull:
      state_multibody =
          std::static_pointer_cast<crocoddyl::StateMultibody>(state);
      actuation =
          std::make_shared<crocoddyl::ActuationModelFull>(state_multibody);
      break;
    case ActuationModelTypes::ActuationModelFloatingBase:
      state_multibody =
          std::static_pointer_cast<crocoddyl::StateMultibody>(state);
      actuation = std::make_shared<crocoddyl::ActuationModelFloatingBase>(
          state_multibody);
      break;
    case ActuationModelTypes::ActuationModelFloatingBaseThrusters:
      state_multibody =
          std::static_pointer_cast<crocoddyl::StateMultibody>(state);
      actuation =
          std::make_shared<crocoddyl::ActuationModelFloatingBaseThrusters>(
              state_multibody, ps);
      break;
    case ActuationModelTypes::ActuationModelSquashingFull:
      state_multibody =
          std::static_pointer_cast<crocoddyl::StateMultibody>(state);

      act = std::make_shared<crocoddyl::ActuationModelFull>(state_multibody);

      lb = Eigen::VectorXd::Zero(state->get_nv());
      ub = Eigen::VectorXd::Zero(state->get_nv());
      lb.fill(-100.0);
      ub.fill(100.0);
      squash = std::make_shared<crocoddyl::SquashingModelSmoothSat>(
          lb, ub, state->get_nv());

      actuation = std::make_shared<crocoddyl::ActuationSquashingModel>(
          act, squash, state->get_nv());
      break;
    default:
      throw_pretty(__FILE__ ":\n Construct wrong ActuationModelTypes::Type");
      break;
  }
  return actuation;
}

}  // namespace unittest
}  // namespace crocoddyl
