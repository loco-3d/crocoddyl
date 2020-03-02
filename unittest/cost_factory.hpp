///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/centroidal-derivatives.hpp>

// #include "crocoddyl/multibody/costs/centroidal-momentum.hpp"
#include "crocoddyl/multibody/costs/com-position.hpp"
#include "crocoddyl/multibody/costs/contact-force.hpp"
#include "crocoddyl/multibody/costs/control.hpp"
// #include "crocoddyl/multibody/costs/cost-sum.hpp"
// #include "crocoddyl/multibody/costs/frame-force.hpp"
#include "crocoddyl/multibody/costs/frame-placement.hpp"
#include "crocoddyl/multibody/costs/frame-rotation.hpp"
#include "crocoddyl/multibody/costs/frame-translation.hpp"
#include "crocoddyl/multibody/costs/frame-velocity.hpp"
// #include "crocoddyl/multibody/costs/impulse.hpp"
#include "crocoddyl/multibody/costs/state.hpp"
#include "crocoddyl/multibody/numdiff/cost.hpp"

#include "state_factory.hpp"
#include "activation_factory.hpp"

#ifndef CROCODDYL_COST_FACTORY_HPP_
#define CROCODDYL_COST_FACTORY_HPP_

namespace crocoddyl_unit_test {

struct CostModelTypes {
  enum Type {
    // CostModelCentroidalMomentum, // @todo Figure out the pinocchio callbacks.
    CostModelCoMPosition,
    // CostModelContactForce, // @todo Figure out the contacts creations.
    CostModelControl,
    // CostModelSum, // @todo Implement a separate unittests for this one?
    // CostModelFrameForce, // @todo Implement the CostModelFrameForce class.
    CostModelFramePlacement,
    CostModelFrameRotation,
    CostModelFrameTranslation,
    CostModelFrameVelocity,
    // CostModelImpulse,  // @todo Implement the CostModelImpulses class.
    CostModelState,
    NbCostModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbCostModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};
const std::vector<CostModelTypes::Type> CostModelTypes::all(CostModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, CostModelTypes::Type type) {
  switch (type) {
    // case CostModelCentroidalMomentum : os << "CostModelCentroidalMomentum"; break;
    case CostModelTypes::CostModelCoMPosition:
      os << "CostModelCoMPosition";
      break;
    // case CostModelContactForce : os << "CostModelContactForce"; break;
    case CostModelTypes::CostModelControl:
      os << "CostModelControl";
      break;
    // case CostModelSum : os << "CostModelSum"; break;
    // case CostModelFrameForce : os << "CostModelFrameForce"; break;
    case CostModelTypes::CostModelFramePlacement:
      os << "CostModelFramePlacement";
      break;
    case CostModelTypes::CostModelFrameRotation:
      os << "CostModelFrameRotation";
      break;
    case CostModelTypes::CostModelFrameTranslation:
      os << "CostModelFrameTranslation";
      break;
    case CostModelTypes::CostModelFrameVelocity:
      os << "CostModelFrameVelocity";
      break;
    // case CostModelImpulse : os << "CostModelImpulse"; break;
    case CostModelTypes::CostModelState:
      os << "CostModelState";
      break;
    case CostModelTypes::NbCostModelTypes:
      os << "NbCostModelTypes";
      break;
    default:
      break;
  }
  return os;
}

class CostModelFactory {
 public:
  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef typename MathBase::Vector6s Vector6d;

  CostModelFactory(CostModelTypes::Type test_type, ActivationModelTypes::Type activation_type,
                   StateTypes::Type state_multibody_type)
      : state_factory_(state_multibody_type),
        state_multibody_(boost::static_pointer_cast<crocoddyl::StateMultibody>(state_factory_.get_state())),
        // Setup some reference for the costs.
        frame_index_(state_multibody_->get_pinocchio().frames.size() - 1),
        mom_ref_(Vector6d::Random()),
        com_ref_(Eigen::Vector3d::Random()),
        force_ref_(frame_index_, pinocchio::Force(Vector6d::Random())),
        u_ref_(Eigen::VectorXd::Random(state_multibody_->get_nv())),
        frame_(pinocchio::SE3::Random()),
        frame_ref_(frame_index_, frame_),
        rotation_ref_(frame_index_, frame_.rotation()),
        translation_ref_(frame_index_, frame_.translation()),
        velocity_ref_(frame_index_, pinocchio::Motion::Random()) {
    num_diff_modifier_ = 1e4;
    type_ = test_type;

    // Construct the different cost.
    switch (type_) {
      // case CostModelTypes::CostModelCentroidalMomentum:
      //   activation_factory_ = boost::make_shared<ActivationModelFactory>(activation_type, 6);
      //   cost_ = boost::make_shared<crocoddyl::CostModelCentroidalMomentum>(
      //       state_multibody_, activation_factory_->get_activation(), mom_ref_);
      //   break;
      case CostModelTypes::CostModelCoMPosition:
        activation_factory_ = boost::make_shared<ActivationModelFactory>(activation_type, 3);
        cost_ = boost::make_shared<crocoddyl::CostModelCoMPosition>(state_multibody_,
                                                                    activation_factory_->get_activation(), com_ref_);
        break;
      // case CostModelTypes::CostModelContactForce:
      //   activation_factory_ = boost::make_shared<ActivationModelFactory>(activation_type, 6);
      //   cost_ = boost::make_shared<crocoddyl::CostModelContactForce>(
      //       state_multibody_, activation_factory_->get_activation(), force_ref_);
      //   break;
      case CostModelTypes::CostModelControl:
        activation_factory_ = boost::make_shared<ActivationModelFactory>(activation_type, state_multibody_->get_nv());
        cost_ = boost::make_shared<crocoddyl::CostModelControl>(state_multibody_,
                                                                activation_factory_->get_activation(), u_ref_);
        break;
      // case CostModelTypes::CostModelSum:
      //   break;
      // case CostModelTypes::CostModelFrameForce:
      //   break;
      case CostModelTypes::CostModelFramePlacement:
        activation_factory_ = boost::make_shared<ActivationModelFactory>(activation_type, 6);
        cost_ = boost::make_shared<crocoddyl::CostModelFramePlacement>(
            state_multibody_, activation_factory_->get_activation(), frame_ref_);
        break;
      case CostModelTypes::CostModelFrameRotation:
        activation_factory_ = boost::make_shared<ActivationModelFactory>(activation_type, 3);
        cost_ = boost::make_shared<crocoddyl::CostModelFrameRotation>(
            state_multibody_, activation_factory_->get_activation(), rotation_ref_);
        break;
      case CostModelTypes::CostModelFrameTranslation:
        activation_factory_ = boost::make_shared<ActivationModelFactory>(activation_type, 3);
        cost_ = boost::make_shared<crocoddyl::CostModelFrameTranslation>(
            state_multibody_, activation_factory_->get_activation(), translation_ref_);
        break;
      case CostModelTypes::CostModelFrameVelocity:
        activation_factory_ = boost::make_shared<ActivationModelFactory>(activation_type, 6);
        cost_ = boost::make_shared<crocoddyl::CostModelFrameVelocity>(
            state_multibody_, activation_factory_->get_activation(), velocity_ref_);
        break;
      // case CostModelTypes::CostModelImpulse:
      //   break;
      case CostModelTypes::CostModelState:
        activation_factory_ = boost::make_shared<ActivationModelFactory>(activation_type, state_multibody_->get_nx());
        cost_ = boost::make_shared<crocoddyl::CostModelState>(state_multibody_, activation_factory_->get_activation(),
                                                              state_multibody_->rand());
        break;
      default:
        throw_pretty(__FILE__ ": Wrong CostModelTypes::Type given");
        break;
    }
  }

  ~CostModelFactory() {}

  boost::shared_ptr<crocoddyl::CostModelAbstract> get_cost() { return cost_; }
  double get_num_diff_modifier() { return num_diff_modifier_; }

 private:
  double num_diff_modifier_;
  std::size_t nu_;
  CostModelTypes::Type type_;
  boost::shared_ptr<crocoddyl::CostModelAbstract> cost_;
  boost::shared_ptr<ActivationModelFactory> activation_factory_;
  StateFactory state_factory_;
  boost::shared_ptr<crocoddyl::StateMultibody> state_multibody_;

  // some reference:
  crocoddyl::FrameIndex frame_index_;
  Vector6d mom_ref_;
  Eigen::Vector3d com_ref_;
  crocoddyl::FrameForce force_ref_;
  Eigen::VectorXd u_ref_;
  pinocchio::SE3 frame_;
  crocoddyl::FramePlacement frame_ref_;
  crocoddyl::FrameRotation rotation_ref_;
  crocoddyl::FrameTranslation translation_ref_;
  crocoddyl::FrameMotion velocity_ref_;
};

/**
 * @brief Compute all the pinocchio data needed for the numerical
 * differentiation. We use the address of the object to avoid a copy from the
 * "boost::bind".
 *
 * @param model is the rigid body robot model.
 * @param data contains the results of the computations.
 * @param x is the state vector.
 */
void updateAllPinocchio(pinocchio::Model* const model, pinocchio::Data* data, const Eigen::VectorXd& x) {
  const Eigen::VectorXd& q = x.segment(0, model->nq);
  const Eigen::VectorXd& v = x.segment(model->nq, model->nv);
  Eigen::VectorXd a = Eigen::VectorXd::Zero(model->nv);
  Eigen::Matrix<double, 6, Eigen::Dynamic> tmp;
  tmp.resize(6, model->nv);
  pinocchio::forwardKinematics(*model, *data, q);
  pinocchio::computeJointJacobians(*model, *data, q);
  pinocchio::updateFramePlacements(*model, *data);
  pinocchio::jacobianCenterOfMass(*model, *data, q);
  pinocchio::computeCentroidalMomentum(*model, *data, q, v);
  pinocchio::computeCentroidalDynamicsDerivatives(*model, *data, q, v, a, tmp, tmp, tmp, tmp);
}

}  // namespace crocoddyl_unit_test

#endif  // CROCODDYL_COST_FACTORY_HPP_
