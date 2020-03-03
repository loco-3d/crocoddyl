///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"
#include "crocoddyl/core/activations/smooth-abs.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/activations/weighted-quadratic-barrier.hpp"
#include "crocoddyl/core/numdiff/activation.hpp"

#ifndef CROCODDYL_ACTIVATION_FACTORY_HPP_
#define CROCODDYL_ACTIVATION_FACTORY_HPP_

namespace crocoddyl_unit_test {

struct ActivationModelTypes {
  enum Type {
    ActivationModelQuad,
    ActivationModelSmoothAbs,
    ActivationModelWeightedQuad,
    ActivationModelQuadraticBarrier,
    ActivationModelWeightedQuadraticBarrier,
    NbActivationModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbActivationModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};
const std::vector<ActivationModelTypes::Type> ActivationModelTypes::all(ActivationModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ActivationModelTypes::Type type) {
  switch (type) {
    case ActivationModelTypes::ActivationModelQuad:
      os << "ActivationModelQuad";
      break;
    case ActivationModelTypes::ActivationModelSmoothAbs:
      os << "ActivationModelSmoothAbs";
      break;
    case ActivationModelTypes::ActivationModelWeightedQuad:
      os << "ActivationModelWeightedQuad";
      break;
    case ActivationModelTypes::ActivationModelQuadraticBarrier:
      os << "ActivationModelQuadraticBarrier";
      break;
    case ActivationModelTypes::ActivationModelWeightedQuadraticBarrier:
      os << "ActivationModelWeightedQuadraticBarrier";
      break;
    case ActivationModelTypes::NbActivationModelTypes:
      os << "NbActivationModelTypes";
      break;
    default:
      break;
  }
  return os;
}

class ActivationModelFactory {
 public:
  ActivationModelFactory(ActivationModelTypes::Type test_type, std::size_t nr = 5) {
    nr_ = nr;
    num_diff_modifier_ = 1e4;
    Eigen::VectorXd lb = Eigen::VectorXd::Random(nr_);
    Eigen::VectorXd ub = lb + Eigen::VectorXd::Ones(nr_) + Eigen::VectorXd::Random(nr_);
    Eigen::VectorXd weights = Eigen::VectorXd::Random(nr_);

    switch (test_type) {
      case ActivationModelTypes::ActivationModelQuad:
        activation_ = boost::make_shared<crocoddyl::ActivationModelQuad>(nr_);
        break;
      case ActivationModelTypes::ActivationModelSmoothAbs:
        activation_ = boost::make_shared<crocoddyl::ActivationModelSmoothAbs>(nr_);
        break;
      case ActivationModelTypes::ActivationModelWeightedQuad:
        activation_ = boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(weights);
        break;
      case ActivationModelTypes::ActivationModelQuadraticBarrier:
        activation_ =
            boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(crocoddyl::ActivationBounds(lb, ub));
        break;
      case ActivationModelTypes::ActivationModelWeightedQuadraticBarrier:
        activation_ = boost::make_shared<crocoddyl::ActivationModelWeightedQuadraticBarrier>(
            crocoddyl::ActivationBounds(lb, ub), weights);
        break;
      default:
        throw_pretty(__FILE__ ":\n Construct wrong ActivationModelTypes::Type");
        break;
    }
  }

  ~ActivationModelFactory() {}

  boost::shared_ptr<crocoddyl::ActivationModelAbstract> create() { return activation_; }
  const std::size_t& get_nr() { return nr_; }
  double get_num_diff_modifier() { return num_diff_modifier_; }

 private:
  double num_diff_modifier_;
  std::size_t nr_;
  boost::shared_ptr<crocoddyl::ActivationModelAbstract> activation_;
};

}  // namespace crocoddyl_unit_test

#endif  // CROCODDYL_ACTIVATION_FACTORY_HPP_
