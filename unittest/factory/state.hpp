///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/sample-models.hpp>
#include <example-robot-data/path.hpp>

#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/numdiff/state.hpp"
#include "crocoddyl/core/utils/exception.hpp"

#include "pinocchio_model.hpp"

#ifndef CROCODDYL_STATE_FACTORY_HPP_
#define CROCODDYL_STATE_FACTORY_HPP_

namespace crocoddyl {
namespace unittest {

struct StateModelTypes {
  enum Type { StateVector, StateMultibody, NbStateModelTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbStateModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};
const std::vector<StateModelTypes::Type> StateModelTypes::all(StateModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, StateModelTypes::Type type) {
  switch (type) {
    case StateModelTypes::StateVector:
      os << "StateVector";
      break;
    case StateModelTypes::StateMultibody:
      os << "StateMultibody";
      break;
      os << "NbStateModelTypes";
      break;
    default:
      break;
  }
  return os;
}

class StateModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  StateModelFactory(StateModelTypes::Type state_type,
                    PinocchioModelTypes::Type model_type = PinocchioModelTypes::NbPinocchioModelTypes) {
    nx_ = 0;
    PinocchioModelFactory factory(model_type);
    boost::shared_ptr<pinocchio::Model> model;

    switch (state_type) {
      case StateModelTypes::StateVector:
        nx_ = 80;
        state_ = boost::make_shared<crocoddyl::StateVector>(nx_);
        break;
      case StateModelTypes::StateMultibody:
        model = factory.create();
        nx_ = model->nq + model->nv;
        state_ = boost::make_shared<crocoddyl::StateMultibody>(model);
        break;
      default:
        throw_pretty(__FILE__ ": Wrong StateModelTypes::Type given");
        break;
    }
  }

  ~StateModelFactory() {}

  boost::shared_ptr<crocoddyl::StateAbstract> create() { return state_; }
  const std::size_t& get_nx() { return nx_; }

 private:
  boost::shared_ptr<crocoddyl::StateAbstract> state_;  //!< The pointer to the state in testing
  std::size_t nx_;                                     //!< The size of the StateVector to test.
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_STATE_FACTORY_HPP_
