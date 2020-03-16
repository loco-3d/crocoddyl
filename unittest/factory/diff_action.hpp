///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iterator>
#include <pinocchio/fwd.hpp>

#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/actions/diff-lqr.hpp"
#include "crocoddyl/core/numdiff/diff-action.hpp"
#include "crocoddyl/core/utils/exception.hpp"

#ifndef CROCODDYL_ACTION_FACTORY_HPP_
#define CROCODDYL_ACTION_FACTORY_HPP_

namespace crocoddyl {
namespace unittest {

struct DifferentialActionModelTypes {
  enum Type { DifferentialActionModelLQR, DifferentialActionModelLQRDriftFree, NbDifferentialActionModelTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbDifferentialActionModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};
const std::vector<DifferentialActionModelTypes::Type> DifferentialActionModelTypes::all(DifferentialActionModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, DifferentialActionModelTypes::Type type) {
  switch (type) {
    case DifferentialActionModelTypes::DifferentialActionModelLQR:
      os << "DifferentialActionModelLQR";
      break;
    case DifferentialActionModelTypes::DifferentialActionModelLQRDriftFree:
      os << "DifferentialActionModelLQRDriftFree";
      break;
    case DifferentialActionModelTypes::NbDifferentialActionModelTypes:
      os << "NbDifferentialActionModelTypes";
      break;
    default:
      break;
  }
  return os;
}

class DifferentialActionModelFactory {
 public:
  DifferentialActionModelFactory(DifferentialActionModelTypes::Type type) {
    switch (type) {
      case DifferentialActionModelTypes::DifferentialActionModelLQR:
        nq_ = 40;
        nu_ = 40;
        action_ = boost::make_shared<crocoddyl::DifferentialActionModelLQR>(nq_, nu_, false);
        break;
      case DifferentialActionModelTypes::DifferentialActionModelLQRDriftFree:
        nq_ = 40;
        nu_ = 40;
        action_ = boost::make_shared<crocoddyl::DifferentialActionModelLQR>(nq_, nu_, true);
        break;
      default:
        throw_pretty(__FILE__ ": Wrong DifferentialActionModelTypes::Type given");
        break;
    }
  }

  ~DifferentialActionModelFactory() {}

  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> create() { return action_; }

  const std::size_t& get_nq() { return nq_; }

 private:
  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> action_;
  std::size_t nq_;
  std::size_t nu_;
};

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_ACTION_FACTORY_HPP_
