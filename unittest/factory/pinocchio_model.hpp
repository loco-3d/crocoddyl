///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_PINOCCHIO_MODEL_FACTORY_HPP_
#define CROCODDYL_PINOCCHIO_MODEL_FACTORY_HPP_

#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/centroidal-derivatives.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/sample-models.hpp>

#include <example-robot-data/path.hpp>

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

struct PinocchioModelTypes {
  enum Type { TalosArm, HyQ, Talos, RandomHumanoid, Hector, NbPinocchioModelTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.clear();
    for (int i = 0; i < NbPinocchioModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, PinocchioModelTypes::Type type);

class PinocchioModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PinocchioModelFactory(PinocchioModelTypes::Type type);
  ~PinocchioModelFactory();

  void construct_model(const std::string& urdf_file = "", const std::string& srdf_file = "", bool free_flyer = true);

  boost::shared_ptr<pinocchio::Model> create() const;
  const std::string& get_frame_name() const;
  const std::size_t& get_frame_id() const;

 private:
  boost::shared_ptr<pinocchio::Model> model_;  //!< The pointer to the state in testing
  std::string frame_name_;                     //!< Frame name for unittesting
  std::size_t frame_id_;                       //!< Frame id for unittesting
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
void updateAllPinocchio(pinocchio::Model* const model, pinocchio::Data* data, const Eigen::VectorXd& x);

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_PINOCCHIO_MODEL_FACTORY_HPP_
