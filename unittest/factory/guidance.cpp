///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "guidance.hpp"

#include "../random_generator.hpp"
#include "crocoddyl/core/guidance/componentwise-saturation.hpp"
#include "crocoddyl/core/guidance/linear.hpp"
#include "crocoddyl/core/guidance/smooth-saturation.hpp"

namespace crocoddyl {
namespace unittest {

const std::vector<GuidanceModelTypes::Type> GuidanceModelTypes::all(
    GuidanceModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, GuidanceModelTypes::Type type) {
  switch (type) {
    case GuidanceModelTypes::GuidanceModelLinear:
      os << "GuidanceModelLinear";
      break;
    case GuidanceModelTypes::GuidanceModelSmoothSaturation:
      os << "GuidanceModelSmoothSaturation";
      break;
    case GuidanceModelTypes::GuidanceModelComponentwiseSaturation:
      os << "GuidanceModelComponentwiseSaturation";
      break;
    case GuidanceModelTypes::NbGuidanceModelTypes:
      os << "NbGuidanceModelTypes";
      break;
    default:
      break;
  }
  return os;
}

GuidanceModelFactory::GuidanceModelFactory() {}
GuidanceModelFactory::~GuidanceModelFactory() {}

std::shared_ptr<crocoddyl::GuidanceModelAbstract> GuidanceModelFactory::create(
    GuidanceModelTypes::Type guidance_type, std::size_t nr) const {
  std::shared_ptr<crocoddyl::GuidanceModelAbstract> guidance;
  Eigen::VectorXd gain = random_vector<double>(nr, 0.1, 2.0);
  Eigen::VectorXd max_rate = random_vector<double>(nr, 0.1, 2.0);
  Eigen::MatrixXd gain_matrix = random_matrix<double>(nr, nr, -1.0, 1.0);
  double scalar_gain = random_real_in_range<double>(0.1, 2.0);
  double max_rate_scalar = random_real_in_range<double>(0.1, 2.0);
  double epsilon = random_real_in_range<double>(1e-6, 1e-2);

  switch (guidance_type) {
    case GuidanceModelTypes::GuidanceModelLinear:
      guidance = std::make_shared<crocoddyl::GuidanceModelLinear>(gain_matrix);
      break;
    case GuidanceModelTypes::GuidanceModelSmoothSaturation:
      guidance = std::make_shared<crocoddyl::GuidanceModelSmoothSaturation>(
          nr, scalar_gain, max_rate_scalar, epsilon);
      break;
    case GuidanceModelTypes::GuidanceModelComponentwiseSaturation:
      guidance =
          std::make_shared<crocoddyl::GuidanceModelComponentwiseSaturation>(
              gain, max_rate);
      break;
    default:
      throw_pretty(__FILE__ ":\n Construct wrong GuidanceModelTypes::Type");
      break;
  }
  return guidance;
}

}  // namespace unittest
}  // namespace crocoddyl
