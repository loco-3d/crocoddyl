///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2026, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <boost/test/unit_test.hpp>

#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/multibody/dynamics/impulse-forward.hpp"
#include "crocoddyl/multibody/implicit-constraints/contact.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/residuals/impulse-com.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "factory/state.hpp"
#include "unittest_common.hpp"

BOOST_AUTO_TEST_CASE(test_generic_impulse_constraint_collector) {
  const std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(
          crocoddyl::unittest::StateModelFactory().create(
              crocoddyl::unittest::StateModelTypes::StateMultibody_TalosArm));
  const std::shared_ptr<pinocchio::Model> pinocchio = state->get_pinocchio();
  const std::shared_ptr<crocoddyl::ImplicitConstraintModelMultiple>
      constraints =
          std::make_shared<crocoddyl::ImplicitConstraintModelMultiple>(state,
                                                                       0);
  const pinocchio::FrameIndex id = pinocchio->frames.size() - 1;
  crocoddyl::ContactModel::MaskArray mask = {
      {true, true, true, false, false, false}};
  constraints->addConstraint(
      "contact", std::make_shared<crocoddyl::ContactModel>(
                     state, id, pinocchio->frames[id].placement,
                     pinocchio::LOCAL_WORLD_ALIGNED, 0,
                     crocoddyl::ContactModel::Vector2s::Zero(), mask));
  crocoddyl::DynamicsModelImpulseForward dynamics(state, constraints);
  const std::shared_ptr<crocoddyl::DynamicsDataAbstract> dynamics_data =
      dynamics.createData();
  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u(0);
  dynamics.calc(dynamics_data, x, u);
  dynamics.calcDiff(dynamics_data, x, u);

  const std::shared_ptr<crocoddyl::ResidualModelImpulseCoM> residual =
      std::make_shared<crocoddyl::ResidualModelImpulseCoM>(state);
  crocoddyl::ConstraintModelResidual constraint(state, residual);
  const std::shared_ptr<crocoddyl::ConstraintDataAbstract> data =
      constraint.createData(dynamics_data->shared);
  constraint.calc(data, x, u);
  constraint.calcDiff(data, x, u);
  BOOST_CHECK_EQUAL(data->h.size(), 3);
  BOOST_CHECK(data->Hx.allFinite());
}
