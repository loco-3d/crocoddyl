///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2026, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <boost/test/unit_test.hpp>

#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/dynamics/constrained-forward.hpp"
#include "crocoddyl/multibody/implicit-constraints/contact.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "factory/state.hpp"
#include "unittest_common.hpp"

BOOST_AUTO_TEST_CASE(test_generic_contact_cost_collector) {
  const std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(
          crocoddyl::unittest::StateModelFactory().create(
              crocoddyl::unittest::StateModelTypes::StateMultibody_TalosArm));
  const std::shared_ptr<pinocchio::Model> pinocchio = state->get_pinocchio();
  const std::shared_ptr<crocoddyl::ActuationModelMultibody> actuation =
      std::make_shared<crocoddyl::ActuationModelMultibody>(state);
  const std::shared_ptr<crocoddyl::ImplicitConstraintModelMultiple>
      constraints =
          std::make_shared<crocoddyl::ImplicitConstraintModelMultiple>(
              state, actuation->get_nu());
  const pinocchio::FrameIndex id = pinocchio->frames.size() - 1;
  crocoddyl::ContactModel::MaskArray mask = {
      {true, true, true, false, false, false}};
  constraints->addConstraint(
      "contact", std::make_shared<crocoddyl::ContactModel>(
                     state, id, pinocchio->frames[id].placement,
                     pinocchio::LOCAL_WORLD_ALIGNED, actuation->get_nu(),
                     crocoddyl::ContactModel::Vector2s::Zero(), mask));
  crocoddyl::DynamicsModelConstrainedForward dynamics(state, actuation,
                                                      constraints);
  const std::shared_ptr<crocoddyl::DynamicsDataAbstract> dynamics_data =
      dynamics.createData();
  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(dynamics.get_nu());
  dynamics.calc(dynamics_data, x, u);
  dynamics.calcDiff(dynamics_data, x, u);

  const std::shared_ptr<crocoddyl::ResidualModelContactForce> residual =
      std::make_shared<crocoddyl::ResidualModelContactForce>(
          state, id, pinocchio::Force::Zero(), 3, dynamics.get_nu());
  crocoddyl::CostModelResidual cost(state, residual);
  const std::shared_ptr<crocoddyl::CostDataAbstract> data =
      cost.createData(dynamics_data->shared);
  cost.calc(data, x, u);
  cost.calcDiff(data, x, u);
  BOOST_CHECK_EQUAL(data->residual->r.size(), 3);
  BOOST_CHECK(data->Lx.allFinite());
  BOOST_CHECK(data->Lu.allFinite());
}
