///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/multibody/actuations/full.hpp"
#include "crocoddyl/multibody/data/implicit-constraints.hpp"
#include "crocoddyl/multibody/implicit-constraints/contact.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/residuals/contact-control-gravity.hpp"
#include "crocoddyl/multibody/residuals/contact-cop-position.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/multibody/residuals/impulse-com.hpp"
#include "factory/pinocchio_model.hpp"
#include "factory/state.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

namespace {

template <typename Scalar>
void test_generic_residual_collectors() {
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef crocoddyl::ContactModelTpl<Scalar> Contact;
  typedef crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar> Multiple;
  typedef crocoddyl::ImplicitConstraintDataMultipleTpl<Scalar> MultipleData;
  typedef crocoddyl::ActuationModelFullTpl<Scalar> Actuation;
  typedef crocoddyl::ActuationDataAbstractTpl<Scalar> ActuationData;
  typedef crocoddyl::ResidualDataAbstractTpl<Scalar> ResidualData;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

  const std::shared_ptr<crocoddyl::StateMultibody> state64 =
      std::static_pointer_cast<crocoddyl::StateMultibody>(
          StateModelFactory().create(StateModelTypes::StateMultibody_TalosArm));
  const std::shared_ptr<State> state =
      std::make_shared<State>(state64->template cast<Scalar>());
  const pinocchio::FrameIndex id = 2;
  const typename Contact::MaskArray mask = {
      {true, true, true, true, true, true}};
  const typename Contact::SE3 reference =
      state->get_pinocchio()->frames[id].placement;
  const std::shared_ptr<Contact> contact = std::make_shared<Contact>(
      state, id, reference, pinocchio::LOCAL_WORLD_ALIGNED, state->get_nv(),
      Contact::Vector2s::Zero(), mask);
  const std::shared_ptr<Multiple> manager =
      std::make_shared<Multiple>(state, state->get_nv());
  manager->addConstraint("contact", contact);
  pinocchio::DataTpl<Scalar> pinocchio_data(*state->get_pinocchio());
  const std::shared_ptr<MultipleData> constraints =
      manager->createData(&pinocchio_data);
  const VectorXs x = state->rand();
  const VectorXs u =
      VectorXs::LinSpaced(state->get_nv(), Scalar(-0.6), Scalar(0.8));
  crocoddyl::unittest::updateAllPinocchio(state->get_pinocchio().get(),
                                          &pinocchio_data, x);
  manager->calc(constraints, x);
  manager->calcDiff(constraints, x);
  const VectorXs force = VectorXs::LinSpaced(6, Scalar(0.4), Scalar(2.4));
  const MatrixXs df_dx =
      MatrixXs::Random(6, state->get_ndx()).template cast<Scalar>();
  const MatrixXs df_du =
      MatrixXs::Random(6, state->get_nv()).template cast<Scalar>();
  manager->updateForce(constraints, force);
  manager->updateForceDiff(constraints, df_dx, df_du);
  manager->updateVelocity(
      constraints,
      VectorXs::LinSpaced(state->get_nv(), Scalar(-0.2), Scalar(0.3)));
  manager->updateVelocityDiff(
      constraints, MatrixXs::Identity(state->get_nv(), state->get_ndx()));

  const std::shared_ptr<Actuation> actuation =
      std::make_shared<Actuation>(state);
  const std::shared_ptr<ActuationData> actuation_data = actuation->createData();
  actuation->calc(actuation_data, x, u);
  actuation->calcDiff(actuation_data, x, u);

  crocoddyl::DataCollectorImplicitConstraintTpl<Scalar> force_shared(
      constraints);
  crocoddyl::DataCollectorMultibodyInImplicitConstraintTpl<Scalar>
      multibody_shared(&pinocchio_data, constraints);
  crocoddyl::DataCollectorActMultibodyInImplicitConstraintTpl<Scalar>
      actuation_shared(&pinocchio_data, actuation_data, constraints);

  const pinocchio::ForceTpl<Scalar> zero_force =
      pinocchio::ForceTpl<Scalar>::Zero();
  crocoddyl::ResidualModelContactForceTpl<Scalar> force_residual(
      state, id, zero_force, 6, state->get_nv(), true);
  const std::shared_ptr<ResidualData> force_data =
      force_residual.createData(&force_shared);
  force_residual.calc(force_data, x, u);
  force_residual.calcDiff(force_data, x, u);
  BOOST_CHECK(force_data->r.isApprox(force));
  BOOST_CHECK(force_data->Rx.isApprox(df_dx));
  BOOST_CHECK(force_data->Ru.isApprox(df_du));

  const typename Contact::MaskArray force_masks[] = {
      {{false, false, true, false, false, false}},
      {{true, true, true, false, false, false}},
      {{true, true, true, true, true, true}}};
  const std::size_t force_dimensions[] = {1, 3, 6};
  for (std::size_t i = 0; i < 3; ++i) {
    const std::size_t nc = force_dimensions[i];
    const std::shared_ptr<Contact> dimension_contact =
        std::make_shared<Contact>(state, id, reference, pinocchio::LOCAL,
                                  state->get_nv(), Contact::Vector2s::Zero(),
                                  force_masks[i]);
    const std::shared_ptr<Multiple> dimension_manager =
        std::make_shared<Multiple>(state, state->get_nv());
    dimension_manager->addConstraint("contact", dimension_contact);
    const std::shared_ptr<MultipleData> dimension_constraints =
        dimension_manager->createData(&pinocchio_data);
    dimension_manager->calc(dimension_constraints, x);
    dimension_manager->calcDiff(dimension_constraints, x);
    const VectorXs dimension_force =
        VectorXs::LinSpaced(nc, Scalar(0.4), Scalar(2.4));
    const MatrixXs dimension_df_dx =
        MatrixXs::Random(nc, state->get_ndx()).template cast<Scalar>();
    const MatrixXs dimension_df_du =
        MatrixXs::Random(nc, state->get_nv()).template cast<Scalar>();
    dimension_manager->updateForce(dimension_constraints, dimension_force);
    dimension_manager->updateForceDiff(dimension_constraints, dimension_df_dx,
                                       dimension_df_du);
    crocoddyl::DataCollectorImplicitConstraintTpl<Scalar> dimension_shared(
        dimension_constraints);
    crocoddyl::ResidualModelContactForceTpl<Scalar> dimension_residual(
        state, id, zero_force, nc, state->get_nv(), true);
    const std::shared_ptr<ResidualData> dimension_data =
        dimension_residual.createData(&dimension_shared);
    dimension_residual.calc(dimension_data, x, u);
    dimension_residual.calcDiff(dimension_data, x, u);
    BOOST_CHECK_EQUAL(static_cast<std::size_t>(dimension_data->r.size()),
                      dimension_residual.get_nr());
    BOOST_CHECK_EQUAL(dimension_residual.get_nr(), nc);
    BOOST_CHECK(dimension_data->r.isApprox(dimension_force));
    BOOST_CHECK(dimension_data->Rx.isApprox(dimension_df_dx));
    BOOST_CHECK(dimension_data->Ru.isApprox(dimension_df_du));
    crocoddyl::ResidualModelContactForceTpl<Scalar> wrong_dimension_residual(
        state, id, zero_force, nc == 6 ? 5 : nc + 1, state->get_nv(), true);
    BOOST_CHECK_THROW(wrong_dimension_residual.createData(&dimension_shared),
                      crocoddyl::Exception);
  }

  const typename crocoddyl::MathBaseTpl<Scalar>::Matrix3s rotation =
      crocoddyl::MathBaseTpl<Scalar>::Matrix3s::Identity();
  crocoddyl::ResidualModelContactFrictionConeTpl<Scalar> friction_residual(
      state, id, crocoddyl::FrictionConeTpl<Scalar>(rotation, Scalar(0.7)),
      state->get_nv(), true);
  const std::shared_ptr<ResidualData> friction_data =
      friction_residual.createData(&force_shared);
  friction_residual.calc(friction_data, x, u);
  friction_residual.calcDiff(friction_data, x, u);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(friction_data->r.size()),
                    friction_residual.get_nr());
  BOOST_CHECK(friction_data->r.isApprox(
      friction_residual.get_reference().get_A() *
      constraints->constraints.at("contact")->f.linear()));
  BOOST_CHECK(friction_data->Rx.isApprox(
      friction_residual.get_reference().get_A() * df_dx.topRows(3)));
  BOOST_CHECK(friction_data->Ru.isApprox(
      friction_residual.get_reference().get_A() * df_du.topRows(3)));

  const typename Contact::MaskArray friction_masks[] = {
      {{true, false, true, false, false, false}},
      {{true, true, true, false, false, false}},
      {{true, true, true, true, true, true}}};
  const std::size_t friction_dimensions[] = {2, 3, 6};
  for (std::size_t i = 0; i < 3; ++i) {
    const std::size_t nc = friction_dimensions[i];
    const std::shared_ptr<Contact> dimension_contact =
        std::make_shared<Contact>(state, id, reference, pinocchio::LOCAL,
                                  state->get_nv(), Contact::Vector2s::Zero(),
                                  friction_masks[i]);
    const std::shared_ptr<Multiple> dimension_manager =
        std::make_shared<Multiple>(state, state->get_nv());
    dimension_manager->addConstraint("contact", dimension_contact);
    const std::shared_ptr<MultipleData> dimension_constraints =
        dimension_manager->createData(&pinocchio_data);
    dimension_manager->calc(dimension_constraints, x);
    dimension_manager->calcDiff(dimension_constraints, x);
    const VectorXs dimension_force =
        VectorXs::LinSpaced(nc, Scalar(0.4), Scalar(2.4));
    dimension_manager->updateForce(dimension_constraints, dimension_force);
    const MatrixXs dimension_df_dx =
        MatrixXs::Random(nc, state->get_ndx()).template cast<Scalar>();
    const MatrixXs dimension_df_du =
        MatrixXs::Random(nc, state->get_nv()).template cast<Scalar>();
    dimension_manager->updateForceDiff(dimension_constraints, dimension_df_dx,
                                       dimension_df_du);
    crocoddyl::DataCollectorImplicitConstraintTpl<Scalar> dimension_shared(
        dimension_constraints);
    crocoddyl::ResidualModelContactFrictionConeTpl<Scalar> dimension_residual(
        state, id, crocoddyl::FrictionConeTpl<Scalar>(rotation, Scalar(0.7)),
        state->get_nv(), true);
    const std::shared_ptr<ResidualData> dimension_data =
        dimension_residual.createData(&dimension_shared);
    dimension_residual.calc(dimension_data, x, u);
    dimension_residual.calcDiff(dimension_data, x, u);
    BOOST_CHECK_EQUAL(static_cast<std::size_t>(dimension_data->r.size()),
                      dimension_residual.get_nr());
    const std::shared_ptr<
        crocoddyl::ImplicitConstraintDataAbstractTpl<Scalar> >& contact_data =
        dimension_constraints->constraints.at("contact");
    BOOST_CHECK(dimension_data->r.isApprox(
        dimension_residual.get_reference().get_A() * contact_data->f.linear()));
    const typename crocoddyl::MathBaseTpl<Scalar>::MatrixX3s& A =
        dimension_residual.get_reference().get_A();
    if (nc == 2) {
      BOOST_CHECK(
          dimension_data->Rx.isApprox(A.col(0) * dimension_df_dx.row(0) +
                                      A.col(2) * dimension_df_dx.row(1)));
      BOOST_CHECK(
          dimension_data->Ru.isApprox(A.col(0) * dimension_df_du.row(0) +
                                      A.col(2) * dimension_df_du.row(1)));
    } else if (nc == 3) {
      BOOST_CHECK(dimension_data->Rx.isApprox(A * dimension_df_dx));
      BOOST_CHECK(dimension_data->Ru.isApprox(A * dimension_df_du));
    } else {
      BOOST_CHECK(dimension_data->Rx.isApprox(A * dimension_df_dx.topRows(3)));
      BOOST_CHECK(dimension_data->Ru.isApprox(A * dimension_df_du.topRows(3)));
    }
  }

  const typename Contact::MaskArray unsupported_masks[] = {
      {{true, false, false, false, false, false}},
      {{true, true, false, false, false, false}},
      {{true, true, false, true, false, false}},
      {{true, true, true, true, false, false}},
      {{true, true, true, true, true, false}}};
  const std::size_t unsupported_dimensions[] = {1, 2, 3, 4, 5};
  for (std::size_t i = 0; i < 5; ++i) {
    const std::size_t nc = unsupported_dimensions[i];
    const std::shared_ptr<Contact> unsupported_contact =
        std::make_shared<Contact>(state, id, reference, pinocchio::LOCAL,
                                  state->get_nv(), Contact::Vector2s::Zero(),
                                  unsupported_masks[i]);
    const std::shared_ptr<Multiple> unsupported_manager =
        std::make_shared<Multiple>(state, state->get_nv());
    unsupported_manager->addConstraint("contact", unsupported_contact);
    const std::shared_ptr<MultipleData> unsupported_constraints =
        unsupported_manager->createData(&pinocchio_data);
    crocoddyl::DataCollectorImplicitConstraintTpl<Scalar> unsupported_shared(
        unsupported_constraints);

    crocoddyl::ResidualModelContactForceTpl<Scalar> unsupported_force(
        state, id, zero_force, nc, state->get_nv(), true);
    BOOST_CHECK_THROW(unsupported_force.createData(&unsupported_shared),
                      crocoddyl::Exception);
    crocoddyl::ResidualModelContactFrictionConeTpl<Scalar> unsupported_friction(
        state, id, crocoddyl::FrictionConeTpl<Scalar>(rotation, Scalar(0.7)),
        state->get_nv(), true);
    BOOST_CHECK_THROW(unsupported_friction.createData(&unsupported_shared),
                      crocoddyl::Exception);
    crocoddyl::ResidualModelContactWrenchConeTpl<Scalar> unsupported_wrench(
        state, id,
        crocoddyl::WrenchConeTpl<Scalar>(
            rotation, Scalar(0.7),
            (typename crocoddyl::MathBaseTpl<Scalar>::Vector2s() << Scalar(0.1),
             Scalar(0.2))
                .finished()),
        state->get_nv(), true);
    BOOST_CHECK_THROW(unsupported_wrench.createData(&unsupported_shared),
                      crocoddyl::Exception);
    crocoddyl::ResidualModelContactCoPPositionTpl<Scalar> unsupported_cop(
        state, id,
        crocoddyl::CoPSupportTpl<Scalar>(
            rotation,
            (typename crocoddyl::MathBaseTpl<Scalar>::Vector2s() << Scalar(0.1),
             Scalar(0.2))
                .finished()),
        state->get_nv(), true);
    BOOST_CHECK_THROW(unsupported_cop.createData(&unsupported_shared),
                      crocoddyl::Exception);
  }

  crocoddyl::ResidualModelContactWrenchConeTpl<Scalar> wrench_residual(
      state, id,
      crocoddyl::WrenchConeTpl<Scalar>(
          rotation, Scalar(0.7),
          (typename crocoddyl::MathBaseTpl<Scalar>::Vector2s() << Scalar(0.1),
           Scalar(0.2))
              .finished()),
      state->get_nv(), true);
  const std::shared_ptr<ResidualData> wrench_data =
      wrench_residual.createData(&force_shared);
  wrench_residual.calc(wrench_data, x, u);
  wrench_residual.calcDiff(wrench_data, x, u);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(wrench_data->r.size()),
                    wrench_residual.get_nr());
  BOOST_CHECK(
      wrench_data->r.isApprox(wrench_residual.get_reference().get_A() * force));
  BOOST_CHECK(wrench_data->Rx.isApprox(wrench_residual.get_reference().get_A() *
                                       df_dx));
  BOOST_CHECK(wrench_data->Ru.isApprox(wrench_residual.get_reference().get_A() *
                                       df_du));

  crocoddyl::ResidualModelContactCoPPositionTpl<Scalar> cop_residual(
      state, id,
      crocoddyl::CoPSupportTpl<Scalar>(
          rotation,
          (typename crocoddyl::MathBaseTpl<Scalar>::Vector2s() << Scalar(0.1),
           Scalar(0.2))
              .finished()),
      state->get_nv(), true);
  const std::shared_ptr<ResidualData> cop_data =
      cop_residual.createData(&force_shared);
  cop_residual.calc(cop_data, x, u);
  cop_residual.calcDiff(cop_data, x, u);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(cop_data->r.size()),
                    cop_residual.get_nr());
  BOOST_CHECK(
      cop_data->r.isApprox(cop_residual.get_reference().get_A() * force));
  BOOST_CHECK(
      cop_data->Rx.isApprox(cop_residual.get_reference().get_A() * df_dx));
  BOOST_CHECK(
      cop_data->Ru.isApprox(cop_residual.get_reference().get_A() * df_du));

  crocoddyl::ResidualModelContactControlGravTpl<Scalar> gravity_residual(
      state, state->get_nv());
  const std::shared_ptr<ResidualData> gravity_data =
      gravity_residual.createData(&actuation_shared);
  gravity_residual.calc(gravity_data, x, u);
  gravity_residual.calcDiff(gravity_data, x, u);
  BOOST_CHECK(gravity_data->r.allFinite());
  BOOST_CHECK(gravity_data->Rx.allFinite());
  BOOST_CHECK(gravity_data->Ru.isApprox(actuation_data->dtau_du));

  crocoddyl::ResidualModelImpulseCoMTpl<Scalar> com_residual(state);
  const std::shared_ptr<ResidualData> com_data =
      com_residual.createData(&multibody_shared);
  com_residual.calc(com_data, x, VectorXs());
  com_residual.calcDiff(com_data, x, VectorXs());
  BOOST_CHECK_EQUAL(com_data->r.size(), 3);
  BOOST_CHECK(com_data->r.allFinite());
  BOOST_CHECK(com_data->Rx.allFinite());

  crocoddyl::DataCollectorImplicitConstraintTpl<Scalar> copied_force_shared(
      force_shared);
  BOOST_CHECK(copied_force_shared.constraints == constraints);
  crocoddyl::DataCollectorActMultibodyInImplicitConstraintTpl<Scalar>
      copied_actuation_shared(actuation_shared);
  BOOST_CHECK(copied_actuation_shared.constraints == constraints);
  BOOST_CHECK(copied_actuation_shared.actuation == actuation_data);

  Multiple empty_manager(state, state->get_nv());
  const std::shared_ptr<MultipleData> empty_data =
      empty_manager.createData(&pinocchio_data);
  crocoddyl::DataCollectorImplicitConstraintTpl<Scalar> empty_shared(
      empty_data);
  BOOST_CHECK_THROW(force_residual.createData(&empty_shared), std::exception);
}

void test_generic_residual_collectors_double() {
  test_generic_residual_collectors<double>();
}

void test_generic_residual_collectors_float() {
  test_generic_residual_collectors<float>();
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_implicit_constraint_residuals");
  ts->add(BOOST_TEST_CASE(&test_generic_residual_collectors_double));
  ts->add(BOOST_TEST_CASE(&test_generic_residual_collectors_float));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
