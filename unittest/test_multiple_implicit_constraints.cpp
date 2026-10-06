///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <sstream>
#include <type_traits>

#include "crocoddyl/multibody/data/implicit-constraints.hpp"
#include "crocoddyl/multibody/implicit-constraints/contact.hpp"
#include "crocoddyl/multibody/implicit-constraints/kinematic-loop.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "factory/state.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

namespace {
typedef crocoddyl::KinematicLoopModel LoopModel;
typedef crocoddyl::ContactModel ContactModel;
typedef LoopModel::MaskArray LoopMaskArray;
typedef ContactModel::MaskArray ContactMaskArray;

std::shared_ptr<crocoddyl::StateMultibody> create_state() {
  return std::static_pointer_cast<crocoddyl::StateMultibody>(
      StateModelFactory().create(StateModelTypes::StateMultibody_TalosArm));
}

pinocchio::SE3 create_placement1() {
  return pinocchio::SE3(Eigen::Matrix3d::Identity(),
                        Eigen::Vector3d(0.1, -0.2, 0.3));
}

pinocchio::SE3 create_placement2() {
  return pinocchio::SE3(
      Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()).toRotationMatrix(),
      Eigen::Vector3d(-0.05, 0.1, 0.2));
}

pinocchio::SE3 create_contact_reference() {
  return pinocchio::SE3(
      Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()).toRotationMatrix(),
      Eigen::Vector3d(0.1, -0.2, 0.3));
}

std::shared_ptr<LoopModel> create_constraint(
    const std::shared_ptr<crocoddyl::StateMultibody>& state,
    const LoopMaskArray& mask, const std::size_t nu,
    const Eigen::Vector2d& gains = Eigen::Vector2d::Zero()) {
  return std::make_shared<LoopModel>(
      state, 1, create_placement1(), 2, create_placement2(),
      pinocchio::ReferenceFrame::LOCAL, nu, gains, mask);
}

std::shared_ptr<ContactModel> create_contact_constraint(
    const std::shared_ptr<crocoddyl::StateMultibody>& state,
    const pinocchio::FrameIndex id, const ContactMaskArray& mask,
    const std::size_t nu,
    const pinocchio::ReferenceFrame type =
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
    const Eigen::Vector2d& gains = Eigen::Vector2d::Zero()) {
  return std::make_shared<ContactModel>(state, id, create_contact_reference(),
                                        type, nu, gains, mask);
}

template <typename Scalar>
struct ObserverPayload {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

  ObserverPayload()
      : xnext(VectorXs::Random(2)),
        Fx(MatrixXs::Random(2, 2)),
        Fu(MatrixXs::Random(2, 1)),
        Fp(MatrixXs::Random(2, 1)),
        dissipative_E(VectorXs::Random(1)),
        Ex(MatrixXs::Random(1, 2)),
        Eu(MatrixXs::Random(1, 1)),
        Ep(MatrixXs::Random(1, 1)) {}

  VectorXs xnext;
  MatrixXs Fx;
  MatrixXs Fu;
  MatrixXs Fp;
  VectorXs dissipative_E;
  MatrixXs Ex;
  MatrixXs Eu;
  MatrixXs Ep;
};

void test_constructor() {
  const std::shared_ptr<crocoddyl::StateMultibody> state = create_state();
  crocoddyl::ImplicitConstraintModelMultiple model(state);

  std::ostringstream tmp;
  tmp << model;

  BOOST_CHECK(model.get_constraints().empty());
  BOOST_CHECK_EQUAL(model.get_nc(), 0);
  BOOST_CHECK_EQUAL(model.get_nc_total(), 0);
  BOOST_CHECK_EQUAL(model.get_nu(), state->get_nv());
  BOOST_CHECK(!model.getComputeAllConstraints());

#ifdef NDEBUG
  crocoddyl::ImplicitConstraintModelMultipleTpl<float> casted_model =
      model.cast<float>();
  BOOST_CHECK(casted_model.get_constraints().empty());
#endif
}

template <typename Scalar>
void test_copy_ownership() {
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef crocoddyl::KinematicLoopModelTpl<Scalar> Constraint;
  typedef crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar> Multiple;
  typedef crocoddyl::ImplicitConstraintItemTpl<Scalar> Item;
  const std::shared_ptr<crocoddyl::StateMultibody> state64 = create_state();
  const std::shared_ptr<State> state =
      std::make_shared<State>(state64->template cast<Scalar>());
  const typename Constraint::MaskArray mask1 = {
      {true, true, false, false, false, false}};
  const typename Constraint::MaskArray mask2 = {
      {false, false, true, false, false, false}};
  const std::shared_ptr<Constraint> c1 = std::make_shared<Constraint>(
      state, 1, create_placement1().template cast<Scalar>(), 2,
      create_placement2().template cast<Scalar>(), pinocchio::LOCAL,
      state->get_nv(), Constraint::Vector2s::Zero(), mask1);
  const std::shared_ptr<Constraint> c2 = std::make_shared<Constraint>(
      state, 1, create_placement1().template cast<Scalar>(), 2,
      create_placement2().template cast<Scalar>(), pinocchio::LOCAL,
      state->get_nv(), Constraint::Vector2s::Zero(), mask2);
  Multiple model(state);
  model.addConstraint("c1", c1);
  model.addConstraint("c2", c2, false);
  model.setComputeAllConstraints(true);

  static_assert(!std::is_assignable<Item&, const Item&>::value,
                "constraint-item metadata must be manager-owned");
  static_assert(!std::is_assignable<Multiple&, const Multiple&>::value,
                "multiple-constraint assignment must not share mutable items");
  Multiple copied_model(model);
  BOOST_CHECK_EQUAL(copied_model.get_nc(), model.get_nc());
  BOOST_CHECK_EQUAL(copied_model.get_nc_total(), model.get_nc_total());
  BOOST_CHECK_EQUAL(copied_model.get_nu(), model.get_nu());
  BOOST_CHECK_EQUAL(copied_model.getComputeAllConstraints(),
                    model.getComputeAllConstraints());
  BOOST_CHECK(copied_model.get_active_set() == model.get_active_set());
  BOOST_CHECK(copied_model.get_inactive_set() == model.get_inactive_set());
  BOOST_CHECK(copied_model.get_constraints().at("c1") !=
              model.get_constraints().at("c1"));
  BOOST_CHECK(copied_model.get_constraints().at("c1")->get_constraint() == c1);

  copied_model.changeConstraintStatus("c1", false);
  copied_model.changeConstraintStatus("c2", true);
  BOOST_CHECK_EQUAL(copied_model.get_nc(), 1);
  BOOST_CHECK_EQUAL(copied_model.get_active_set().count("c2"), 1);
  BOOST_CHECK_EQUAL(copied_model.get_inactive_set().count("c1"), 1);
  BOOST_CHECK_EQUAL(model.get_nc(), 2);
  BOOST_CHECK_EQUAL(model.get_active_set().count("c1"), 1);
  BOOST_CHECK_EQUAL(model.get_inactive_set().count("c2"), 1);
  BOOST_CHECK(model.getConstraintStatus("c1"));
  BOOST_CHECK(!model.getConstraintStatus("c2"));
}

void test_copy_ownership_double() { test_copy_ownership<double>(); }

void test_copy_ownership_float() { test_copy_ownership<float>(); }

void test_add_remove_and_change_status() {
  const std::shared_ptr<crocoddyl::StateMultibody> state = create_state();
  const std::size_t nu = state->get_nv();
  crocoddyl::ImplicitConstraintModelMultiple model(state, nu);
  const LoopMaskArray mask1 = {{true, true, false, false, false, false}};
  const LoopMaskArray mask2 = {{false, false, true, false, false, false}};
  const std::shared_ptr<LoopModel> c1 =
      create_constraint(state, mask1, nu, Eigen::Vector2d(0.1, 0.2));
  const std::shared_ptr<LoopModel> c2 =
      create_constraint(state, mask2, nu, Eigen::Vector2d(0.2, 0.3));

  typedef crocoddyl::ImplicitConstraintItem Item;
  static_assert(!std::is_assignable<Item&, const Item&>::value,
                "constraint-item metadata must be manager-owned");
  BOOST_CHECK_THROW(Item("null", nullptr), crocoddyl::Exception);

  model.addConstraint("c1", c1);
  model.addConstraint("c2", c2, false);
  BOOST_CHECK_EQUAL(model.get_nc(), 2);
  BOOST_CHECK_EQUAL(model.get_nc_total(), 3);
  BOOST_CHECK(model.getConstraintStatus("c1"));
  BOOST_CHECK(!model.getConstraintStatus("c2"));
  BOOST_CHECK_EQUAL(model.get_active_set().size(), 1);
  BOOST_CHECK_EQUAL(model.get_inactive_set().size(), 1);

  BOOST_CHECK_THROW(model.addConstraint("null", nullptr), crocoddyl::Exception);
  BOOST_CHECK_THROW(
      model.addConstraint("wrong-nu", create_constraint(state, mask1, nu + 1)),
      crocoddyl::Exception);
  const std::shared_ptr<crocoddyl::StateMultibody> incompatible_state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(
          StateModelFactory().create(StateModelTypes::StateMultibody_HyQ));
  BOOST_CHECK_THROW(
      model.addConstraint("wrong-state",
                          create_constraint(incompatible_state, mask1, nu)),
      crocoddyl::Exception);

#ifdef NDEBUG
  const crocoddyl::ImplicitConstraintModelMultipleTpl<float> casted_model =
      model.cast<float>();
  BOOST_CHECK_EQUAL(casted_model.get_nc(), model.get_nc());
  BOOST_CHECK_EQUAL(casted_model.get_nc_total(), model.get_nc_total());
  BOOST_CHECK_EQUAL(casted_model.get_active_set().count("c1"), 1);
  BOOST_CHECK_EQUAL(casted_model.get_inactive_set().count("c2"), 1);
#endif

  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.addConstraint("c1", c1);
  capture_ios.endCapture();
  std::stringstream expected_buffer;
  expected_buffer << "Warning: we couldn't add the c1 constraint item, it "
                     "already existed."
                  << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());

  model.changeConstraintStatus("c2", true);
  BOOST_CHECK_EQUAL(model.get_nc(), 3);
  model.changeConstraintStatus("c1", false);
  BOOST_CHECK_EQUAL(model.get_nc(), 1);

  capture_ios.beginCapture();
  model.changeConstraintStatus("missing", true);
  capture_ios.endCapture();
  expected_buffer.clear();
  expected_buffer << "Warning: we couldn't change the status of the missing "
                     "constraint item, it doesn't exist."
                  << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());

  model.removeConstraint("c2");
  BOOST_CHECK_EQUAL(model.get_nc(), 0);
  BOOST_CHECK_EQUAL(model.get_nc_total(), 2);
  model.removeConstraint("c1");
  BOOST_CHECK_EQUAL(model.get_nc(), 0);
  BOOST_CHECK_EQUAL(model.get_nc_total(), 0);

  capture_ios.beginCapture();
  model.removeConstraint("missing");
  capture_ios.endCapture();
  expected_buffer.clear();
  expected_buffer << "Warning: we couldn't remove the missing constraint "
                     "item, it doesn't exist."
                  << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_calc_and_calc_diff() {
  const std::shared_ptr<crocoddyl::StateMultibody> state = create_state();
  const std::size_t nu = state->get_nv();
  crocoddyl::ImplicitConstraintModelMultiple model(state, nu);
  const LoopMaskArray mask1 = {{true, true, false, false, false, false}};
  const LoopMaskArray mask2 = {{false, false, true, false, false, false}};
  const std::shared_ptr<LoopModel> c1 = create_constraint(state, mask1, nu);
  const std::shared_ptr<LoopModel> c2 = create_constraint(state, mask2, nu);
  model.addConstraint("c1", c1);
  model.addConstraint("c2", c2, false);

  pinocchio::Data pinocchio_data(*state->get_pinocchio());
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataMultiple> data =
      model.createData(&pinocchio_data);
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract> c1_data =
      c1->createData(&pinocchio_data);
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract> c2_data =
      c2->createData(&pinocchio_data);
  const Eigen::VectorXd x = state->rand();

  model.calc(data, x);
  c1->calc(c1_data, x);
  BOOST_CHECK_EQUAL(data->constraints.size(), 2);
  const Eigen::Index nc1 = static_cast<Eigen::Index>(c1->get_nc());
  const Eigen::Index nc2 = static_cast<Eigen::Index>(c2->get_nc());
  BOOST_CHECK((data->a0.head(nc1) - c1_data->a0).isZero(1e-12));
  BOOST_CHECK(data->a0.tail(nc2).isZero(1e-12));
  BOOST_CHECK((data->Jc.topRows(nc1) - c1_data->Jc).isZero(1e-12));
  BOOST_CHECK(data->Jc.bottomRows(nc2).isZero(1e-12));

  model.calcDiff(data, x);
  c1->calcDiff(c1_data, x);
  BOOST_CHECK((data->da0_dx.topRows(nc1) - c1_data->da0_dx).isZero(1e-12));
  BOOST_CHECK(data->da0_dx.bottomRows(nc2).isZero(1e-12));
  BOOST_CHECK((data->dv0_dq.topRows(nc1) - c1_data->dv0_dq).isZero(1e-12));
  BOOST_CHECK(data->dv0_dq.bottomRows(nc2).isZero(1e-12));

  model.setComputeAllConstraints(true);
  model.calc(data, x);
  model.calcDiff(data, x);
  BOOST_CHECK((data->a0.head(nc1) - c1_data->a0).isZero(1e-12));
  BOOST_CHECK(data->a0.tail(nc2).isZero(1e-12));
  BOOST_CHECK((data->Jc.topRows(nc1) - c1_data->Jc).isZero(1e-12));
  BOOST_CHECK(data->Jc.bottomRows(nc2).isZero(1e-12));
  BOOST_CHECK((data->da0_dx.topRows(nc1) - c1_data->da0_dx).isZero(1e-12));
  BOOST_CHECK(data->da0_dx.bottomRows(nc2).isZero(1e-12));
  BOOST_CHECK((data->dv0_dq.topRows(nc1) - c1_data->dv0_dq).isZero(1e-12));
  BOOST_CHECK(data->dv0_dq.bottomRows(nc2).isZero(1e-12));
}

void test_calc_and_calc_diff_with_contact() {
  const std::shared_ptr<crocoddyl::StateMultibody> state = create_state();
  const std::size_t nu = state->get_nv();
  crocoddyl::ImplicitConstraintModelMultiple model(state, nu);
  const ContactMaskArray mask1 = {{true, true, true, false, false, false}};
  const ContactMaskArray mask2 = {{false, false, false, true, true, true}};
  const std::shared_ptr<ContactModel> c1 = create_contact_constraint(
      state, 2, mask1, nu, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      Eigen::Vector2d(0.1, 0.2));
  const std::shared_ptr<ContactModel> c2 = create_contact_constraint(
      state, 4, mask2, nu, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      Eigen::Vector2d(0.2, 0.3));
  model.addConstraint("c1", c1);
  model.addConstraint("c2", c2, false);

  pinocchio::Data pinocchio_data(*state->get_pinocchio());
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataMultiple> data =
      model.createData(&pinocchio_data);
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract> c1_data =
      c1->createData(&pinocchio_data);
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract> c2_data =
      c2->createData(&pinocchio_data);
  const Eigen::VectorXd x = state->rand();

  crocoddyl::unittest::updateAllPinocchio(state->get_pinocchio().get(),
                                          &pinocchio_data, x);
  model.calc(data, x);
  c1->calc(c1_data, x);
  BOOST_CHECK_EQUAL(data->constraints.size(), 2);
  const Eigen::Index nc1 = static_cast<Eigen::Index>(c1->get_nc());
  const Eigen::Index nc2 = static_cast<Eigen::Index>(c2->get_nc());
  BOOST_CHECK((data->a0.head(nc1) - c1_data->a0).isZero(1e-12));
  BOOST_CHECK(data->a0.tail(nc2).isZero(1e-12));
  BOOST_CHECK((data->Jc.topRows(nc1) - c1_data->Jc).isZero(1e-12));
  BOOST_CHECK(data->Jc.bottomRows(nc2).isZero(1e-12));

  model.calcDiff(data, x);
  c1->calcDiff(c1_data, x);
  BOOST_CHECK((data->da0_dx.topRows(nc1) - c1_data->da0_dx).isZero(1e-12));
  BOOST_CHECK(data->da0_dx.bottomRows(nc2).isZero(1e-12));
  BOOST_CHECK((data->dv0_dq.topRows(nc1) - c1_data->dv0_dq).isZero(1e-12));
  BOOST_CHECK(data->dv0_dq.bottomRows(nc2).isZero(1e-12));

  model.setComputeAllConstraints(true);
  model.calc(data, x);
  model.calcDiff(data, x);
  BOOST_CHECK((data->a0.head(nc1) - c1_data->a0).isZero(1e-12));
  BOOST_CHECK(data->a0.tail(nc2).isZero(1e-12));
  BOOST_CHECK((data->Jc.topRows(nc1) - c1_data->Jc).isZero(1e-12));
  BOOST_CHECK(data->Jc.bottomRows(nc2).isZero(1e-12));
  BOOST_CHECK((data->da0_dx.topRows(nc1) - c1_data->da0_dx).isZero(1e-12));
  BOOST_CHECK(data->da0_dx.bottomRows(nc2).isZero(1e-12));
  BOOST_CHECK((data->dv0_dq.topRows(nc1) - c1_data->dv0_dq).isZero(1e-12));
  BOOST_CHECK(data->dv0_dq.bottomRows(nc2).isZero(1e-12));
}

void test_update_helpers_and_rnea_diff() {
  const std::shared_ptr<crocoddyl::StateMultibody> state = create_state();
  const std::size_t nu = state->get_nv();
  crocoddyl::ImplicitConstraintModelMultiple model(state, nu);
  const LoopMaskArray mask1 = {{true, true, false, false, false, false}};
  const LoopMaskArray mask2 = {{false, false, true, false, false, false}};
  const std::shared_ptr<LoopModel> c1 = create_constraint(state, mask1, nu);
  const std::shared_ptr<LoopModel> c2 = create_constraint(state, mask2, nu);
  model.addConstraint("c1", c1);
  model.addConstraint("c2", c2, false);

  pinocchio::Data pinocchio_data(*state->get_pinocchio());
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataMultiple> data =
      model.createData(&pinocchio_data);
  const Eigen::VectorXd x = state->rand();

  const Eigen::VectorXd vnext = Eigen::VectorXd::Random(state->get_nv());
  const Eigen::VectorXd dv = Eigen::VectorXd::Random(state->get_nv());
  const Eigen::MatrixXd dvnext_dx =
      Eigen::MatrixXd::Random(state->get_nv(), state->get_ndx());
  const Eigen::MatrixXd ddv_dx =
      Eigen::MatrixXd::Random(state->get_nv(), state->get_ndx());
  model.updateVelocity(data, vnext);
  model.updateAcceleration(data, dv);
  model.updateVelocityDiff(data, dvnext_dx);
  model.updateAccelerationDiff(data, ddv_dx);
  BOOST_CHECK((data->vnext - vnext).isZero(1e-12));
  BOOST_CHECK((data->dv - dv).isZero(1e-12));
  BOOST_CHECK((data->dvnext_dx - dvnext_dx).isZero(1e-12));
  BOOST_CHECK((data->ddv_dx - ddv_dx).isZero(1e-12));

  const Eigen::VectorXd force =
      Eigen::VectorXd::LinSpaced(model.get_nc(), 1., 2.);
  model.updateForce(data, force);
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract>& c1_data =
      data->constraints.find("c1")->second;
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract>& c2_data =
      data->constraints.find("c2")->second;
  const std::shared_ptr<crocoddyl::KinematicLoopData> c1_loop_data =
      std::static_pointer_cast<crocoddyl::KinematicLoopData>(c1_data);
  const std::shared_ptr<crocoddyl::KinematicLoopData> c2_loop_data =
      std::static_pointer_cast<crocoddyl::KinematicLoopData>(c2_data);
  BOOST_CHECK((data->fext[c1->get_joint1_id()].toVector() -
               c1_loop_data->joint1_f.toVector())
                  .isZero(1e-12));
  BOOST_CHECK((data->fext[c1->get_joint2_id()].toVector() -
               c1_loop_data->joint2_f.toVector())
                  .isZero(1e-12));
  BOOST_CHECK(c2_loop_data->joint1_f.toVector().isZero());
  BOOST_CHECK(c2_loop_data->joint2_f.toVector().isZero());

  const Eigen::MatrixXd df_dx =
      Eigen::MatrixXd::Random(model.get_nc(), state->get_ndx());
  const Eigen::MatrixXd df_du = Eigen::MatrixXd::Random(model.get_nc(), nu);
  model.updateForceDiff(data, df_dx, df_du);
  BOOST_CHECK(
      (c1_data->df_dx + df_dx.topRows(c1_data->df_dx.rows())).isZero(1e-12));
  BOOST_CHECK(
      (c1_data->df_du + df_du.topRows(c1_data->df_du.rows())).isZero(1e-12));
  BOOST_CHECK(c2_data->df_dx.isZero());
  BOOST_CHECK(c2_data->df_du.isZero());

  model.calcDiff(data, x);
  pinocchio_data.dtau_dq.setZero();
  model.updateRneaDiff(data, pinocchio_data);
  BOOST_CHECK((pinocchio_data.dtau_dq - c1_data->dtau_dq).isZero(1e-12));

  model.setComputeAllConstraints(true);
  const Eigen::VectorXd force_all =
      Eigen::VectorXd::LinSpaced(model.get_nc_total(), 1., 3.);
  model.updateForce(data, force_all);
  model.calcDiff(data, x);
  pinocchio_data.dtau_dq.setZero();
  model.updateRneaDiff(data, pinocchio_data);
  BOOST_CHECK((pinocchio_data.dtau_dq - c1_data->dtau_dq).isZero(1e-12));
  BOOST_CHECK((data->fext[c1->get_joint1_id()].toVector() -
               c1_loop_data->joint1_f.toVector())
                  .isZero(1e-12));
  BOOST_CHECK((data->fext[c1->get_joint2_id()].toVector() -
               c1_loop_data->joint2_f.toVector())
                  .isZero(1e-12));
}

void test_update_helpers_and_rnea_diff_with_contact() {
  const std::shared_ptr<crocoddyl::StateMultibody> state = create_state();
  const std::size_t nu = state->get_nv();
  crocoddyl::ImplicitConstraintModelMultiple model(state, nu);
  const ContactMaskArray mask1 = {{true, true, true, false, false, false}};
  const ContactMaskArray mask2 = {{false, false, false, true, true, true}};
  const std::shared_ptr<ContactModel> c1 = create_contact_constraint(
      state, 2, mask1, nu, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
  const std::shared_ptr<ContactModel> c2 = create_contact_constraint(
      state, 4, mask2, nu, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
  model.addConstraint("c1", c1);
  model.addConstraint("c2", c2, false);

  pinocchio::Data pinocchio_data(*state->get_pinocchio());
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataMultiple> data =
      model.createData(&pinocchio_data);
  const Eigen::VectorXd x = state->rand();

  crocoddyl::unittest::updateAllPinocchio(state->get_pinocchio().get(),
                                          &pinocchio_data, x);
  model.calc(data, x);
  model.calcDiff(data, x);

  const Eigen::VectorXd force =
      Eigen::VectorXd::LinSpaced(model.get_nc(), 1., 3.);
  model.updateForce(data, force);
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract>& c1_data =
      data->constraints.find("c1")->second;
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract>& c2_data =
      data->constraints.find("c2")->second;
  const pinocchio::JointIndex joint1 =
      state->get_pinocchio()->frames[c1->get_id()].parentJoint;
  const pinocchio::JointIndex joint2 =
      state->get_pinocchio()->frames[c2->get_id()].parentJoint;
  BOOST_CHECK(
      (data->fext[joint1].toVector() - c1_data->fext.toVector()).isZero(1e-12));
  BOOST_CHECK(data->fext[joint2].toVector().isZero());

  const Eigen::MatrixXd df_dx =
      Eigen::MatrixXd::Random(model.get_nc(), state->get_ndx());
  const Eigen::MatrixXd df_du = Eigen::MatrixXd::Random(model.get_nc(), nu);
  model.updateForceDiff(data, df_dx, df_du);
  BOOST_CHECK(
      (c1_data->df_dx - df_dx.topRows(c1_data->df_dx.rows())).isZero(1e-12));
  BOOST_CHECK(
      (c1_data->df_du - df_du.topRows(c1_data->df_du.rows())).isZero(1e-12));
  BOOST_CHECK(c2_data->df_dx.isZero());
  BOOST_CHECK(c2_data->df_du.isZero());

  pinocchio_data.dtau_dq.setZero();
  model.updateRneaDiff(data, pinocchio_data);
  BOOST_CHECK((pinocchio_data.dtau_dq - c1_data->dtau_dq).isZero(1e-12));

  model.setComputeAllConstraints(true);
  const Eigen::VectorXd force_all =
      Eigen::VectorXd::LinSpaced(model.get_nc_total(), 1., 6.);
  model.updateForce(data, force_all);
  pinocchio_data.dtau_dq.setZero();
  model.updateRneaDiff(data, pinocchio_data);
  BOOST_CHECK((pinocchio_data.dtau_dq - c1_data->dtau_dq).isZero(1e-12));
  BOOST_CHECK(
      (data->fext[joint1].toVector() - c1_data->fext.toVector()).isZero(1e-12));
  BOOST_CHECK(data->fext[joint2].toVector().isZero());
}

void test_accumulated_external_forces() {
  const std::shared_ptr<crocoddyl::StateMultibody> state = create_state();
  const std::size_t nu = state->get_nv();
  crocoddyl::ImplicitConstraintModelMultiple model(state, nu);
  const ContactMaskArray linear_mask = {
      {true, true, true, false, false, false}};
  const ContactMaskArray angular_mask = {
      {false, false, false, true, true, true}};
  const std::shared_ptr<ContactModel> linear = create_contact_constraint(
      state, 2, linear_mask, nu, pinocchio::ReferenceFrame::LOCAL);
  const std::shared_ptr<ContactModel> angular = create_contact_constraint(
      state, 2, angular_mask, nu, pinocchio::ReferenceFrame::LOCAL);
  model.addConstraint("angular", angular);
  model.addConstraint("linear", linear);

  pinocchio::Data pinocchio_data(*state->get_pinocchio());
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataMultiple> data =
      model.createData(&pinocchio_data);
  const Eigen::VectorXd x = state->rand();
  crocoddyl::unittest::updateAllPinocchio(state->get_pinocchio().get(),
                                          &pinocchio_data, x);
  model.calc(data, x);
  model.updateForce(data, Eigen::VectorXd::LinSpaced(6, 1., 6.));

  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract>&
      angular_data = data->constraints.at("angular");
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract>&
      linear_data = data->constraints.at("linear");
  const pinocchio::JointIndex joint =
      state->get_pinocchio()->frames[linear->get_id()].parentJoint;
  BOOST_CHECK(data->fext[joint].toVector().isApprox(
      angular_data->fext.toVector() + linear_data->fext.toVector()));

  const crocoddyl::ImplicitConstraintDataMultiple copied_data(*data);
  BOOST_CHECK(copied_data.Jc.isApprox(data->Jc));
  BOOST_CHECK(copied_data.constraints.at("angular") == angular_data);
}

void test_dimension_checks() {
  const std::shared_ptr<crocoddyl::StateMultibody> state = create_state();
  const std::size_t nu = state->get_nv();
  crocoddyl::ImplicitConstraintModelMultiple model(state, nu);
  const LoopMaskArray mask1 = {{true, true, false, false, false, false}};
  model.addConstraint("c1", create_constraint(state, mask1, nu));

  pinocchio::Data pinocchio_data(*state->get_pinocchio());
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataMultiple> data =
      model.createData(&pinocchio_data);

  BOOST_CHECK_THROW(
      crocoddyl::ImplicitConstraintDataMultiple(
          static_cast<crocoddyl::ImplicitConstraintModelMultiple*>(nullptr),
          &pinocchio_data),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(crocoddyl::ImplicitConstraintDataMultiple(&model, nullptr),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(model.createData(nullptr), crocoddyl::Exception);

  BOOST_CHECK_THROW(
      model.updateVelocity(data, Eigen::VectorXd::Zero(state->get_nv() + 1)),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(model.updateAcceleration(
                        data, Eigen::VectorXd::Zero(state->get_nv() + 1)),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(
      model.updateForce(data, Eigen::VectorXd::Zero(model.get_nc() + 1)),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      model.updateVelocityDiff(
          data, Eigen::MatrixXd::Zero(state->get_nv() + 1, state->get_ndx())),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      model.updateAccelerationDiff(
          data, Eigen::MatrixXd::Zero(state->get_nv() + 1, state->get_ndx())),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      model.updateForceDiff(
          data, Eigen::MatrixXd::Zero(model.get_nc() + 1, state->get_ndx()),
          Eigen::MatrixXd::Zero(model.get_nc(), nu)),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      model.updateForceDiff(
          data, Eigen::MatrixXd::Zero(model.get_nc(), state->get_ndx()),
          Eigen::MatrixXd::Zero(model.get_nc(), nu + 1)),
      crocoddyl::Exception);
}

template <typename Scalar>
void test_hot_path_no_allocation() {
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef crocoddyl::ContactModelTpl<Scalar> Contact;
  typedef crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar> Multiple;
  typedef crocoddyl::ImplicitConstraintDataMultipleTpl<Scalar> MultipleData;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

  const std::shared_ptr<crocoddyl::StateMultibody> state64 = create_state();
  const std::shared_ptr<State> state =
      std::make_shared<State>(state64->template cast<Scalar>());
  typename Contact::SE3 reference =
      create_contact_reference().template cast<Scalar>();
  typename Contact::Vector2s gains;
  gains << Scalar(0.2), Scalar(0.1);
  const typename Contact::MaskArray mask = {
      {true, true, true, false, false, false}};
  const std::shared_ptr<Contact> contact = std::make_shared<Contact>(
      state, 2, reference, pinocchio::LOCAL_WORLD_ALIGNED, state->get_nv(),
      gains, mask);
  Multiple model(state, state->get_nv());
  model.addConstraint("contact", contact);
  pinocchio::DataTpl<Scalar> pinocchio_data(*state->get_pinocchio());
  const std::shared_ptr<MultipleData> data = model.createData(&pinocchio_data);
  const VectorXs x = state->rand();
  const VectorXs force = VectorXs::LinSpaced(3, Scalar(-1), Scalar(2));
  const MatrixXs df_dx = MatrixXs::Random(3, state->get_ndx());
  const MatrixXs df_du = MatrixXs::Random(3, state->get_nv());
  crocoddyl::unittest::updateAllPinocchio(state->get_pinocchio().get(),
                                          &pinocchio_data, x);
  model.calc(data, x);
  model.calcDiff(data, x);
  model.updateForce(data, force);
  model.updateForceDiff(data, df_dx, df_du);

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      model.calc(data, x);
      model.calcDiff(data, x);
      model.updateForce(data, force);
      model.updateForceDiff(data, df_dx, df_du);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
}

void test_hot_path_no_allocation_double() {
  test_hot_path_no_allocation<double>();
}

void test_hot_path_no_allocation_float() {
  test_hot_path_no_allocation<float>();
}

template <typename Scalar>
void test_observer_collector() {
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar> Multiple;
  typedef crocoddyl::ImplicitConstraintDataMultipleTpl<Scalar> MultipleData;
  typedef crocoddyl::DataCollectorMultibodyInImplicitConstraintTpl<Scalar>
      Collector;

  const std::shared_ptr<crocoddyl::StateMultibody> state64 = create_state();
  const std::shared_ptr<State> state =
      std::make_shared<State>(state64->template cast<Scalar>());
  Multiple model(state);
  pinocchio::DataTpl<Scalar> pinocchio_data(*state->get_pinocchio());
  const std::shared_ptr<MultipleData> data = model.createData(&pinocchio_data);
  Collector collector(&pinocchio_data, data);
  ObserverPayload<Scalar> observer;

  BOOST_CHECK(!collector.hasObserverData());
  BOOST_CHECK_THROW(
      collector.template shareObserverData<ObserverPayload<Scalar> >(nullptr),
      crocoddyl::Exception);
  collector.shareObserverData(&observer);
  BOOST_CHECK(collector.hasObserverData());
  BOOST_CHECK(collector.xnext == &observer.xnext);
  BOOST_CHECK(collector.int_Fx == &observer.Fx);
  BOOST_CHECK(collector.int_Fu == &observer.Fu);
  BOOST_CHECK(collector.int_Fp == &observer.Fp);
  BOOST_CHECK(collector.dissipative_E == &observer.dissipative_E);
  BOOST_CHECK(collector.Ex == &observer.Ex);
  BOOST_CHECK(collector.Eu == &observer.Eu);
  BOOST_CHECK(collector.Ep == &observer.Ep);
}

void test_observer_collector_double() { test_observer_collector<double>(); }

void test_observer_collector_float() { test_observer_collector<float>(); }

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_multiple_implicit_constraints");
  ts->add(BOOST_TEST_CASE(&test_constructor));
  ts->add(BOOST_TEST_CASE(&test_copy_ownership_double));
  ts->add(BOOST_TEST_CASE(&test_copy_ownership_float));
  ts->add(BOOST_TEST_CASE(&test_add_remove_and_change_status));
  ts->add(BOOST_TEST_CASE(&test_calc_and_calc_diff));
  ts->add(BOOST_TEST_CASE(&test_calc_and_calc_diff_with_contact));
  ts->add(BOOST_TEST_CASE(&test_update_helpers_and_rnea_diff));
  ts->add(BOOST_TEST_CASE(&test_update_helpers_and_rnea_diff_with_contact));
  ts->add(BOOST_TEST_CASE(&test_accumulated_external_forces));
  ts->add(BOOST_TEST_CASE(&test_dimension_checks));
  ts->add(BOOST_TEST_CASE(&test_hot_path_no_allocation_double));
  ts->add(BOOST_TEST_CASE(&test_hot_path_no_allocation_float));
  ts->add(BOOST_TEST_CASE(&test_observer_collector_double));
  ts->add(BOOST_TEST_CASE(&test_observer_collector_float));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
