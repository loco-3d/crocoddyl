///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <cmath>

#include "crocoddyl/multibody/implicit-constraints/contact.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "factory/state.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

namespace {

typedef crocoddyl::ContactModel Model;
typedef crocoddyl::ContactData Data;
typedef Model::MaskArray MaskArray;

std::shared_ptr<crocoddyl::StateMultibody> create_state() {
  return std::static_pointer_cast<crocoddyl::StateMultibody>(
      StateModelFactory().create(StateModelTypes::StateMultibody_TalosArm));
}

pinocchio::SE3 create_reference() {
  return pinocchio::SE3(
      Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()).toRotationMatrix(),
      Eigen::Vector3d(0.1, -0.2, 0.3));
}

std::shared_ptr<Model> create_model(
    const MaskArray& mask, const pinocchio::ReferenceFrame type,
    const Eigen::Vector2d& gains = Eigen::Vector2d::Zero()) {
  const std::shared_ptr<crocoddyl::StateMultibody> state = create_state();
  return std::make_shared<Model>(state, 2, create_reference(), type,
                                 state->get_nv(), gains, mask);
}

Eigen::VectorXd compute_velocity_drift(const std::shared_ptr<Model>& model,
                                       pinocchio::Data& pinocchio_data,
                                       const Eigen::VectorXd& x) {
  crocoddyl::unittest::updateAllPinocchio(
      model->get_state()->get_pinocchio().get(), &pinocchio_data, x);

  pinocchio::Motion velocity = pinocchio::Motion::Zero();
  switch (model->get_type()) {
    case pinocchio::ReferenceFrame::LOCAL:
      velocity = pinocchio::getFrameVelocity(
          *model->get_state()->get_pinocchio().get(), pinocchio_data,
          model->get_id(), pinocchio::LOCAL);
      break;
    case pinocchio::ReferenceFrame::WORLD:
    case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
      velocity = pinocchio::getFrameVelocity(
          *model->get_state()->get_pinocchio().get(), pinocchio_data,
          model->get_id(), model->get_type());
      break;
  }
  return model->get_selection_matrix().transpose() * velocity.toVector();
}

void test_construct_data_and_accessors() {
  const MaskArray mask = {{true, false, true, false, true, false}};
  const std::shared_ptr<Model> model = create_model(
      mask, pinocchio::ReferenceFrame::LOCAL, Eigen::Vector2d(1., 2.));
  const std::shared_ptr<pinocchio::Model>& pinocchio_model =
      model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model);
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract> data =
      model->createData(&pinocchio_data);
  const std::shared_ptr<Data> contact_data =
      std::static_pointer_cast<Data>(data);

  std::ostringstream tmp;
  tmp << *model;

  BOOST_CHECK_EQUAL(model->get_nc(), 3);
  BOOST_CHECK_EQUAL(model->get_id(), 2);
  BOOST_CHECK(model->get_type() == pinocchio::ReferenceFrame::LOCAL);
  BOOST_CHECK_EQUAL(data->Jc.rows(), 3);
  BOOST_CHECK_EQUAL(data->Jc.cols(), model->get_state()->get_nv());
  BOOST_CHECK_EQUAL(data->df_du.cols(), model->get_nu());
  BOOST_CHECK(model->get_gains().isApprox(Eigen::Vector2d(1., 2.)));
  BOOST_CHECK_EQUAL(model->get_mask()[0], true);
  BOOST_CHECK_EQUAL(model->get_mask()[1], false);
  BOOST_CHECK(model->get_reference().isApprox(create_reference()));
  BOOST_CHECK(contact_data->mask == mask);
}

void test_calc_and_mask_projection() {
  const MaskArray full_mask = {{true, true, true, true, true, true}};
  const MaskArray partial_mask = {{true, false, true, true, false, true}};
  const Eigen::Vector2d gains(0.2, 0.3);
  const std::shared_ptr<Model> full_model = create_model(
      full_mask, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, gains);
  const std::shared_ptr<Model> partial_model = create_model(
      partial_mask, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, gains);

  pinocchio::Data pinocchio_full(*full_model->get_state()->get_pinocchio());
  pinocchio::Data pinocchio_partial(
      *partial_model->get_state()->get_pinocchio());
  const std::shared_ptr<Data> full_data =
      std::static_pointer_cast<Data>(full_model->createData(&pinocchio_full));
  const std::shared_ptr<Data> partial_data = std::static_pointer_cast<Data>(
      partial_model->createData(&pinocchio_partial));
  const Eigen::VectorXd x = full_model->get_state()->rand();

  crocoddyl::unittest::updateAllPinocchio(
      full_model->get_state()->get_pinocchio().get(), &pinocchio_full, x);
  crocoddyl::unittest::updateAllPinocchio(
      partial_model->get_state()->get_pinocchio().get(), &pinocchio_partial, x);

  full_model->calc(full_data, x);
  partial_model->calc(partial_data, x);
  full_model->calcDiff(full_data, x);
  partial_model->calcDiff(partial_data, x);

  std::size_t row = 0;
  for (std::size_t i = 0; i < 6; ++i) {
    if (partial_mask[i]) {
      BOOST_CHECK_SMALL(partial_data->a0[row] - full_data->a0[i], 1e-12);
      BOOST_CHECK((partial_data->Jc.row(static_cast<Eigen::Index>(row)) -
                   full_data->Jc.row(static_cast<Eigen::Index>(i)))
                      .isZero(1e-12));
      BOOST_CHECK((partial_data->da0_dx.row(static_cast<Eigen::Index>(row)) -
                   full_data->da0_dx.row(static_cast<Eigen::Index>(i)))
                      .isZero(1e-12));
      BOOST_CHECK((partial_data->dv0_dq.row(static_cast<Eigen::Index>(row)) -
                   full_data->dv0_dq.row(static_cast<Eigen::Index>(i)))
                      .isZero(1e-12));
      ++row;
    }
  }
}

void test_calc_diff_against_finite_differences() {
  const MaskArray full_mask = {{true, true, true, true, true, true}};
  const std::shared_ptr<Model> model =
      create_model(full_mask, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                   Eigen::Vector2d(0.2, 0.1));
  pinocchio::Data pinocchio_data(*model->get_state()->get_pinocchio());
  const std::shared_ptr<Data> data =
      std::static_pointer_cast<Data>(model->createData(&pinocchio_data));
  const Eigen::VectorXd x = model->get_state()->rand();
  const std::size_t nv = model->get_state()->get_nv();
  const std::size_t ndx = model->get_state()->get_ndx();
  const double eps = 1e-8;

  crocoddyl::unittest::updateAllPinocchio(
      model->get_state()->get_pinocchio().get(), &pinocchio_data, x);
  model->calc(data, x);
  model->calcDiff(data, x);

  Eigen::MatrixXd da0_dx_fd(model->get_nc(), ndx);
  for (std::size_t i = 0; i < ndx; ++i) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(ndx);
    dx[static_cast<Eigen::Index>(i)] = eps;
    Eigen::VectorXd x_plus(model->get_state()->get_nx());
    Eigen::VectorXd x_minus(model->get_state()->get_nx());
    model->get_state()->integrate(x, dx, x_plus);
    model->get_state()->integrate(x, -dx, x_minus);

    pinocchio::Data pinocchio_plus(*model->get_state()->get_pinocchio());
    pinocchio::Data pinocchio_minus(*model->get_state()->get_pinocchio());
    const std::shared_ptr<Data> data_plus =
        std::static_pointer_cast<Data>(model->createData(&pinocchio_plus));
    const std::shared_ptr<Data> data_minus =
        std::static_pointer_cast<Data>(model->createData(&pinocchio_minus));

    crocoddyl::unittest::updateAllPinocchio(
        model->get_state()->get_pinocchio().get(), &pinocchio_plus, x_plus);
    crocoddyl::unittest::updateAllPinocchio(
        model->get_state()->get_pinocchio().get(), &pinocchio_minus, x_minus);
    model->calc(data_plus, x_plus);
    model->calc(data_minus, x_minus);
    da0_dx_fd.col(static_cast<Eigen::Index>(i)).noalias() =
        (data_plus->a0 - data_minus->a0) / (2. * eps);
  }
  BOOST_CHECK((data->da0_dx - da0_dx_fd).isZero(2e-4));

  Eigen::MatrixXd dv0_dq_fd(model->get_nc(), nv);
  for (std::size_t i = 0; i < nv; ++i) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(ndx);
    dx[static_cast<Eigen::Index>(i)] = eps;
    Eigen::VectorXd x_plus(model->get_state()->get_nx());
    Eigen::VectorXd x_minus(model->get_state()->get_nx());
    model->get_state()->integrate(x, dx, x_plus);
    model->get_state()->integrate(x, -dx, x_minus);

    pinocchio::Data pinocchio_plus(*model->get_state()->get_pinocchio());
    pinocchio::Data pinocchio_minus(*model->get_state()->get_pinocchio());
    dv0_dq_fd.col(static_cast<Eigen::Index>(i)).noalias() =
        (compute_velocity_drift(model, pinocchio_plus, x_plus) -
         compute_velocity_drift(model, pinocchio_minus, x_minus)) /
        (2. * eps);
  }
  BOOST_CHECK((data->dv0_dq - dv0_dq_fd).isZero(2e-4));
}

void test_update_force_and_multiple_scatter() {
  const MaskArray mask = {{true, false, true, false, true, false}};
  const std::shared_ptr<Model> model =
      create_model(mask, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
  pinocchio::Data pinocchio_data(*model->get_state()->get_pinocchio());
  const std::shared_ptr<Data> data =
      std::static_pointer_cast<Data>(model->createData(&pinocchio_data));
  const Eigen::VectorXd x = model->get_state()->rand();

  crocoddyl::unittest::updateAllPinocchio(
      model->get_state()->get_pinocchio().get(), &pinocchio_data, x);
  model->calc(data, x);

  const Eigen::VectorXd force =
      Eigen::VectorXd::LinSpaced(model->get_nc(), 1., 3.);
  model->updateForce(data, force);
  BOOST_CHECK_CLOSE(data->f.linear()[0], force[0], 1e-9);
  BOOST_CHECK_CLOSE(data->f.angular()[1], force[2], 1e-9);
  BOOST_CHECK(!data->fext.toVector().isZero());
  BOOST_CHECK_EQUAL(data->dtau_dq.rows(), model->get_state()->get_nv());
  BOOST_CHECK_EQUAL(data->dtau_dq.cols(), model->get_state()->get_nv());
  BOOST_CHECK(data->dtau_dq.allFinite());

  const Eigen::MatrixXd df_dx =
      Eigen::MatrixXd::Random(model->get_nc(), model->get_state()->get_ndx());
  const Eigen::MatrixXd df_du =
      Eigen::MatrixXd::Random(model->get_nc(), model->get_nu());
  model->updateForceDiff(data, df_dx, df_du);
  BOOST_CHECK((data->df_dx - df_dx).isZero(1e-12));
  BOOST_CHECK((data->df_du - df_du).isZero(1e-12));

  crocoddyl::ImplicitConstraintModelMultiple multiple(model->get_state(),
                                                      model->get_nu());
  multiple.addConstraint("contact", model);
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataMultiple>
      multiple_data = multiple.createData(&pinocchio_data);
  multiple.calc(multiple_data, x);
  multiple.updateForce(multiple_data, force);
  const pinocchio::JointIndex joint =
      model->get_state()->get_pinocchio()->frames[model->get_id()].parentJoint;
  BOOST_CHECK(
      multiple_data->fext[joint].toVector().isApprox(data->fext.toVector()));
}

template <typename Scalar>
void check_contact_dimensions_and_masks() {
  typedef crocoddyl::ContactModelTpl<Scalar> GenericModel;
  typedef crocoddyl::ContactDataTpl<Scalar> GenericData;
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;

  const std::shared_ptr<crocoddyl::StateMultibody> state64 = create_state();
  const std::shared_ptr<State> state =
      std::make_shared<State>(state64->template cast<Scalar>());
  const typename GenericModel::MaskArray masks[] = {
      {{false, false, true, false, false, false}},
      {{true, false, true, false, false, false}},
      {{true, true, true, true, true, true}},
      {{true, false, false, false, true, false}}};
  const typename GenericModel::SE3 reference =
      create_reference().template cast<Scalar>();
  for (const typename GenericModel::MaskArray& mask : masks) {
    const std::shared_ptr<GenericModel> model = std::make_shared<GenericModel>(
        state, 2, reference, pinocchio::LOCAL, state->get_nv(),
        GenericModel::Vector2s::Zero(), mask);
    pinocchio::DataTpl<Scalar> pinocchio_data(*state->get_pinocchio());
    const std::shared_ptr<GenericData> data =
        std::static_pointer_cast<GenericData>(
            model->createData(&pinocchio_data));
    const VectorXs x = state->rand();
    crocoddyl::unittest::updateAllPinocchio(state->get_pinocchio().get(),
                                            &pinocchio_data, x);
    model->calc(data, x);
    model->calcDiff(data, x);
    const VectorXs force =
        VectorXs::LinSpaced(model->get_nc(), Scalar(0.5), Scalar(2.5));
    model->updateForce(data, force);

    BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Jc.rows()),
                      model->get_nc());
    BOOST_CHECK(data->Jc.allFinite());
    BOOST_CHECK(data->da0_dx.allFinite());
    std::size_t row = 0;
    for (std::size_t i = 0; i < mask.size(); ++i) {
      if (mask[i]) {
        BOOST_CHECK_SMALL(data->force_6d[static_cast<Eigen::Index>(i)] -
                              force[static_cast<Eigen::Index>(row)],
                          Eigen::NumTraits<Scalar>::dummy_precision());
        ++row;
      }
    }
  }
}

void test_contact_dimensions_and_masks_double() {
  check_contact_dimensions_and_masks<double>();
}

void test_contact_dimensions_and_masks_float() {
  check_contact_dimensions_and_masks<float>();
}

void test_copy_cast_and_validation() {
  const MaskArray mask = {{true, false, true, false, true, false}};
  const std::shared_ptr<Model> model = create_model(
      mask, pinocchio::LOCAL_WORLD_ALIGNED, Eigen::Vector2d(0.2, 0.3));
  pinocchio::Data pinocchio_data(*model->get_state()->get_pinocchio());
  const std::shared_ptr<Data> data =
      std::static_pointer_cast<Data>(model->createData(&pinocchio_data));
  const Eigen::VectorXd x = model->get_state()->rand();
  crocoddyl::unittest::updateAllPinocchio(
      model->get_state()->get_pinocchio().get(), &pinocchio_data, x);
  model->calc(data, x);
  model->calcDiff(data, x);

  const Model copied_model(*model);
  BOOST_CHECK_EQUAL(copied_model.get_nc(), model->get_nc());
  BOOST_CHECK(copied_model.get_reference().isApprox(model->get_reference()));
  const Data copied_data(*data);
  BOOST_CHECK(copied_data.Jc.isApprox(data->Jc));
  BOOST_CHECK(copied_data.pinocchio == data->pinocchio);
  BOOST_CHECK(copied_data.mask == model->get_mask());
#ifdef NDEBUG
  const crocoddyl::ContactModelTpl<float> casted_model = model->cast<float>();
  BOOST_CHECK_EQUAL(casted_model.get_nc(), model->get_nc());
  BOOST_CHECK_EQUAL(casted_model.get_mask()[2], model->get_mask()[2]);
  BOOST_CHECK(
      casted_model.get_gains().isApprox(model->get_gains().cast<float>()));
#endif

  const MaskArray empty_mask = {{false, false, false, false, false, false}};
  BOOST_CHECK_THROW(
      std::make_shared<Model>(model->get_state(), 2, create_reference(),
                              pinocchio::LOCAL, model->get_nu(),
                              Eigen::Vector2d::Zero(), empty_mask),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(std::make_shared<Model>(
                        model->get_state(),
                        model->get_state()->get_pinocchio()->frames.size(),
                        create_reference(), pinocchio::LOCAL, model->get_nu(),
                        Eigen::Vector2d::Zero(), mask),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(
      std::make_shared<Model>(model->get_state(), 2, create_reference(),
                              static_cast<pinocchio::ReferenceFrame>(99),
                              model->get_nu(), Eigen::Vector2d::Zero(), mask),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(model->createData(nullptr), crocoddyl::Exception);
  BOOST_CHECK_THROW(
      model->updateForce(data, Eigen::VectorXd::Zero(model->get_nc() + 1)),
      crocoddyl::Exception);
}

template <typename Scalar>
void check_contact_3d_analytical() {
  typedef crocoddyl::ContactModelTpl<Scalar> GenericModel;
  typedef crocoddyl::ContactDataTpl<Scalar> GenericData;
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  typedef Eigen::Matrix<Scalar, 6, Eigen::Dynamic> Matrix6xs;

  const std::shared_ptr<crocoddyl::StateMultibody> state64 = create_state();
  const std::shared_ptr<State> state =
      std::make_shared<State>(state64->template cast<Scalar>());
  const typename GenericModel::MaskArray mask = {
      {true, true, true, false, false, false}};
  const pinocchio::ReferenceFrame frames[] = {
      pinocchio::LOCAL, pinocchio::WORLD, pinocchio::LOCAL_WORLD_ALIGNED};
  const typename GenericModel::Vector2s gains[] = {
      GenericModel::Vector2s::Zero(),
      (typename GenericModel::Vector2s() << Scalar(0.4), Scalar(0.25))
          .finished()};
  const typename GenericModel::SE3 reference =
      create_reference().template cast<Scalar>();
  const Scalar tolerance =
      std::is_same<Scalar, double>::value ? Scalar(2e-10) : Scalar(3e-4);

  for (const pinocchio::ReferenceFrame frame : frames) {
    for (const typename GenericModel::Vector2s& gain : gains) {
      const std::shared_ptr<GenericModel> model =
          std::make_shared<GenericModel>(state, 2, reference, frame,
                                         state->get_nv(), gain, mask);
      pinocchio::DataTpl<Scalar> pin_data(*state->get_pinocchio());
      const std::shared_ptr<GenericData> data =
          std::static_pointer_cast<GenericData>(model->createData(&pin_data));
      const VectorXs x = state->rand();
      const VectorXs q = x.head(state->get_nq());
      const VectorXs v = x.tail(state->get_nv());
      const VectorXs a = VectorXs::Zero(state->get_nv());

      pinocchio::forwardKinematics(*state->get_pinocchio(), pin_data, q, v, a);
      pinocchio::computeForwardKinematicsDerivatives(*state->get_pinocchio(),
                                                     pin_data, q, v, a);
      pinocchio::computeJointJacobians(*state->get_pinocchio(), pin_data, q);
      pinocchio::updateFramePlacements(*state->get_pinocchio(), pin_data);
      model->calc(data, x);

      Matrix6xs Jlocal(6, state->get_nv());
      pinocchio::getFrameJacobian(*state->get_pinocchio(), pin_data,
                                  model->get_id(), pinocchio::LOCAL, Jlocal);
      pinocchio::MotionTpl<Scalar> a_local =
          pinocchio::MotionTpl<Scalar>::Zero();
      const pinocchio::MotionTpl<Scalar> v_local = pinocchio::getFrameVelocity(
          *state->get_pinocchio(), pin_data, model->get_id(), pinocchio::LOCAL);
      a_local.linear() = pinocchio::getFrameClassicalAcceleration(
                             *state->get_pinocchio(), pin_data, model->get_id(),
                             pinocchio::LOCAL)
                             .linear();
      if (gain[0] != Scalar(0)) {
        a_local.linear().noalias() +=
            gain[0] * pin_data.oMf[model->get_id()].rotation().transpose() *
            (pin_data.oMf[model->get_id()].translation() -
             reference.translation());
      }
      if (gain[1] != Scalar(0)) {
        a_local.linear().noalias() += gain[1] * v_local.linear();
      }
      Matrix6xs Jexpected = Jlocal;
      pinocchio::MotionTpl<Scalar> a_expected = a_local;
      if (frame != pinocchio::LOCAL) {
        typename GenericModel::SE3 rotation = GenericModel::SE3::Identity();
        rotation.rotation(pin_data.oMf[model->get_id()].rotation());
        Jexpected.noalias() = rotation.toActionMatrix() * Jlocal;
        a_expected = rotation.act(a_local);
      }
      BOOST_CHECK(data->Jc.isApprox(Jexpected.topRows(3), tolerance));
      BOOST_CHECK(data->a0.isApprox(a_expected.linear(), tolerance));

      const VectorXs force = VectorXs::LinSpaced(3, Scalar(-1.2), Scalar(2.1));
      model->updateForce(data, force);
      pinocchio::ForceTpl<Scalar> expected_force =
          pinocchio::ForceTpl<Scalar>::Zero();
      expected_force.linear() = force;
      const pinocchio::JointIndex joint =
          state->get_pinocchio()->frames[model->get_id()].parentJoint;
      const typename GenericModel::SE3 jMf =
          pin_data.oMi[joint].inverse() * pin_data.oMf[model->get_id()];
      pinocchio::ForceTpl<Scalar> expected_local = expected_force;
      if (frame != pinocchio::LOCAL) {
        typename GenericModel::SE3 rotation = GenericModel::SE3::Identity();
        rotation.rotation(pin_data.oMf[model->get_id()].rotation());
        expected_local = rotation.actInv(expected_force);
      }
      BOOST_CHECK(
          data->f.toVector().isApprox(expected_force.toVector(), tolerance));
      BOOST_CHECK(data->fext.toVector().isApprox(
          jMf.act(expected_local).toVector(), Scalar(10) * tolerance));
    }
  }
}

template <typename Scalar>
void check_contact_hot_path_no_allocation() {
  typedef crocoddyl::ContactModelTpl<Scalar> Model;
  typedef crocoddyl::ContactDataTpl<Scalar> Data;
  typedef crocoddyl::StateMultibodyTpl<Scalar> State;
  typedef typename Model::VectorXs VectorXs;
  typedef typename Model::MatrixXs MatrixXs;

  const std::shared_ptr<crocoddyl::StateMultibody> state64 = create_state();
  const std::shared_ptr<State> state =
      std::make_shared<State>(state64->template cast<Scalar>());
  const typename Model::MaskArray mask = {{true, true, true, true, true, true}};
  typename Model::Vector2s gains;
  gains << Scalar(0.2), Scalar(0.1);
  const std::shared_ptr<Model> model = std::make_shared<Model>(
      state, 2, create_reference().template cast<Scalar>(),
      pinocchio::LOCAL_WORLD_ALIGNED, state->get_nv(), gains, mask);
  pinocchio::DataTpl<Scalar> pinocchio_data(*state->get_pinocchio());
  const std::shared_ptr<Data> data =
      std::static_pointer_cast<Data>(model->createData(&pinocchio_data));
  const VectorXs x = state->rand();
  const VectorXs force =
      VectorXs::LinSpaced(model->get_nc(), Scalar(-1), Scalar(2));
  const MatrixXs df_dx = MatrixXs::Random(model->get_nc(), state->get_ndx());
  const MatrixXs df_du = MatrixXs::Random(model->get_nc(), model->get_nu());
  crocoddyl::unittest::updateAllPinocchio(state->get_pinocchio().get(),
                                          &pinocchio_data, x);
  model->calc(data, x);
  model->calcDiff(data, x);
  model->updateForce(data, force);
  model->updateForceDiff(data, df_dx, df_du);

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  try {
    Eigen::internal::set_is_malloc_allowed(false);
    for (std::size_t i = 0; i < 100; ++i) {
      model->calc(data, x);
      model->calcDiff(data, x);
      model->updateForce(data, force);
      model->updateForceDiff(data, df_dx, df_du);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
}

void test_contact_hot_path_no_allocation_double() {
  check_contact_hot_path_no_allocation<double>();
}

void test_contact_hot_path_no_allocation_float() {
  check_contact_hot_path_no_allocation<float>();
}

void test_contact_3d_analytical_double() {
  check_contact_3d_analytical<double>();
}

void test_contact_3d_analytical_float() {
  check_contact_3d_analytical<float>();
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_generic_contact");

  ts->add(BOOST_TEST_CASE(&test_construct_data_and_accessors));
  ts->add(BOOST_TEST_CASE(&test_calc_and_mask_projection));
  ts->add(BOOST_TEST_CASE(&test_calc_diff_against_finite_differences));
  ts->add(BOOST_TEST_CASE(&test_update_force_and_multiple_scatter));
  ts->add(BOOST_TEST_CASE(&test_contact_dimensions_and_masks_double));
  ts->add(BOOST_TEST_CASE(&test_contact_dimensions_and_masks_float));
  ts->add(BOOST_TEST_CASE(&test_copy_cast_and_validation));
  ts->add(BOOST_TEST_CASE(&test_contact_3d_analytical_double));
  ts->add(BOOST_TEST_CASE(&test_contact_3d_analytical_float));
  ts->add(BOOST_TEST_CASE(&test_contact_hot_path_no_allocation_double));
  ts->add(BOOST_TEST_CASE(&test_contact_hot_path_no_allocation_float));

  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
