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
#include <limits>
#include <type_traits>

#include "crocoddyl/multibody/implicit-constraints/kinematic-loop.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "factory/state.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

namespace {

std::shared_ptr<crocoddyl::StateMultibody> create_state() {
  return std::static_pointer_cast<crocoddyl::StateMultibody>(
      StateModelFactory().create(StateModelTypes::StateMultibody_TalosArm));
}

template <typename Scalar>
std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > create_scalar_state() {
  const std::shared_ptr<crocoddyl::StateMultibody> state = create_state();
  return std::make_shared<crocoddyl::StateMultibodyTpl<Scalar> >(
      state->template cast<Scalar>());
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

template <typename Scalar>
std::shared_ptr<crocoddyl::KinematicLoopModelTpl<Scalar> > create_model(
    const typename crocoddyl::KinematicLoopModelTpl<Scalar>::MaskArray& mask,
    const typename crocoddyl::KinematicLoopModelTpl<Scalar>::Vector2s& gains =
        crocoddyl::KinematicLoopModelTpl<Scalar>::Vector2s::Zero()) {
  typedef crocoddyl::KinematicLoopModelTpl<Scalar> Model;
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > state =
      create_scalar_state<Scalar>();
  return std::make_shared<Model>(
      state, 1, create_placement1().template cast<Scalar>(), 2,
      create_placement2().template cast<Scalar>(), pinocchio::LOCAL,
      state->get_nv(), gains, mask);
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> compute_velocity_drift(
    const std::shared_ptr<crocoddyl::KinematicLoopModelTpl<Scalar> >& model,
    pinocchio::DataTpl<Scalar>& pinocchio_data,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x) {
  typedef pinocchio::MotionTpl<Scalar> Motion;
  typedef pinocchio::SE3Tpl<Scalar> SE3;

  crocoddyl::unittest::updateAllPinocchio(
      model->get_state()->get_pinocchio().get(), &pinocchio_data, x);
  Motion f1vf1 = Motion::Zero();
  Motion f2vf2 = Motion::Zero();
  Motion f1vf2 = Motion::Zero();
  const SE3 oMf1 = pinocchio_data.oMi[model->get_joint1_id()].act(
      model->get_joint1_placement());
  const SE3 oMf2 = pinocchio_data.oMi[model->get_joint2_id()].act(
      model->get_joint2_placement());
  const SE3 f1Mf2 = oMf1.actInv(oMf2);

  if (model->get_joint1_id() > 0) {
    f1vf1 = model->get_joint1_placement().actInv(
        pinocchio_data.v[model->get_joint1_id()]);
  }
  if (model->get_joint2_id() > 0) {
    f2vf2 = model->get_joint2_placement().actInv(
        pinocchio_data.v[model->get_joint2_id()]);
    f1vf2 = f1Mf2.act(f2vf2);
  }

  return model->get_selection_matrix().transpose() * (f1vf1 - f1vf2).toVector();
}

template <typename Scalar>
void test_construct_data_and_accessors() {
  typedef crocoddyl::KinematicLoopModelTpl<Scalar> Model;
  typedef crocoddyl::KinematicLoopDataTpl<Scalar> Data;
  const typename Model::MaskArray mask = {
      {true, false, true, false, true, false}};
  typename Model::Vector2s gains;
  gains << Scalar(1), Scalar(2);
  const std::shared_ptr<Model> model = create_model<Scalar>(mask, gains);
  pinocchio::DataTpl<Scalar> pinocchio_data(
      *model->get_state()->get_pinocchio());
  const std::shared_ptr<Data> data =
      std::static_pointer_cast<Data>(model->createData(&pinocchio_data));

  std::ostringstream tmp;
  tmp << *model;
  BOOST_CHECK_EQUAL(model->get_nc(), 3);
  BOOST_CHECK_EQUAL(model->get_joint1_id(), 1);
  BOOST_CHECK_EQUAL(model->get_joint2_id(), 2);
  BOOST_CHECK(model->get_type() == pinocchio::LOCAL);
  BOOST_CHECK_EQUAL(data->Jc.rows(), 3);
  BOOST_CHECK_EQUAL(data->Jc.cols(), model->get_state()->get_nv());
  BOOST_CHECK_EQUAL(data->df_du.cols(), model->get_nu());
  BOOST_CHECK(model->get_gains().isApprox(gains));
  BOOST_CHECK(model->get_mask() == mask);

  const Data copied_data(*data);
  BOOST_CHECK(copied_data.Jc.isApprox(data->Jc));
  BOOST_CHECK(copied_data.pinocchio == data->pinocchio);

  BOOST_CHECK_THROW(
      std::make_shared<Model>(model->get_state(), 1,
                              create_placement1().template cast<Scalar>(), 2,
                              create_placement2().template cast<Scalar>(),
                              pinocchio::WORLD, model->get_state()->get_nv()),
      crocoddyl::Exception);
  const typename Model::MaskArray empty_mask = {
      {false, false, false, false, false, false}};
  BOOST_CHECK_THROW(
      std::make_shared<Model>(
          model->get_state(), 1, create_placement1().template cast<Scalar>(), 2,
          create_placement2().template cast<Scalar>(), pinocchio::LOCAL,
          model->get_state()->get_nv(), Model::Vector2s::Zero(), empty_mask),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      std::make_shared<Model>(
          model->get_state(), model->get_state()->get_pinocchio()->njoints,
          create_placement1().template cast<Scalar>(), 2,
          create_placement2().template cast<Scalar>(), pinocchio::LOCAL,
          model->get_state()->get_nv(), Model::Vector2s::Zero(), mask),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(model->createData(nullptr), crocoddyl::Exception);
  BOOST_CHECK_THROW(
      model->updateForce(data, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(
                                   model->get_nc() + 1)),
      crocoddyl::Exception);
}

template <typename Scalar>
void test_calc_and_mask_projection() {
  typedef crocoddyl::KinematicLoopModelTpl<Scalar> Model;
  typedef crocoddyl::KinematicLoopDataTpl<Scalar> Data;
  const typename Model::MaskArray full_mask = {
      {true, true, true, true, true, true}};
  const typename Model::MaskArray partial_mask = {
      {true, false, true, true, false, true}};
  typename Model::Vector2s gains;
  gains << Scalar(0.2), Scalar(0.3);
  const std::shared_ptr<Model> full_model =
      create_model<Scalar>(full_mask, gains);
  const std::shared_ptr<Model> partial_model =
      create_model<Scalar>(partial_mask, gains);
  pinocchio::DataTpl<Scalar> pinocchio_full(
      *full_model->get_state()->get_pinocchio());
  pinocchio::DataTpl<Scalar> pinocchio_partial(
      *partial_model->get_state()->get_pinocchio());
  const std::shared_ptr<Data> full_data =
      std::static_pointer_cast<Data>(full_model->createData(&pinocchio_full));
  const std::shared_ptr<Data> partial_data = std::static_pointer_cast<Data>(
      partial_model->createData(&pinocchio_partial));
  const typename Model::VectorXs x = full_model->get_state()->rand();
  const Scalar tolerance =
      std::is_same<Scalar, double>::value ? Scalar(1e-11) : Scalar(2e-4);

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
      BOOST_CHECK_SMALL(partial_data->a0[row] - full_data->a0[i], tolerance);
      BOOST_CHECK(partial_data->Jc.row(static_cast<Eigen::Index>(row))
                      .isApprox(full_data->Jc.row(static_cast<Eigen::Index>(i)),
                                tolerance));
      BOOST_CHECK(
          partial_data->da0_dx.row(static_cast<Eigen::Index>(row))
              .isApprox(full_data->da0_dx.row(static_cast<Eigen::Index>(i)),
                        tolerance));
      BOOST_CHECK(
          partial_data->dv0_dq.row(static_cast<Eigen::Index>(row))
              .isApprox(full_data->dv0_dq.row(static_cast<Eigen::Index>(i)),
                        tolerance));
      ++row;
    }
  }
}

template <typename Scalar>
void test_calc_diff_against_finite_differences() {
  typedef crocoddyl::KinematicLoopModelTpl<Scalar> Model;
  typedef crocoddyl::KinematicLoopDataTpl<Scalar> Data;
  typedef typename Model::VectorXs VectorXs;
  typedef typename Model::MatrixXs MatrixXs;
  const typename Model::MaskArray full_mask = {
      {true, true, true, true, true, true}};
  typename Model::Vector2s gains;
  gains << Scalar(0.2), Scalar(0.1);
  const std::shared_ptr<Model> model = create_model<Scalar>(full_mask, gains);
  pinocchio::DataTpl<Scalar> pinocchio_data(
      *model->get_state()->get_pinocchio());
  const std::shared_ptr<Data> data =
      std::static_pointer_cast<Data>(model->createData(&pinocchio_data));
  const VectorXs x = model->get_state()->rand();
  const std::size_t nv = model->get_state()->get_nv();
  const std::size_t ndx = model->get_state()->get_ndx();
  const Scalar eps =
      std::is_same<Scalar, double>::value ? Scalar(1e-7) : Scalar(2e-3);
  const Scalar tolerance =
      std::is_same<Scalar, double>::value ? Scalar(3e-4) : Scalar(4e-2);

  crocoddyl::unittest::updateAllPinocchio(
      model->get_state()->get_pinocchio().get(), &pinocchio_data, x);
  model->calc(data, x);
  model->calcDiff(data, x);

  MatrixXs da0_dx_fd(model->get_nc(), ndx);
  for (std::size_t i = 0; i < ndx; ++i) {
    VectorXs dx = VectorXs::Zero(ndx);
    dx[static_cast<Eigen::Index>(i)] = eps;
    VectorXs x_plus(model->get_state()->get_nx());
    VectorXs x_minus(model->get_state()->get_nx());
    model->get_state()->integrate(x, dx, x_plus);
    model->get_state()->integrate(x, -dx, x_minus);
    pinocchio::DataTpl<Scalar> pinocchio_plus(
        *model->get_state()->get_pinocchio());
    pinocchio::DataTpl<Scalar> pinocchio_minus(
        *model->get_state()->get_pinocchio());
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
        (data_plus->a0 - data_minus->a0) / (Scalar(2) * eps);
  }
  BOOST_CHECK(data->da0_dx.isApprox(da0_dx_fd, tolerance));

  MatrixXs dv0_dq_fd(model->get_nc(), nv);
  for (std::size_t i = 0; i < nv; ++i) {
    VectorXs dx = VectorXs::Zero(ndx);
    dx[static_cast<Eigen::Index>(i)] = eps;
    VectorXs x_plus(model->get_state()->get_nx());
    VectorXs x_minus(model->get_state()->get_nx());
    model->get_state()->integrate(x, dx, x_plus);
    model->get_state()->integrate(x, -dx, x_minus);
    pinocchio::DataTpl<Scalar> pinocchio_plus(
        *model->get_state()->get_pinocchio());
    pinocchio::DataTpl<Scalar> pinocchio_minus(
        *model->get_state()->get_pinocchio());
    dv0_dq_fd.col(static_cast<Eigen::Index>(i)).noalias() =
        (compute_velocity_drift(model, pinocchio_plus, x_plus) -
         compute_velocity_drift(model, pinocchio_minus, x_minus)) /
        (Scalar(2) * eps);
  }
  BOOST_CHECK_MESSAGE((data->dv0_dq - dv0_dq_fd).isZero(tolerance),
                      "dv0_dq error " << (data->dv0_dq - dv0_dq_fd).norm()
                                      << ", reference " << dv0_dq_fd.norm()
                                      << ", tolerance " << tolerance);
}

template <typename Scalar>
void test_update_force_and_multiple_scatter() {
  typedef crocoddyl::KinematicLoopModelTpl<Scalar> Model;
  typedef crocoddyl::KinematicLoopDataTpl<Scalar> Data;
  typedef crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar> Multiple;
  typedef crocoddyl::ImplicitConstraintDataMultipleTpl<Scalar> MultipleData;
  typedef typename Model::VectorXs VectorXs;
  typedef typename Model::MatrixXs MatrixXs;
  const typename Model::MaskArray mask = {
      {true, false, true, false, true, false}};
  const std::shared_ptr<Model> model = create_model<Scalar>(mask);
  pinocchio::DataTpl<Scalar> pinocchio_data(
      *model->get_state()->get_pinocchio());
  const std::shared_ptr<Data> data =
      std::static_pointer_cast<Data>(model->createData(&pinocchio_data));
  const VectorXs x = model->get_state()->rand();

  crocoddyl::unittest::updateAllPinocchio(
      model->get_state()->get_pinocchio().get(), &pinocchio_data, x);
  model->calc(data, x);
  const VectorXs force =
      VectorXs::LinSpaced(model->get_nc(), Scalar(1), Scalar(3));
  model->updateForce(data, force);
  BOOST_CHECK_SMALL(data->f.linear()[0] + force[0],
                    Eigen::NumTraits<Scalar>::dummy_precision());
  BOOST_CHECK_SMALL(data->f.angular()[1] + force[2],
                    Eigen::NumTraits<Scalar>::dummy_precision());
  BOOST_CHECK(data->joint1_f.toVector().isApprox(-data->fext.toVector()));
  BOOST_CHECK(!data->joint2_f.toVector().isZero());
  BOOST_CHECK(!data->dtau_dq.isZero());

  const MatrixXs df_dx =
      MatrixXs::Random(model->get_nc(), model->get_state()->get_ndx());
  const MatrixXs df_du = MatrixXs::Random(model->get_nc(), model->get_nu());
  model->updateForceDiff(data, df_dx, df_du);
  BOOST_CHECK(data->df_dx.isApprox(-df_dx));
  BOOST_CHECK(data->df_du.isApprox(-df_du));

  Multiple multiple(model->get_state(), model->get_nu());
  multiple.addConstraint("loop", model);
  const std::shared_ptr<MultipleData> multiple_data =
      multiple.createData(&pinocchio_data);
  multiple.calc(multiple_data, x);
  multiple.updateForce(multiple_data, force);
  BOOST_CHECK(multiple_data->fext[model->get_joint1_id()].toVector().isApprox(
      data->joint1_f.toVector()));
  BOOST_CHECK(multiple_data->fext[model->get_joint2_id()].toVector().isApprox(
      data->joint2_f.toVector()));
}

template <typename Scalar>
void test_tiny_nonzero_gains() {
  typedef crocoddyl::KinematicLoopModelTpl<Scalar> Model;
  typedef crocoddyl::KinematicLoopDataTpl<Scalar> Data;
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar> > state =
      create_scalar_state<Scalar>();
  const typename Model::MaskArray mask = {{true, true, true, true, true, true}};
  typename Model::Vector2s tiny_gains;
  tiny_gains.setConstant(std::numeric_limits<Scalar>::epsilon() / Scalar(4));
  typename Model::SE3 placement1 = create_placement1().template cast<Scalar>();
  placement1.translation()[0] =
      Scalar(1) / std::sqrt(std::numeric_limits<Scalar>::epsilon());
  const std::shared_ptr<Model> tiny_model = std::make_shared<Model>(
      state, 1, placement1, 2, create_placement2().template cast<Scalar>(),
      pinocchio::LOCAL, state->get_nv(), tiny_gains, mask);
  const std::shared_ptr<Model> zero_model = std::make_shared<Model>(
      state, 1, placement1, 2, create_placement2().template cast<Scalar>(),
      pinocchio::LOCAL, state->get_nv(), Model::Vector2s::Zero(), mask);
  pinocchio::DataTpl<Scalar> tiny_pinocchio(*state->get_pinocchio());
  pinocchio::DataTpl<Scalar> zero_pinocchio(*state->get_pinocchio());
  const std::shared_ptr<Data> tiny_data =
      std::static_pointer_cast<Data>(tiny_model->createData(&tiny_pinocchio));
  const std::shared_ptr<Data> zero_data =
      std::static_pointer_cast<Data>(zero_model->createData(&zero_pinocchio));
  const typename Model::VectorXs x = state->rand();
  crocoddyl::unittest::updateAllPinocchio(state->get_pinocchio().get(),
                                          &tiny_pinocchio, x);
  crocoddyl::unittest::updateAllPinocchio(state->get_pinocchio().get(),
                                          &zero_pinocchio, x);
  tiny_model->calc(tiny_data, x);
  zero_model->calc(zero_data, x);
  tiny_model->calcDiff(tiny_data, x);
  zero_model->calcDiff(zero_data, x);
  BOOST_CHECK((tiny_data->a0 - zero_data->a0).norm() > Scalar(0));
  BOOST_CHECK((tiny_data->da0_dx - zero_data->da0_dx).norm() > Scalar(0));
}

template <typename Scalar>
void test_hot_path_no_allocation() {
  typedef crocoddyl::KinematicLoopModelTpl<Scalar> Model;
  typedef crocoddyl::KinematicLoopDataTpl<Scalar> Data;
  typedef typename Model::VectorXs VectorXs;
  typedef typename Model::MatrixXs MatrixXs;
  const typename Model::MaskArray mask = {{true, true, true, true, true, true}};
  typename Model::Vector2s gains;
  gains << Scalar(0.2), Scalar(0.1);
  const std::shared_ptr<Model> model = create_model<Scalar>(mask, gains);
  pinocchio::DataTpl<Scalar> pinocchio_data(
      *model->get_state()->get_pinocchio());
  const std::shared_ptr<Data> data =
      std::static_pointer_cast<Data>(model->createData(&pinocchio_data));
  const VectorXs x = model->get_state()->rand();
  const VectorXs force =
      VectorXs::LinSpaced(model->get_nc(), Scalar(-1), Scalar(2));
  const MatrixXs df_dx =
      MatrixXs::Random(model->get_nc(), model->get_state()->get_ndx());
  const MatrixXs df_du = MatrixXs::Random(model->get_nc(), model->get_nu());
  crocoddyl::unittest::updateAllPinocchio(
      model->get_state()->get_pinocchio().get(), &pinocchio_data, x);
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

#ifdef NDEBUG
void test_cast() {
  typedef crocoddyl::KinematicLoopModel Model;
  const Model::MaskArray mask = {{true, false, true, true, false, true}};
  const std::shared_ptr<Model> model =
      create_model<double>(mask, (Model::Vector2s() << 0.4, 0.5).finished());
  const crocoddyl::KinematicLoopModelTpl<float> casted_model =
      model->cast<float>();
  BOOST_CHECK_EQUAL(casted_model.get_joint1_id(), model->get_joint1_id());
  BOOST_CHECK_EQUAL(casted_model.get_joint2_id(), model->get_joint2_id());
  BOOST_CHECK(casted_model.get_mask() == mask);
}
#endif

void test_construct_data_and_accessors_double() {
  test_construct_data_and_accessors<double>();
}
void test_construct_data_and_accessors_float() {
  test_construct_data_and_accessors<float>();
}
void test_calc_and_mask_projection_double() {
  test_calc_and_mask_projection<double>();
}
void test_calc_and_mask_projection_float() {
  test_calc_and_mask_projection<float>();
}
void test_calc_diff_against_finite_differences_double() {
  test_calc_diff_against_finite_differences<double>();
}
void test_calc_diff_against_finite_differences_float() {
  test_calc_diff_against_finite_differences<float>();
}
void test_update_force_and_multiple_scatter_double() {
  test_update_force_and_multiple_scatter<double>();
}
void test_update_force_and_multiple_scatter_float() {
  test_update_force_and_multiple_scatter<float>();
}
void test_tiny_nonzero_gains_double() { test_tiny_nonzero_gains<double>(); }
void test_tiny_nonzero_gains_float() { test_tiny_nonzero_gains<float>(); }
void test_hot_path_no_allocation_double() {
  test_hot_path_no_allocation<double>();
}
void test_hot_path_no_allocation_float() {
  test_hot_path_no_allocation<float>();
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_kinematic_loop");
  ts->add(BOOST_TEST_CASE(&test_construct_data_and_accessors_double));
  ts->add(BOOST_TEST_CASE(&test_construct_data_and_accessors_float));
  ts->add(BOOST_TEST_CASE(&test_calc_and_mask_projection_double));
  ts->add(BOOST_TEST_CASE(&test_calc_and_mask_projection_float));
  ts->add(BOOST_TEST_CASE(&test_calc_diff_against_finite_differences_double));
  ts->add(BOOST_TEST_CASE(&test_calc_diff_against_finite_differences_float));
  ts->add(BOOST_TEST_CASE(&test_update_force_and_multiple_scatter_double));
  ts->add(BOOST_TEST_CASE(&test_update_force_and_multiple_scatter_float));
  ts->add(BOOST_TEST_CASE(&test_tiny_nonzero_gains_double));
  ts->add(BOOST_TEST_CASE(&test_tiny_nonzero_gains_float));
  ts->add(BOOST_TEST_CASE(&test_hot_path_no_allocation_double));
  ts->add(BOOST_TEST_CASE(&test_hot_path_no_allocation_float));
#ifdef NDEBUG
  ts->add(BOOST_TEST_CASE(&test_cast));
#endif
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
