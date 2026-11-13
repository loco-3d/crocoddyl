///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <algorithm>
#include <sstream>

#include "crocoddyl/multibody/implicit-constraint-base.hpp"
#include "factory/state.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

namespace {

class DummyImplicitConstraintModel
    : public crocoddyl::ImplicitConstraintModelAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_BASE_DERIVED_CAST(crocoddyl::ImplicitConstraintModelBase,
                              DummyImplicitConstraintModel)

  typedef crocoddyl::ImplicitConstraintDataAbstract Data;
  typedef crocoddyl::ImplicitConstraintModelAbstract Base;
  typedef Base::MatrixXs MatrixXs;
  typedef Base::VectorXs VectorXs;

  DummyImplicitConstraintModel(
      const std::shared_ptr<crocoddyl::StateMultibody>& state,
      const pinocchio::ReferenceFrame type, const std::size_t nc,
      const std::size_t nu)
      : Base(state, type, nc, nu) {}

  void calc(const std::shared_ptr<Data>& data,
            const Eigen::Ref<const VectorXs>& x) override {
    data->Jc.setZero();
    data->Jc.leftCols(get_nc()).setIdentity();
    data->a0 = x.head(get_nc());
    data->dv0_dq.setZero();
    data->dtau_dq.setZero();
  }

  void calcDiff(const std::shared_ptr<Data>& data,
                const Eigen::Ref<const VectorXs>&) override {
    data->da0_dx.setZero();
    data->da0_dx.leftCols(get_nc()).setIdentity();
    data->dv0_dq.setZero();
    data->dv0_dq.leftCols(get_nc()).setIdentity();
    data->dtau_dq.setIdentity();
  }

  void updateForce(const std::shared_ptr<Data>& data,
                   const VectorXs& force) override {
    data->f.linear() = Eigen::Vector3d::Zero();
    data->f.angular() = Eigen::Vector3d::Zero();
    data->f.linear().head(std::min<std::size_t>(3, get_nc())) =
        force.head(std::min<std::size_t>(3, get_nc()));
    data->fext = data->f;
  }
};

std::shared_ptr<DummyImplicitConstraintModel> create_model() {
  const std::shared_ptr<crocoddyl::StateMultibody> state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(
          StateModelFactory().create(StateModelTypes::StateMultibody_TalosArm));
  return std::make_shared<DummyImplicitConstraintModel>(
      state, pinocchio::ReferenceFrame::LOCAL, 3, 2);
}

void test_construct_data() {
  const std::shared_ptr<DummyImplicitConstraintModel> model = create_model();
  const std::shared_ptr<pinocchio::Model>& pinocchio_model =
      model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model);
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract> data =
      model->createData(&pinocchio_data);

  std::ostringstream tmp;
  tmp << *model;

  BOOST_CHECK_EQUAL(data->Jc.rows(), 3);
  BOOST_CHECK_EQUAL(data->Jc.cols(), model->get_state()->get_nv());
  BOOST_CHECK_EQUAL(data->df_du.rows(), 3);
  BOOST_CHECK_EQUAL(data->df_du.cols(), 2);
  BOOST_CHECK_EQUAL(data->da0_dx.rows(), 3);
  BOOST_CHECK_EQUAL(data->da0_dx.cols(), model->get_state()->get_ndx());
  BOOST_CHECK_EQUAL(data->dv0_dq.rows(), 3);
  BOOST_CHECK_EQUAL(data->dv0_dq.cols(), model->get_state()->get_nv());
  BOOST_CHECK_THROW(
      crocoddyl::ImplicitConstraintDataAbstract(
          static_cast<DummyImplicitConstraintModel*>(nullptr), &pinocchio_data),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(
      crocoddyl::ImplicitConstraintDataAbstract(model.get(), nullptr),
      crocoddyl::Exception);
  BOOST_CHECK_THROW(model->createData(nullptr), crocoddyl::Exception);
}

void test_calc_and_calc_diff() {
  const std::shared_ptr<DummyImplicitConstraintModel> model = create_model();
  const std::shared_ptr<pinocchio::Model>& pinocchio_model =
      model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model);
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract> data =
      model->createData(&pinocchio_data);
  const Eigen::VectorXd x = model->get_state()->rand();

  model->calc(data, x);
  BOOST_CHECK(!data->Jc.isZero());
  BOOST_CHECK((data->a0 - x.head(model->get_nc())).isZero(1e-12));
  BOOST_CHECK(data->da0_dx.isZero());
  BOOST_CHECK(data->df_dx.isZero());
  BOOST_CHECK(data->df_du.isZero());

  model->calcDiff(data, x);
  BOOST_CHECK(!data->da0_dx.isZero());
  BOOST_CHECK(!data->dv0_dq.isZero());
  BOOST_CHECK(!data->dtau_dq.isZero());
}

void test_force_helpers() {
  const std::shared_ptr<DummyImplicitConstraintModel> model = create_model();
  const std::shared_ptr<pinocchio::Model>& pinocchio_model =
      model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model);
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract> data =
      model->createData(&pinocchio_data);

  const Eigen::VectorXd force = Eigen::VectorXd::Random(model->get_nc());
  model->updateForce(data, force);
  BOOST_CHECK(!data->f.toVector().isZero());
  BOOST_CHECK(!data->fext.toVector().isZero());

  const Eigen::MatrixXd df_dx =
      Eigen::MatrixXd::Random(model->get_nc(), model->get_state()->get_ndx());
  const Eigen::MatrixXd df_du =
      Eigen::MatrixXd::Random(model->get_nc(), model->get_nu());
  model->updateForceDiff(data, df_dx, df_du);
  BOOST_CHECK((data->df_dx - df_dx).isZero(1e-12));
  BOOST_CHECK((data->df_du - df_du).isZero(1e-12));

  model->setZeroForce(data);
  model->setZeroForceDiff(data);
  BOOST_CHECK(data->f.toVector().isZero());
  BOOST_CHECK(data->fext.toVector().isZero());
  BOOST_CHECK(data->df_dx.isZero());
  BOOST_CHECK(data->df_du.isZero());
}

void test_accessors_and_dimension_checks() {
  const std::shared_ptr<DummyImplicitConstraintModel> model = create_model();
  const std::shared_ptr<pinocchio::Model>& pinocchio_model =
      model->get_state()->get_pinocchio();
  pinocchio::Data pinocchio_data(*pinocchio_model);
  const std::shared_ptr<crocoddyl::ImplicitConstraintDataAbstract> data =
      model->createData(&pinocchio_data);

  BOOST_CHECK_EQUAL(model->get_nc(), 3);
  BOOST_CHECK_EQUAL(model->get_nu(), 2);
  BOOST_CHECK(model->get_type() == pinocchio::ReferenceFrame::LOCAL);
  model->set_id(7);
  model->set_type(pinocchio::ReferenceFrame::WORLD);
  BOOST_CHECK_EQUAL(model->get_id(), 7);
  BOOST_CHECK(model->get_type() == pinocchio::ReferenceFrame::WORLD);

  const Eigen::MatrixXd wrong_df_dx =
      Eigen::MatrixXd::Zero(model->get_nc() + 1, model->get_state()->get_ndx());
  const Eigen::MatrixXd wrong_df_du =
      Eigen::MatrixXd::Zero(model->get_nc(), model->get_nu() + 1);
  BOOST_CHECK_THROW(model->updateForceDiff(data, wrong_df_dx, data->df_du),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(model->updateForceDiff(data, data->df_dx, wrong_df_du),
                    crocoddyl::Exception);
}

}  // namespace

bool init_function() {
  test_suite* ts = BOOST_TEST_SUITE("test_implicit_constraints");
  ts->add(BOOST_TEST_CASE(&test_construct_data));
  ts->add(BOOST_TEST_CASE(&test_calc_and_calc_diff));
  ts->add(BOOST_TEST_CASE(&test_force_helpers));
  ts->add(BOOST_TEST_CASE(&test_accessors_and_dimension_checks));
  framework::master_test_suite().add(ts);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
