///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/data/observer.hpp"
#include "crocoddyl/core/observer-base.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;

namespace {

class DummyObserverModel : public crocoddyl::ObserverModelAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_BASE_DERIVED_CAST(crocoddyl::ActionModelBase, DummyObserverModel)

  typedef Eigen::VectorXd VectorXs;

  explicit DummyObserverModel(
      const std::shared_ptr<crocoddyl::StateVector>& state)
      : crocoddyl::ObserverModelAbstract(state, 2, 3, 0, 0, 0, 0, 0, 1),
        last_p(Eigen::VectorXd::Zero(1)) {}

  virtual void calc(const std::shared_ptr<crocoddyl::ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>&) override {
    data->xnext = x;
    data->cost = 0.;
  }

  virtual void calcDiff(
      const std::shared_ptr<crocoddyl::ActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>&,
      const Eigen::Ref<const VectorXs>&) override {
    data->Fx.setZero();
    data->Fu.setZero();
    data->Fp.setZero();
    data->Lx.setZero();
    data->Lu.setZero();
    data->Lp.setZero();
    data->Lxx.setZero();
    data->Lxu.setZero();
    data->Luu.setZero();
    data->Lpp.setZero();
    data->Lpx.setZero();
    data->Lpu.setZero();
  }

  virtual void update_p(const std::shared_ptr<crocoddyl::ActionDataAbstract>&,
                        const Eigen::Ref<const VectorXs>& p) override {
    if (static_cast<std::size_t>(p.size()) != get_np()) {
      throw_pretty(
          "Invalid argument: " << "p has wrong dimension (it should be " +
                                      std::to_string(get_np()) + ")");
    }
    last_p = p;
  }

  virtual bool checkData(
      const std::shared_ptr<crocoddyl::ActionDataAbstract>& data) override {
    return std::dynamic_pointer_cast<crocoddyl::ObserverDataAbstract>(data) !=
           nullptr;
  }

  Eigen::VectorXd last_p;
};

void test_observer_base_create_data_and_getters() {
  const std::shared_ptr<crocoddyl::StateVector> state =
      std::make_shared<crocoddyl::StateVector>(4);
  DummyObserverModel model(state);

  BOOST_CHECK_EQUAL(model.get_ntau(), 2);
  BOOST_CHECK_EQUAL(model.get_nu(), 3);
  BOOST_CHECK_EQUAL(model.get_np(), 1);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(model.get_tau_meas().size()), 2);
  BOOST_CHECK(model.get_tau_meas().isZero());

  const std::shared_ptr<crocoddyl::ActionDataAbstract> data =
      model.createData();
  BOOST_CHECK(model.checkData(data));

  const std::shared_ptr<crocoddyl::ObserverDataAbstract> observer_data =
      std::dynamic_pointer_cast<crocoddyl::ObserverDataAbstract>(data);
  BOOST_REQUIRE(observer_data != nullptr);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(observer_data->Fp.cols()), 1);
  BOOST_CHECK_EQUAL(
      static_cast<std::size_t>(observer_data->dissipative_E.size()), 1);
  BOOST_CHECK_EQUAL(observer_data->dE_dv.rows(), 1);
  BOOST_CHECK_EQUAL(observer_data->dE_dv.cols(), state->get_nv());
  BOOST_CHECK_EQUAL(observer_data->dE_dp.rows(), 1);
  BOOST_CHECK_EQUAL(observer_data->dE_dp.cols(), 1);
  BOOST_CHECK(observer_data->dissipative_E.isZero());
  BOOST_CHECK(observer_data->dE_dv.isZero());
  BOOST_CHECK(observer_data->dE_dp.isZero());

  crocoddyl::DataCollectorObserverTpl<double> shared;
  BOOST_CHECK(!shared.hasObserverData());
  shared.shareObserverData(observer_data.get());
  BOOST_CHECK(shared.hasObserverData());
  BOOST_CHECK(shared.xnext != nullptr);
  BOOST_CHECK(shared.int_Fx != nullptr);
  BOOST_CHECK(shared.int_Fu != nullptr);
  BOOST_CHECK(shared.int_Fp != nullptr);
  BOOST_CHECK(shared.dissipative_E != nullptr);
  BOOST_CHECK(shared.dE_dv != nullptr);
  BOOST_CHECK(shared.dE_dp != nullptr);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(shared.dissipative_E->size()), 1);
}

void test_observer_base_update_tau_and_parameters() {
  const std::shared_ptr<crocoddyl::StateVector> state =
      std::make_shared<crocoddyl::StateVector>(4);
  DummyObserverModel model(state);
  const std::shared_ptr<crocoddyl::ActionDataAbstract> data =
      model.createData();

  Eigen::VectorXd tau(2);
  tau << 1., -2.;
  model.update_tau(tau);
  BOOST_CHECK(model.get_tau_meas().isApprox(tau));

  Eigen::VectorXd p(1);
  p << 3.;
  model.update_p(data, p);
  BOOST_CHECK(model.last_p.isApprox(p));

  BOOST_CHECK_THROW(model.update_tau(Eigen::VectorXd::Zero(1)),
                    crocoddyl::Exception);
  BOOST_CHECK_THROW(model.update_p(data, Eigen::VectorXd::Zero(2)),
                    crocoddyl::Exception);
}

}  // namespace

void register_unit_tests() {
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_observer_base_create_data_and_getters));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_observer_base_update_tau_and_parameters));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
