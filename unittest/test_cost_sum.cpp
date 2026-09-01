///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "factory/cost.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

template <typename _Scalar>
class CostSumParameterResidualTpl
    : public crocoddyl::ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(crocoddyl::ResidualModelBase,
                         CostSumParameterResidualTpl)

  typedef _Scalar Scalar;
  typedef crocoddyl::ResidualModelAbstractTpl<Scalar> Base;
  typedef typename Base::ResidualDataAbstract ResidualDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;

  CostSumParameterResidualTpl(std::shared_ptr<StateAbstract> state,
                              const std::size_t nu, const std::size_t np)
      : Base(state, 2, nu, true, true, true, np) {}

  void calc(const std::shared_ptr<ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>&,
            const Eigen::Ref<const VectorXs>&) override {
    data->r << Scalar(1), Scalar(-2);
  }

  void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>&,
                const Eigen::Ref<const VectorXs>&) override {
    data->Rx.setConstant(Scalar(0.25));
    data->Ru.setConstant(Scalar(-0.5));
    if (this->get_np() != 0) {
      data->Rp << Scalar(1), Scalar(2), Scalar(-1), Scalar(3);
    }
  }

  template <typename NewScalar>
  CostSumParameterResidualTpl<NewScalar> cast() const {
    typedef CostSumParameterResidualTpl<NewScalar> ReturnType;
    return ReturnType(this->get_state()->template cast<NewScalar>(),
                      this->get_nu(), this->get_np());
  }
};

template <typename _Scalar>
struct CostSumActionShapeTpl {
  typedef _Scalar Scalar;
  typedef crocoddyl::StateVectorTpl<Scalar> State;

  CostSumActionShapeTpl(const std::size_t nx, const std::size_t nu,
                        const std::size_t np)
      : state(std::make_shared<State>(nx)), nu(nu), np(np) {}

  const std::shared_ptr<State>& get_state() const { return state; }
  std::size_t get_nu() const { return nu; }
  std::size_t get_np() const { return np; }
  std::size_t get_nr() const { return 0; }
  std::size_t get_ng() const { return 0; }
  std::size_t get_nh() const { return 0; }
  std::size_t get_ng_T() const { return 0; }
  std::size_t get_nh_T() const { return 0; }

  std::shared_ptr<State> state;
  std::size_t nu;
  std::size_t np;
};

typedef CostSumParameterResidualTpl<double> CostSumParameterResidual;
typedef CostSumActionShapeTpl<double> CostSumActionShape;

void test_parameter_aggregation_and_terminal_lifecycle() {
  const std::shared_ptr<crocoddyl::StateVector> state =
      std::make_shared<crocoddyl::StateVector>(4);
  const std::size_t nu = 2;
  const std::size_t np = 2;
  const std::shared_ptr<CostSumParameterResidual> residual =
      std::make_shared<CostSumParameterResidual>(state, nu, np);
  const std::shared_ptr<crocoddyl::CostModelResidual> cost =
      std::make_shared<crocoddyl::CostModelResidual>(state, residual);
  crocoddyl::CostModelSum model(state, nu, np);
  model.addCost("active", cost, 2.);
  model.addCost("inactive", cost, 3., false);
  crocoddyl::DataCollectorAbstract shared;
  const std::shared_ptr<crocoddyl::CostDataSum> data =
      model.createData(&shared);
  const Eigen::VectorXd x = state->rand();
  const Eigen::VectorXd u = Eigen::VectorXd::Random(nu);

  BOOST_CHECK_EQUAL(model.get_np(), np);
  BOOST_CHECK_EQUAL(data->Lp.size(), np);
  BOOST_CHECK_EQUAL(data->Lpp.rows(), np);
  BOOST_CHECK_EQUAL(data->Lpp.cols(), np);
  BOOST_CHECK_EQUAL(data->Lpx.rows(), np);
  BOOST_CHECK_EQUAL(data->Lpx.cols(), state->get_ndx());
  BOOST_CHECK_EQUAL(data->Lpu.rows(), np);
  BOOST_CHECK_EQUAL(data->Lpu.cols(), nu);
  BOOST_CHECK(data->Lp.isZero(0.));
  BOOST_CHECK(data->Lpp.isZero(0.));
  BOOST_CHECK(data->Lpx.isZero(0.));
  BOOST_CHECK(data->Lpu.isZero(0.));

  model.calc(data, x, u);
  model.calcDiff(data, x, u);
  const std::shared_ptr<crocoddyl::CostDataAbstract>& active_data =
      data->costs.find("active")->second;
  BOOST_CHECK(data->Lp.isApprox(2. * active_data->Lp, 1e-12));
  BOOST_CHECK(data->Lpp.isApprox(2. * active_data->Lpp, 1e-12));
  BOOST_CHECK(data->Lpx.isApprox(2. * active_data->Lpx, 1e-12));
  BOOST_CHECK(data->Lpu.isApprox(2. * active_data->Lpu, 1e-12));

  model.changeCostStatus("inactive", true);
  data->Lu.setConstant(41.);
  data->Luu.setConstant(42.);
  data->Lxu.setConstant(43.);
  data->Lpu.setConstant(44.);
  model.calc(data, x);
  model.calcDiff(data, x);
  BOOST_CHECK(data->Lp.isApprox(5. * active_data->Lp, 1e-12));
  BOOST_CHECK(data->Lpp.isApprox(5. * active_data->Lpp, 1e-12));
  BOOST_CHECK(data->Lpx.isApprox(5. * active_data->Lpx, 1e-12));
  BOOST_CHECK(data->Lu.isConstant(41., 0.));
  BOOST_CHECK(data->Luu.isConstant(42., 0.));
  BOOST_CHECK(data->Lxu.isConstant(43., 0.));
  BOOST_CHECK(data->Lpu.isConstant(44., 0.));

#ifdef NDEBUG  // Run only in release mode
  crocoddyl::CostModelSumTpl<float> casted_model = model.cast<float>();
  BOOST_CHECK_EQUAL(casted_model.get_np(), np);
  BOOST_CHECK_EQUAL(casted_model.get_costs().size(), model.get_costs().size());
  crocoddyl::DataCollectorAbstractTpl<float> casted_shared;
  const std::shared_ptr<crocoddyl::CostDataSumTpl<float>> casted_data =
      casted_model.createData(&casted_shared);
  const Eigen::VectorXf x_f = x.cast<float>();
  const Eigen::VectorXf u_f = u.cast<float>();
  casted_model.calc(casted_data, x_f, u_f);
  casted_model.calcDiff(casted_data, x_f, u_f);
  BOOST_CHECK(casted_data->Lp.isApprox(data->Lp.cast<float>(), 1e-5f));
  BOOST_CHECK(casted_data->Lpp.isApprox(data->Lpp.cast<float>(), 1e-5f));
  BOOST_CHECK(casted_data->Lpx.isApprox(data->Lpx.cast<float>(), 1e-5f));
#endif
}

void test_parameter_dimension_validation() {
  const std::shared_ptr<crocoddyl::StateVector> state =
      std::make_shared<crocoddyl::StateVector>(4);
  const std::shared_ptr<CostSumParameterResidual> residual =
      std::make_shared<CostSumParameterResidual>(state, 2, 2);
  const std::shared_ptr<crocoddyl::CostModelResidual> cost =
      std::make_shared<crocoddyl::CostModelResidual>(state, residual);
  crocoddyl::CostModelSum model(state, 2, 3);
  BOOST_CHECK_THROW(model.addCost("invalid", cost, 1.), std::exception);
  const std::shared_ptr<crocoddyl::CostItem> item =
      std::make_shared<crocoddyl::CostItem>("invalid", cost, 1.);
  BOOST_CHECK_THROW(model.addCost(item), std::exception);

  const std::shared_ptr<CostSumParameterResidual> parameter_free_residual =
      std::make_shared<CostSumParameterResidual>(state, 2, 0);
  const std::shared_ptr<crocoddyl::CostModelResidual> parameter_free_cost =
      std::make_shared<crocoddyl::CostModelResidual>(state,
                                                     parameter_free_residual);
  BOOST_CHECK_NO_THROW(
      model.addCost("parameter_free", parameter_free_cost, 1.));
}

void test_parameter_share_memory() {
  const std::shared_ptr<crocoddyl::StateVector> state =
      std::make_shared<crocoddyl::StateVector>(4);
  crocoddyl::CostModelSum model(state, 2, 2);
  crocoddyl::DataCollectorAbstract shared;
  const std::shared_ptr<crocoddyl::CostDataSum> data =
      model.createData(&shared);
  CostSumActionShape action_model(4, 2, 2);
  crocoddyl::ActionDataAbstract action_data(&action_model);
  data->shareMemory(&action_data);
  data->Lp.setConstant(1.);
  data->Lpp.setConstant(2.);
  data->Lpx.setConstant(3.);
  data->Lpu.setConstant(4.);
  BOOST_CHECK(action_data.Lp.isApprox(data->Lp, 0.));
  BOOST_CHECK(action_data.Lpp.isApprox(data->Lpp, 0.));
  BOOST_CHECK(action_data.Lpx.isApprox(data->Lpx, 0.));
  BOOST_CHECK(action_data.Lpu.isApprox(data->Lpu, 0.));

  CostSumActionShape invalid_action_model(4, 2, 3);
  crocoddyl::ActionDataAbstract invalid_action_data(&invalid_action_model);
  BOOST_CHECK_THROW(data->shareMemory(&invalid_action_data), std::exception);

  crocoddyl::CostModelSum parameter_free_model(state, 2);
  const std::shared_ptr<crocoddyl::CostDataSum> parameter_free_data =
      parameter_free_model.createData(&shared);
  CostSumActionShape terminal_action_model(4, 0, 0);
  crocoddyl::ActionDataAbstract terminal_action_data(&terminal_action_model);
  BOOST_CHECK_NO_THROW(parameter_free_data->shareMemory(&terminal_action_data));
  BOOST_CHECK_EQUAL(parameter_free_data->Lpu.rows(), 0);
  BOOST_CHECK_EQUAL(parameter_free_data->Lpu.cols(), 2);
}

void test_parameter_setters_and_no_allocation() {
  const std::shared_ptr<crocoddyl::StateVector> state =
      std::make_shared<crocoddyl::StateVector>(4);
  const std::shared_ptr<CostSumParameterResidual> residual =
      std::make_shared<CostSumParameterResidual>(state, 2, 2);
  const std::shared_ptr<crocoddyl::CostModelResidual> cost =
      std::make_shared<crocoddyl::CostModelResidual>(state, residual);
  crocoddyl::CostModelSum model(state, 2, 2);
  model.addCost("cost", cost, 1.);
  crocoddyl::DataCollectorAbstract shared;
  const std::shared_ptr<crocoddyl::CostDataSum> data =
      model.createData(&shared);
  const Eigen::VectorXd x = state->rand();
  const Eigen::Vector2d u = Eigen::Vector2d::Random();

  BOOST_CHECK_NO_THROW(data->set_Lp(Eigen::Vector2d::Ones()));
  BOOST_CHECK_NO_THROW(data->set_Lpp(Eigen::Matrix2d::Ones()));
  BOOST_CHECK_NO_THROW(data->set_Lpx(Eigen::MatrixXd::Ones(2, 4)));
  BOOST_CHECK_NO_THROW(data->set_Lpu(Eigen::Matrix2d::Ones()));
  BOOST_CHECK_THROW(data->set_Lp(Eigen::Vector3d::Zero()), std::exception);
  BOOST_CHECK_THROW(data->set_Lpp(Eigen::Matrix3d::Zero()), std::exception);
  BOOST_CHECK_THROW(data->set_Lpx(Eigen::MatrixXd::Zero(3, 4)), std::exception);
  BOOST_CHECK_THROW(data->set_Lpu(Eigen::MatrixXd::Zero(2, 3)), std::exception);

  model.calc(data, x, u);
  model.calcDiff(data, x, u);
  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      model.calc(data, x, u);
      model.calcDiff(data, x, u);
      model.calc(data, x);
      model.calcDiff(data, x);
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
}

//----------------------------------------------------------------------------//

void test_constructor(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(state_factory.create(state_type));

  // Run the print function
  std::ostringstream tmp;
  tmp << model;

  // Test the initial size of the map
  BOOST_CHECK(model.get_costs().size() == 0);

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  crocoddyl::CostModelSumTpl<float> casted_model = model.cast<float>();
  BOOST_CHECK(casted_model.get_costs().size() == 0);
#endif
}

void test_addCost(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(state_factory.create(state_type));
  crocoddyl::CostModelSumTpl<float> casted_model = model.cast<float>();

  // add an active cost
  std::shared_ptr<crocoddyl::CostModelAbstract> rand_cost_1 =
      create_random_cost(state_type);
  model.addCost("random_cost_1", rand_cost_1, 1.);
  BOOST_CHECK(model.get_nr() == rand_cost_1->get_activation()->get_nr());
  BOOST_CHECK(model.get_nr_total() == rand_cost_1->get_activation()->get_nr());

  // add an inactive cost
  std::shared_ptr<crocoddyl::CostModelAbstract> rand_cost_2 =
      create_random_cost(state_type);
  model.addCost("random_cost_2", rand_cost_2, 1., false);
  BOOST_CHECK(model.get_nr() == rand_cost_1->get_activation()->get_nr());
  BOOST_CHECK(model.get_nr_total() ==
              rand_cost_1->get_activation()->get_nr() +
                  rand_cost_2->get_activation()->get_nr());

  // change the random cost 2 status
  model.changeCostStatus("random_cost_2", true);
  BOOST_CHECK(model.get_nr() == rand_cost_1->get_activation()->get_nr() +
                                    rand_cost_2->get_activation()->get_nr());
  BOOST_CHECK(model.get_nr_total() ==
              rand_cost_1->get_activation()->get_nr() +
                  rand_cost_2->get_activation()->get_nr());

  // change the random cost 1 status
  model.changeCostStatus("random_cost_1", false);
  BOOST_CHECK(model.get_nr() == rand_cost_2->get_activation()->get_nr());
  BOOST_CHECK(model.get_nr_total() ==
              rand_cost_1->get_activation()->get_nr() +
                  rand_cost_2->get_activation()->get_nr());

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::CostModelAbstractTpl<float>> casted_rand_cost_1 =
      rand_cost_1->cast<float>();
  casted_model.addCost("random_cost_1", casted_rand_cost_1, 1.f);
  BOOST_CHECK(casted_model.get_nr() ==
              casted_rand_cost_1->get_activation()->get_nr());
  BOOST_CHECK(casted_model.get_nr_total() ==
              casted_rand_cost_1->get_activation()->get_nr());
  std::shared_ptr<crocoddyl::CostModelAbstractTpl<float>> casted_rand_cost_2 =
      rand_cost_2->cast<float>();
  casted_model.addCost("random_cost_2", casted_rand_cost_2, 1.f, false);
  BOOST_CHECK(casted_model.get_nr() ==
              casted_rand_cost_1->get_activation()->get_nr());
  BOOST_CHECK(casted_model.get_nr_total() ==
              casted_rand_cost_1->get_activation()->get_nr() +
                  casted_rand_cost_2->get_activation()->get_nr());
  casted_model.changeCostStatus("random_cost_2", true);
  BOOST_CHECK(casted_model.get_nr() ==
              casted_rand_cost_1->get_activation()->get_nr() +
                  casted_rand_cost_2->get_activation()->get_nr());
  BOOST_CHECK(casted_model.get_nr_total() ==
              casted_rand_cost_1->get_activation()->get_nr() +
                  casted_rand_cost_2->get_activation()->get_nr());
  casted_model.changeCostStatus("random_cost_1", false);
  BOOST_CHECK(casted_model.get_nr() ==
              casted_rand_cost_2->get_activation()->get_nr());
  BOOST_CHECK(casted_model.get_nr_total() ==
              casted_rand_cost_1->get_activation()->get_nr() +
                  casted_rand_cost_2->get_activation()->get_nr());
#endif
}

void test_addCost_error_message(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(state_factory.create(state_type));

  // create an cost object
  std::shared_ptr<crocoddyl::CostModelAbstract> rand_cost =
      create_random_cost(state_type);

  // add twice the same cost object to the container
  model.addCost("random_cost", rand_cost, 1.);

  // test error message when we add a duplicate cost
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.addCost("random_cost", rand_cost, 1.);
  capture_ios.endCapture();
  std::stringstream expected_buffer;
  expected_buffer << "Warning: we couldn't add the random_cost cost item, it "
                     "already existed."
                  << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());

  // test error message when we change the cost status of an inexistent cost
  capture_ios.beginCapture();
  model.changeCostStatus("no_exist_cost", true);
  capture_ios.endCapture();
  expected_buffer.clear();
  expected_buffer << "Warning: we couldn't change the status of the "
                     "no_exist_cost cost item, it doesn't exist."
                  << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_removeCost(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(state_factory.create(state_type));
  crocoddyl::CostModelSumTpl<float> casted_model = model.cast<float>();

  // add an active cost
  std::shared_ptr<crocoddyl::CostModelAbstract> rand_cost =
      create_random_cost(state_type);
  model.addCost("random_cost", rand_cost, 1.);
  BOOST_CHECK(model.get_nr() == rand_cost->get_activation()->get_nr());

  // remove the cost
  model.removeCost("random_cost");
  BOOST_CHECK(model.get_nr() == 0);
  BOOST_CHECK(model.get_nr_total() == 0);

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  std::shared_ptr<crocoddyl::CostModelAbstractTpl<float>> casted_rand_cost =
      rand_cost->cast<float>();
  casted_model.addCost("random_cost", casted_rand_cost, 1.f);
  BOOST_CHECK(casted_model.get_nr() ==
              casted_rand_cost->get_activation()->get_nr());
  casted_model.removeCost("random_cost");
  BOOST_CHECK(casted_model.get_nr() == 0);
  BOOST_CHECK(casted_model.get_nr_total() == 0);
#endif
}

void test_removeCost_error_message(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(state_factory.create(state_type));

  // remove a none existing cost form the container, we expect a cout message
  // here
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.removeCost("random_cost");
  capture_ios.endCapture();

  // Test that the error message is sent.
  std::stringstream expected_buffer;
  expected_buffer << "Warning: we couldn't remove the random_cost cost item, "
                     "it doesn't exist."
                  << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_calc(StateModelTypes::Type state_type) {
  // setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(state_factory.create(state_type));
  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model.get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);

  // create and add some cost objects
  std::vector<std::shared_ptr<crocoddyl::CostModelAbstract>> models;
  std::vector<std::shared_ptr<crocoddyl::CostDataAbstract>> datas;
  for (std::size_t i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_cost_" << i;
    const std::shared_ptr<crocoddyl::CostModelAbstract>& m =
        create_random_cost(state_type);
    model.addCost(os.str(), m, 1.);
    models.push_back(m);
    datas.push_back(m->createData(&shared_data));
  }

  // create the data of the cost sum
  const std::shared_ptr<crocoddyl::CostDataSum>& data =
      model.createData(&shared_data);

  // compute the cost sum data for the case when all costs are defined as active
  const Eigen::VectorXd x1 = state->rand();
  const Eigen::VectorXd u1 = Eigen::VectorXd::Random(model.get_nu());
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x1);
  model.calc(data, x1, u1);

  // check that the cost has been filled
  BOOST_CHECK(data->cost > 0.);

  // check the cost against single cost computations
  double cost = 0;
  for (std::size_t i = 0; i < 5; ++i) {
    models[i]->calc(datas[i], x1, u1);
    cost += datas[i]->cost;
  }
  BOOST_CHECK(data->cost == cost);

  // compute the cost sum data for the case when the first three costs are
  // defined as active
  model.changeCostStatus("random_cost_3", false);
  model.changeCostStatus("random_cost_4", false);
  const Eigen::VectorXd x2 = state->rand();
  const Eigen::VectorXd u2 = Eigen::VectorXd::Random(model.get_nu());
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x1);
  model.calc(data, x2, u2);
  cost = 0;
  for (std::size_t i = 0; i < 3;
       ++i) {  // we need to update data because this costs are active
    models[i]->calc(datas[i], x2, u2);
    cost += datas[i]->cost;
  }
  BOOST_CHECK(data->cost == cost);

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  model.changeCostStatus("random_cost_3", true);
  model.changeCostStatus("random_cost_4", true);
  crocoddyl::CostModelSumTpl<float> casted_model = model.cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model.get_state());
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(casted_pinocchio_model);
  crocoddyl::DataCollectorMultibodyTpl<float> casted_shared_data(
      &casted_pinocchio_data);
  std::vector<std::shared_ptr<crocoddyl::CostModelAbstractTpl<float>>>
      casted_models;
  std::vector<std::shared_ptr<crocoddyl::CostDataAbstractTpl<float>>>
      casted_datas;
  for (std::size_t i = 0; i < 5; ++i) {
    casted_models.push_back(models[i]->cast<float>());
    casted_datas.push_back(casted_models[i]->createData(&casted_shared_data));
  }
  const std::shared_ptr<crocoddyl::CostDataSumTpl<float>>& casted_data =
      casted_model.createData(&casted_shared_data);
  const Eigen::VectorXf x1_f = x1.cast<float>();
  const Eigen::VectorXf u1_f = u1.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x1);
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x1_f);
  model.calc(data, x1, u1);
  casted_model.calc(casted_data, x1_f, u1_f);
  BOOST_CHECK(casted_data->cost > 0.f);
  float tol_f = std::sqrt(2.0f * std::numeric_limits<float>::epsilon());
  BOOST_CHECK(std::abs(float(data->cost) - casted_data->cost) <= tol_f);
  float cost_f = 0.f;
  for (std::size_t i = 0; i < 5; ++i) {
    casted_models[i]->calc(casted_datas[i], x1_f, u1_f);
    cost_f += casted_datas[i]->cost;
  }
  BOOST_CHECK(casted_data->cost == cost_f);
#endif
}

void test_calcDiff(StateModelTypes::Type state_type) {
  // setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(state_factory.create(state_type));
  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model.get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);

  // create and add some cost objects
  std::vector<std::shared_ptr<crocoddyl::CostModelAbstract>> models;
  std::vector<std::shared_ptr<crocoddyl::CostDataAbstract>> datas;
  for (std::size_t i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_cost_" << i;
    const std::shared_ptr<crocoddyl::CostModelAbstract>& m =
        create_random_cost(state_type);
    model.addCost(os.str(), m, 1.);
    models.push_back(m);
    datas.push_back(m->createData(&shared_data));
  }

  // create the data of the cost sum
  const std::shared_ptr<crocoddyl::CostDataSum>& data =
      model.createData(&shared_data);

  // compute the cost sum data for the case when all costs are defined as active
  Eigen::VectorXd x1 = state->rand();
  const Eigen::VectorXd u1 = Eigen::VectorXd::Random(model.get_nu());
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x1);
  model.calc(data, x1, u1);
  model.calcDiff(data, x1, u1);

  // check that the cost has been filled
  BOOST_CHECK(data->cost > 0.);

  // check the cost against single cost computations
  double cost = 0;
  Eigen::VectorXd Lx = Eigen::VectorXd::Zero(state->get_ndx());
  Eigen::VectorXd Lu = Eigen::VectorXd::Zero(model.get_nu());
  Eigen::MatrixXd Lxx =
      Eigen::MatrixXd::Zero(state->get_ndx(), state->get_ndx());
  Eigen::MatrixXd Lxu = Eigen::MatrixXd::Zero(state->get_ndx(), model.get_nu());
  Eigen::MatrixXd Luu = Eigen::MatrixXd::Zero(model.get_nu(), model.get_nu());
  for (std::size_t i = 0; i < 5; ++i) {
    models[i]->calc(datas[i], x1, u1);
    models[i]->calcDiff(datas[i], x1, u1);
    cost += datas[i]->cost;
    Lx += datas[i]->Lx;
    Lu += datas[i]->Lu;
    Lxx += datas[i]->Lxx;
    Lxu += datas[i]->Lxu;
    Luu += datas[i]->Luu;
  }
  BOOST_CHECK(data->cost == cost);
  BOOST_CHECK(data->Lx == Lx);
  BOOST_CHECK(data->Lu == Lu);
  BOOST_CHECK(data->Lxx == Lxx);
  BOOST_CHECK(data->Lxu == Lxu);
  BOOST_CHECK(data->Luu == Luu);

  x1 = state->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x1);
  model.calc(data, x1);
  model.calcDiff(data, x1);
  cost = 0.;
  Lx.setZero();
  Lxx.setZero();
  for (std::size_t i = 0; i < 5; ++i) {
    models[i]->calc(datas[i], x1);
    models[i]->calcDiff(datas[i], x1);
    cost += datas[i]->cost;
    Lx += datas[i]->Lx;
    Lxx += datas[i]->Lxx;
  }
  BOOST_CHECK(data->cost == cost);
  BOOST_CHECK(data->Lx == Lx);
  BOOST_CHECK(data->Lxx == Lxx);

  // compute the cost sum data for the case when the first three costs are
  // defined as active
  model.changeCostStatus("random_cost_3", false);
  model.changeCostStatus("random_cost_4", false);
  Eigen::VectorXd x2 = state->rand();
  const Eigen::VectorXd u2 = Eigen::VectorXd::Random(model.get_nu());
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x2);
  model.calc(data, x2, u2);
  model.calcDiff(data, x2, u2);
  cost = 0;
  Lx.setZero();
  Lu.setZero();
  Lxx.setZero();
  Lxu.setZero();
  Luu.setZero();
  for (std::size_t i = 0; i < 3;
       ++i) {  // we need to update data because this costs are active
    models[i]->calc(datas[i], x2, u2);
    models[i]->calcDiff(datas[i], x2, u2);
    cost += datas[i]->cost;
    Lx += datas[i]->Lx;
    Lu += datas[i]->Lu;
    Lxx += datas[i]->Lxx;
    Lxu += datas[i]->Lxu;
    Luu += datas[i]->Luu;
  }
  BOOST_CHECK(data->cost == cost);
  BOOST_CHECK(data->Lx == Lx);
  BOOST_CHECK(data->Lu == Lu);
  BOOST_CHECK(data->Lxx == Lxx);
  BOOST_CHECK(data->Lxu == Lxu);
  BOOST_CHECK(data->Luu == Luu);

  x2 = state->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x2);
  model.calc(data, x2);
  model.calcDiff(data, x2);
  cost = 0.;
  Lx.setZero();
  Lxx.setZero();
  for (std::size_t i = 0; i < 3; ++i) {
    models[i]->calc(datas[i], x2);
    models[i]->calcDiff(datas[i], x2);
    cost += datas[i]->cost;
    Lx += datas[i]->Lx;
    Lxx += datas[i]->Lxx;
  }
  BOOST_CHECK(data->cost == cost);
  BOOST_CHECK(data->Lx == Lx);
  BOOST_CHECK(data->Lxx == Lxx);

  // Checking that casted computation is the same
#ifdef NDEBUG  // Run only in release mode
  model.changeCostStatus("random_cost_3", true);
  model.changeCostStatus("random_cost_4", true);
  crocoddyl::CostModelSumTpl<float> casted_model = model.cast<float>();
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<float>>& casted_state =
      std::static_pointer_cast<crocoddyl::StateMultibodyTpl<float>>(
          casted_model.get_state());
  pinocchio::ModelTpl<float>& casted_pinocchio_model =
      *casted_state->get_pinocchio().get();
  pinocchio::DataTpl<float> casted_pinocchio_data(casted_pinocchio_model);
  crocoddyl::DataCollectorMultibodyTpl<float> casted_shared_data(
      &casted_pinocchio_data);
  std::vector<std::shared_ptr<crocoddyl::CostModelAbstractTpl<float>>>
      casted_models;
  std::vector<std::shared_ptr<crocoddyl::CostDataAbstractTpl<float>>>
      casted_datas;
  for (std::size_t i = 0; i < 5; ++i) {
    casted_models.push_back(models[i]->cast<float>());
    casted_datas.push_back(casted_models[i]->createData(&casted_shared_data));
  }
  const std::shared_ptr<crocoddyl::CostDataSumTpl<float>>& casted_data =
      casted_model.createData(&casted_shared_data);
  const Eigen::VectorXf x1_f = x1.cast<float>();
  const Eigen::VectorXf u1_f = u1.cast<float>();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data,
                                          x1);
  crocoddyl::unittest::updateAllPinocchio(&casted_pinocchio_model,
                                          &casted_pinocchio_data, x1_f);
  model.calc(data, x1, u1);
  model.calcDiff(data, x1, u1);
  casted_model.calc(casted_data, x1_f, u1_f);
  casted_model.calcDiff(casted_data, x1_f, u1_f);
  Lx.setZero();
  Lu.setZero();
  Lxx.setZero();
  Lxu.setZero();
  Luu.setZero();
  float cost_f = 0.f;
  Eigen::VectorXf Lx_f = Lx.cast<float>();
  Eigen::VectorXf Lu_f = Lu.cast<float>();
  Eigen::MatrixXf Lxx_f = Lxx.cast<float>();
  Eigen::MatrixXf Lxu_f = Lxu.cast<float>();
  Eigen::MatrixXf Luu_f = Luu.cast<float>();
  for (std::size_t i = 0; i < 5;
       ++i) {  // we need to update data because this costs are active
    casted_models[i]->calc(casted_datas[i], x1_f, u1_f);
    casted_models[i]->calcDiff(casted_datas[i], x1_f, u1_f);
    cost_f += casted_datas[i]->cost;
    Lx_f += casted_datas[i]->Lx;
    Lu_f += casted_datas[i]->Lu;
    Lxx_f += casted_datas[i]->Lxx;
    Lxu_f += casted_datas[i]->Lxu;
    Luu_f += casted_datas[i]->Luu;
  }
  BOOST_CHECK(casted_data->cost == cost_f);
  BOOST_CHECK(casted_data->Lx == Lx_f);
  BOOST_CHECK(casted_data->Lu == Lu_f);
  BOOST_CHECK(casted_data->Lxx == Lxx_f);
  BOOST_CHECK(casted_data->Lxu == Lxu_f);
  BOOST_CHECK(casted_data->Luu == Luu_f);
#endif
}

void test_get_costs(StateModelTypes::Type state_type) {
  // setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(state_factory.create(state_type));
  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model.get_state());
  pinocchio::Data pinocchio_data(*state->get_pinocchio().get());

  // create and add some contact objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_cost_" << i;
    model.addCost(os.str(), create_random_cost(state_type), 1.);
  }

  // get the contacts
  const crocoddyl::CostModelSum::CostModelContainer& costs = model.get_costs();

  // test
  crocoddyl::CostModelSum::CostModelContainer::const_iterator it_m, end_m;
  unsigned i;
  for (i = 0, it_m = costs.begin(), end_m = costs.end(); it_m != end_m;
       ++it_m, ++i) {
    std::ostringstream os;
    os << "random_cost_" << i;
    BOOST_CHECK(it_m->first == os.str());
  }
}

void test_get_nr(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(state_factory.create(state_type));

  // create the corresponding data object
  const std::shared_ptr<crocoddyl::StateMultibody>& state =
      std::static_pointer_cast<crocoddyl::StateMultibody>(model.get_state());
  pinocchio::Data pinocchio_data(*state->get_pinocchio().get());

  // create and add some contact objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_cost_" << i;
    model.addCost(os.str(), create_random_cost(state_type), 1.);
  }

  // compute ni
  std::size_t nr = 0;
  crocoddyl::CostModelSum::CostModelContainer::const_iterator it_m, end_m;
  for (it_m = model.get_costs().begin(), end_m = model.get_costs().end();
       it_m != end_m; ++it_m) {
    nr += it_m->second->cost->get_activation()->get_nr();
  }

  BOOST_CHECK(nr == model.get_nr());
}

void test_shareMemory(StateModelTypes::Type state_type) {
  // setup the test
  StateModelFactory state_factory;
  const std::shared_ptr<crocoddyl::StateAbstract> state =
      state_factory.create(state_type);
  crocoddyl::CostModelSum cost_model(state);
  crocoddyl::DataCollectorAbstract shared_data;
  const std::shared_ptr<crocoddyl::CostDataSum>& cost_data =
      cost_model.createData(&shared_data);

  const std::size_t ndx = state->get_ndx();
  const std::size_t nu = cost_model.get_nu();
  crocoddyl::ActionModelLQR action_model(ndx, nu);
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& action_data =
      action_model.createData();

  cost_data->shareMemory(action_data.get());
  cost_data->Lx = Eigen::VectorXd::Random(ndx);
  cost_data->Lu = Eigen::VectorXd::Random(nu);
  cost_data->Lxx = Eigen::MatrixXd::Random(ndx, ndx);
  cost_data->Luu = Eigen::MatrixXd::Random(nu, nu);
  cost_data->Lxu = Eigen::MatrixXd::Random(ndx, nu);

  // check that the data has been shared
  BOOST_CHECK(action_data->Lx.isApprox(cost_data->Lx, 1e-9));
  BOOST_CHECK(action_data->Lu.isApprox(cost_data->Lu, 1e-9));
  BOOST_CHECK(action_data->Lxx.isApprox(cost_data->Lxx, 1e-9));
  BOOST_CHECK(action_data->Luu.isApprox(cost_data->Luu, 1e-9));
  BOOST_CHECK(action_data->Lxu.isApprox(cost_data->Lxu, 1e-9));
}

//----------------------------------------------------------------------------//

void register_unit_tests(StateModelTypes::Type state_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_CostModelSum"
            << "_" << state_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_constructor, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_addCost, state_type)));
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_addCost_error_message, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_removeCost, state_type)));
  ts->add(
      BOOST_TEST_CASE(boost::bind(&test_removeCost_error_message, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calcDiff, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_get_costs, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_get_nr, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_shareMemory, state_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_parameter_aggregation_and_terminal_lifecycle));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_parameter_dimension_validation));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_parameter_share_memory));
  framework::master_test_suite().add(
      BOOST_TEST_CASE(&test_parameter_setters_and_no_allocation));
  register_unit_tests(StateModelTypes::StateMultibody_TalosArm);
  register_unit_tests(StateModelTypes::StateMultibody_HyQ);
  register_unit_tests(StateModelTypes::StateMultibody_Talos);
  return true;
}

int main(int argc, char** argv) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
