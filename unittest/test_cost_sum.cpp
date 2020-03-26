///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/data/multibody.hpp"

#include "factory/cost.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_constructor() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // Test the initial size of the map
  BOOST_CHECK(model.get_costs().size() == 0);
}

void test_addCost() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // add an active cost
  boost::shared_ptr<crocoddyl::CostModelAbstract> rand_cost_1 = create_random_cost();
  model.addCost("random_cost_1", rand_cost_1, 1.);
  BOOST_CHECK(model.get_nr() == rand_cost_1->get_activation()->get_nr());

  // add an inactive cost
  boost::shared_ptr<crocoddyl::CostModelAbstract> rand_cost_2 = create_random_cost();
  model.addCost("random_cost_2", rand_cost_2, 1., false);
  BOOST_CHECK(model.get_nr() == rand_cost_1->get_activation()->get_nr());

  // change the random cost 2 status
  model.changeCostStatus("random_cost_2", true);
  BOOST_CHECK(model.get_nr() == rand_cost_1->get_activation()->get_nr() + rand_cost_2->get_activation()->get_nr());

  // change the random cost 1 status
  model.changeCostStatus("random_cost_1", false);
  BOOST_CHECK(model.get_nr() == rand_cost_2->get_activation()->get_nr());
}

void test_addCost_error_message() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // create an cost object
  boost::shared_ptr<crocoddyl::CostModelAbstract> rand_cost = create_random_cost();

  // add twice the same cost object to the container
  model.addCost("random_cost", rand_cost, 1.);

  // test error message when we add a duplicate cost
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.addCost("random_cost", rand_cost, 1.);
  capture_ios.endCapture();
  std::stringstream expected_buffer;
  expected_buffer << "Warning: this cost item already existed, we cannot add it" << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());

  // test error message when we change the cost status of an inexistent cost
  capture_ios.beginCapture();
  model.changeCostStatus("no_exist_cost", true);
  capture_ios.endCapture();
  expected_buffer.clear();
  expected_buffer << "Warning: this cost item doesn't exist, we cannot change its status" << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_removeCost() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // add an active cost
  boost::shared_ptr<crocoddyl::CostModelAbstract> rand_cost = create_random_cost();
  model.addCost("random_cost", rand_cost, 1.);
  BOOST_CHECK(model.get_nr() == rand_cost->get_activation()->get_nr());

  // remove the cost
  model.removeCost("random_cost");
  BOOST_CHECK(model.get_nr() == 0);
}

void test_removeCost_error_message() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // remove a none existing cost form the container, we expect a cout message here
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.removeCost("random_cost");
  capture_ios.endCapture();

  // Test that the error message is sent.
  std::stringstream expected_buffer;
  expected_buffer << "Warning: this cost item doesn't exist, we cannot remove it" << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_calc() {
  // setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));
  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state = model.get_state();
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);

  // create and add some cost objects
  std::vector<boost::shared_ptr<crocoddyl::CostModelAbstract> > models;
  std::vector<boost::shared_ptr<crocoddyl::CostDataAbstract> > datas;
  for (std::size_t i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_cost_" << i;
    const boost::shared_ptr<crocoddyl::CostModelAbstract>& m = create_random_cost();
    model.addCost(os.str(), m, 1.);
    models.push_back(m);
    datas.push_back(m->createData(&shared_data));
  }

  // create the data of the cost sum
  const boost::shared_ptr<crocoddyl::CostDataSum>& data = model.createData(&shared_data);

  // compute the cost sum data for the case when all costs are defined as active
  const Eigen::VectorXd& x1 = state->rand();
  const Eigen::VectorXd& u1 = Eigen::VectorXd::Random(model.get_nu());
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x1);
  model.calc(data, x1, u1);

  // check that the cost has been filled
  BOOST_CHECK(data->cost != 0.);

  // check the cost against single cost computations
  double cost = 0;
  for (std::size_t i = 0; i < 5; ++i) {
    models[i]->calc(datas[i], x1, u1);
    cost += datas[i]->cost;
  }
  BOOST_CHECK(data->cost == cost);

  // compute the cost sum data for the case when the first three costs are defined as active
  model.changeCostStatus("random_cost_3", false);
  model.changeCostStatus("random_cost_4", false);
  const Eigen::VectorXd& x2 = state->rand();
  const Eigen::VectorXd& u2 = Eigen::VectorXd::Random(model.get_nu());
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x1);
  model.calc(data, x2, u2);
  cost = 0;
  for (std::size_t i = 0; i < 3; ++i) {  // we need to update data because this costs are active
    models[i]->calc(datas[i], x2, u2);
    cost += datas[i]->cost;
  }
  BOOST_CHECK(data->cost == cost);
}

void test_calcDiff() {
  // setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));
  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state = model.get_state();
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);

  // create and add some cost objects
  std::vector<boost::shared_ptr<crocoddyl::CostModelAbstract> > models;
  std::vector<boost::shared_ptr<crocoddyl::CostDataAbstract> > datas;
  for (std::size_t i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_cost_" << i;
    const boost::shared_ptr<crocoddyl::CostModelAbstract>& m = create_random_cost();
    model.addCost(os.str(), m, 1.);
    models.push_back(m);
    datas.push_back(m->createData(&shared_data));
  }

  // create the data of the cost sum
  const boost::shared_ptr<crocoddyl::CostDataSum>& data = model.createData(&shared_data);

  // compute the cost sum data for the case when all costs are defined as active
  const Eigen::VectorXd& x1 = state->rand();
  const Eigen::VectorXd& u1 = Eigen::VectorXd::Random(model.get_nu());
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x1);
  model.calc(data, x1, u1);
  model.calcDiff(data, x1, u1);

  // check that the cost has been filled
  BOOST_CHECK(data->cost != 0.);

  // check the cost against single cost computations
  double cost = 0;
  Eigen::VectorXd Lx;
  Eigen::VectorXd Lu;
  Eigen::MatrixXd Lxx;
  Eigen::MatrixXd Lxu;
  Eigen::MatrixXd Luu;
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

  // compute the cost sum data for the case when the first three costs are defined as active
  model.changeCostStatus("random_cost_3", false);
  model.changeCostStatus("random_cost_4", false);
  const Eigen::VectorXd& x2 = state->rand();
  const Eigen::VectorXd& u2 = Eigen::VectorXd::Random(model.get_nu());
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x1);
  model.calc(data, x2, u2);
  model.calcDiff(data, x2, u2);
  cost = 0;
  Lx.setZero();
  Lu.setZero();
  Lxx.setZero();
  Lxu.setZero();
  Luu.setZero();
  for (std::size_t i = 0; i < 3; ++i) {  // we need to update data because this costs are active
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
}

void test_get_costs() {
  // setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));
  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some contact objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_cost_" << i;
    model.addCost(os.str(), create_random_cost(), 1.);
  }

  // get the contacts
  const crocoddyl::CostModelSum::CostModelContainer& costs = model.get_costs();

  // test
  crocoddyl::CostModelSum::CostModelContainer::const_iterator it_m, end_m;
  unsigned i;
  for (i = 0, it_m = costs.begin(), end_m = costs.end(); it_m != end_m; ++it_m, ++i) {
    std::ostringstream os;
    os << "random_cost_" << i;
    BOOST_CHECK(it_m->first == os.str());
  }
}

void test_get_nr() {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::CostModelSum model(boost::static_pointer_cast<crocoddyl::StateMultibody>(
      state_factory.create(StateModelTypes::StateMultibody_RandomHumanoid)));

  // create the corresponding data object
  pinocchio::Data pinocchio_data(*model.get_state()->get_pinocchio().get());

  // create and add some contact objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_cost_" << i;
    model.addCost(os.str(), create_random_cost(), 1.);
  }

  // compute ni
  std::size_t nr = 0;
  crocoddyl::CostModelSum::CostModelContainer::const_iterator it_m, end_m;
  for (it_m = model.get_costs().begin(), end_m = model.get_costs().end(); it_m != end_m; ++it_m) {
    nr += it_m->second->cost->get_activation()->get_nr();
  }

  BOOST_CHECK(nr == model.get_nr());
}

//----------------------------------------------------------------------------//

void register_unit_tests() {
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_constructor)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_addCost)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_addCost_error_message)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_removeCost)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_removeCost_error_message)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calc)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_calcDiff)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_get_costs)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_get_nr)));
}

bool init_function() {
  register_unit_tests();
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }