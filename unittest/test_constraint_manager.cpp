///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2022, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"

#include "factory/constraint.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

//----------------------------------------------------------------------------//

void test_constructor(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));

  // Run the print function
  std::ostringstream tmp;
  tmp << model;

  // Test the initial size of the map
  BOOST_CHECK(model.get_constraints().size() == 0);
}

void test_addConstraint(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));

  // add an active constraint
  boost::shared_ptr<crocoddyl::ConstraintModelAbstract> rand_constraint_1 = create_random_constraint(state_type);
  model.addConstraint("random_constraint_1", rand_constraint_1);
  std::size_t ng = rand_constraint_1->get_ng();
  std::size_t nh = rand_constraint_1->get_nh();
  BOOST_CHECK(model.get_ng() == ng);
  BOOST_CHECK(model.get_nh() == nh);

  // add an inactive constraint
  boost::shared_ptr<crocoddyl::ConstraintModelAbstract> rand_constraint_2 = create_random_constraint(state_type);
  model.addConstraint("random_constraint_2", rand_constraint_2, false);
  BOOST_CHECK(model.get_ng() == ng);
  BOOST_CHECK(model.get_nh() == nh);

  // change the random constraint 2 status
  model.changeConstraintStatus("random_constraint_2", true);
  ng += rand_constraint_2->get_ng();
  nh += rand_constraint_2->get_nh();
  BOOST_CHECK(model.get_ng() == ng);
  BOOST_CHECK(model.get_nh() == nh);

  // change the random constraint 1 status
  model.changeConstraintStatus("random_constraint_1", false);
  ng -= rand_constraint_1->get_ng();
  nh -= rand_constraint_1->get_nh();
  BOOST_CHECK(model.get_ng() == ng);
  BOOST_CHECK(model.get_nh() == nh);
}

void test_addConstraint_error_message(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));

  // create an constraint object
  boost::shared_ptr<crocoddyl::ConstraintModelAbstract> rand_constraint = create_random_constraint(state_type);

  // add twice the same constraint object to the container
  model.addConstraint("random_constraint", rand_constraint);

  // test error message when we add a duplicate constraint
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.addConstraint("random_constraint", rand_constraint);
  capture_ios.endCapture();
  std::stringstream expected_buffer;
  expected_buffer << "Warning: we couldn't add the random_constraint constraint item, it already existed."
                  << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());

  // test error message when we change the constraint status of an inexistent constraint
  capture_ios.beginCapture();
  model.changeConstraintStatus("no_exist_constraint", true);
  capture_ios.endCapture();
  expected_buffer.clear();
  expected_buffer
      << "Warning: we couldn't change the status of the no_exist_constraint constraint item, it doesn't exist."
      << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_removeConstraint(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));

  // add an active constraint
  boost::shared_ptr<crocoddyl::ConstraintModelAbstract> rand_constraint = create_random_constraint(state_type);
  model.addConstraint("random_constraint", rand_constraint);
  std::size_t ng = rand_constraint->get_ng();
  std::size_t nh = rand_constraint->get_nh();
  BOOST_CHECK(model.get_ng() == ng);
  BOOST_CHECK(model.get_nh() == nh);

  // remove the constraint
  model.removeConstraint("random_constraint");
  BOOST_CHECK(model.get_ng() == 0);
  BOOST_CHECK(model.get_nh() == 0);
}

void test_removeConstraint_error_message(StateModelTypes::Type state_type) {
  // Setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));

  // remove a none existing constraint form the container, we expect a cout message here
  CaptureIOStream capture_ios;
  capture_ios.beginCapture();
  model.removeConstraint("random_constraint");
  capture_ios.endCapture();

  // Test that the error message is sent.
  std::stringstream expected_buffer;
  expected_buffer << "Warning: we couldn't remove the random_constraint constraint item, it doesn't exist."
                  << std::endl;
  BOOST_CHECK(capture_ios.str() == expected_buffer.str());
}

void test_calc(StateModelTypes::Type state_type) {
  // setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));
  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model.get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);

  // create and add some constraint objects
  std::vector<boost::shared_ptr<crocoddyl::ConstraintModelAbstract> > models;
  std::vector<boost::shared_ptr<crocoddyl::ConstraintDataAbstract> > datas;
  for (std::size_t i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_constraint_" << i;
    const boost::shared_ptr<crocoddyl::ConstraintModelAbstract>& m = create_random_constraint(state_type);
    model.addConstraint(os.str(), m, 1.);
    models.push_back(m);
    datas.push_back(m->createData(&shared_data));
  }

  // create the data of the constraint sum
  const boost::shared_ptr<crocoddyl::ConstraintDataManager>& data = model.createData(&shared_data);

  // compute the constraint sum data for the case when all constraints are defined as active
  const Eigen::VectorXd& x1 = state->rand();
  const Eigen::VectorXd& u1 = Eigen::VectorXd::Random(model.get_nu());
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x1);
  model.calc(data, x1, u1);

  // check the constraint against single constraint computations
  std::size_t ng_i = 0;
  std::size_t nh_i = 0;
  Eigen::VectorXd g = Eigen::VectorXd::Zero(model.get_ng());
  Eigen::VectorXd h = Eigen::VectorXd::Zero(model.get_nh());
  for (std::size_t i = 0; i < 5; ++i) {
    models[i]->calc(datas[i], x1, u1);
    const std::size_t ng = models[i]->get_ng();
    const std::size_t nh = models[i]->get_nh();
    g.segment(ng_i, ng) = datas[i]->g;
    h.segment(nh_i, nh) = datas[i]->h;
    ng_i += ng;
    nh_i += nh;
  }
  BOOST_CHECK(data->g.isApprox(g, 1e-9));
  BOOST_CHECK(data->h.isApprox(h, 1e-9));
}

void test_calcDiff(StateModelTypes::Type state_type) {
  // setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));
  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model.get_state());
  pinocchio::Model& pinocchio_model = *state->get_pinocchio().get();
  pinocchio::Data pinocchio_data(pinocchio_model);
  crocoddyl::DataCollectorMultibody shared_data(&pinocchio_data);

  // create and add some constraint objects
  std::vector<boost::shared_ptr<crocoddyl::ConstraintModelAbstract> > models;
  std::vector<boost::shared_ptr<crocoddyl::ConstraintDataAbstract> > datas;
  for (std::size_t i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_constraint_" << i;
    const boost::shared_ptr<crocoddyl::ConstraintModelAbstract>& m = create_random_constraint(state_type);
    model.addConstraint(os.str(), m, 1.);
    models.push_back(m);
    datas.push_back(m->createData(&shared_data));
  }

  // create the data of the constraint sum
  const boost::shared_ptr<crocoddyl::ConstraintDataManager>& data = model.createData(&shared_data);

  // compute the constraint sum data for the case when all constraints are defined as active
  Eigen::VectorXd x1 = state->rand();
  const Eigen::VectorXd& u1 = Eigen::VectorXd::Random(model.get_nu());
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x1);
  model.calc(data, x1, u1);
  model.calcDiff(data, x1, u1);

  // check the constraint against single constraint computations
  std::size_t ng_i = 0;
  std::size_t nh_i = 0;
  const std::size_t ndx = state->get_ndx();
  const std::size_t nu = model.get_nu();
  Eigen::VectorXd g = Eigen::VectorXd::Zero(model.get_ng());
  Eigen::VectorXd h = Eigen::VectorXd::Zero(model.get_nh());
  Eigen::MatrixXd Gx = Eigen::MatrixXd::Zero(model.get_ng(), ndx);
  Eigen::MatrixXd Gu = Eigen::MatrixXd::Zero(model.get_ng(), nu);
  Eigen::MatrixXd Hx = Eigen::MatrixXd::Zero(model.get_nh(), ndx);
  Eigen::MatrixXd Hu = Eigen::MatrixXd::Zero(model.get_nh(), nu);
  for (std::size_t i = 0; i < 5; ++i) {
    models[i]->calc(datas[i], x1, u1);
    models[i]->calcDiff(datas[i], x1, u1);
    const std::size_t ng = models[i]->get_ng();
    const std::size_t nh = models[i]->get_nh();
    g.segment(ng_i, ng) = datas[i]->g;
    h.segment(nh_i, nh) = datas[i]->h;
    Gx.block(ng_i, 0, ng, ndx) = datas[i]->Gx;
    Gu.block(ng_i, 0, ng, nu) = datas[i]->Gu;
    Hx.block(nh_i, 0, nh, ndx) = datas[i]->Hx;
    Hu.block(nh_i, 0, nh, nu) = datas[i]->Hu;
    ng_i += ng;
    nh_i += nh;
  }
  BOOST_CHECK(data->g.isApprox(g, 1e-9));
  BOOST_CHECK(data->h.isApprox(h, 1e-9));
  BOOST_CHECK(data->Gx.isApprox(Gx, 1e-9));
  BOOST_CHECK(data->Gu.isApprox(Gu, 1e-9));
  BOOST_CHECK(data->Hx.isApprox(Hx, 1e-9));
  BOOST_CHECK(data->Hu.isApprox(Hu, 1e-9));

  x1 = state->rand();
  crocoddyl::unittest::updateAllPinocchio(&pinocchio_model, &pinocchio_data, x1);
  model.calc(data, x1);
  model.calcDiff(data, x1);

  ng_i = 0;
  nh_i = 0;
  g.setZero();
  h.setZero();
  Gx.setZero();
  Gu.setZero();
  Hx.setZero();
  Hu.setZero();
  for (std::size_t i = 0; i < 5; ++i) {
    models[i]->calc(datas[i], x1);
    models[i]->calcDiff(datas[i], x1);
    const std::size_t ng = models[i]->get_ng();
    const std::size_t nh = models[i]->get_nh();
    g.segment(ng_i, ng) = datas[i]->g;
    h.segment(nh_i, nh) = datas[i]->h;
    Gx.block(ng_i, 0, ng, ndx) = datas[i]->Gx;
    Gu.block(ng_i, 0, ng, nu) = datas[i]->Gu;
    Hx.block(nh_i, 0, nh, ndx) = datas[i]->Hx;
    Hu.block(nh_i, 0, nh, nu) = datas[i]->Hu;
    ng_i += ng;
    nh_i += nh;
  }
  BOOST_CHECK(data->g.isApprox(g, 1e-9));
  BOOST_CHECK(data->h.isApprox(h, 1e-9));
  BOOST_CHECK(data->Gx.isApprox(Gx, 1e-9));
  BOOST_CHECK(data->Hx.isApprox(Hx, 1e-9));
}

void test_get_constraints(StateModelTypes::Type state_type) {
  // setup the test
  StateModelFactory state_factory;
  crocoddyl::ConstraintModelManager model(state_factory.create(state_type));
  // create the corresponding data object
  const boost::shared_ptr<crocoddyl::StateMultibody>& state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(model.get_state());
  pinocchio::Data pinocchio_data(*state->get_pinocchio().get());

  // create and add some contact objects
  for (unsigned i = 0; i < 5; ++i) {
    std::ostringstream os;
    os << "random_constraint_" << i;
    model.addConstraint(os.str(), create_random_constraint(state_type), 1.);
  }

  // get the contacts
  const crocoddyl::ConstraintModelManager::ConstraintModelContainer& constraints = model.get_constraints();

  // test
  crocoddyl::ConstraintModelManager::ConstraintModelContainer::const_iterator it_m, end_m;
  unsigned i;
  for (i = 0, it_m = constraints.begin(), end_m = constraints.end(); it_m != end_m; ++it_m, ++i) {
    std::ostringstream os;
    os << "random_constraint_" << i;
    BOOST_CHECK(it_m->first == os.str());
  }
}

void test_shareMemory(StateModelTypes::Type state_type) {
  // setup the test
  StateModelFactory state_factory;
  const boost::shared_ptr<crocoddyl::StateAbstract> state = state_factory.create(state_type);
  crocoddyl::ConstraintModelManager constraint_model(state);
  crocoddyl::DataCollectorAbstract shared_data;
  const boost::shared_ptr<crocoddyl::ConstraintDataManager>& constraint_data =
      constraint_model.createData(&shared_data);

  std::size_t ng = state->get_ndx();
  std::size_t nh = state->get_ndx();
  const std::size_t ndx = state->get_ndx();
  const std::size_t nu = constraint_model.get_nu();
  crocoddyl::ActionModelLQR action_model(ndx, nu);
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& action_data = action_model.createData();

  constraint_data->shareMemory(action_data.get());
  constraint_data->h = Eigen::VectorXd::Random(nh);
  constraint_data->g = Eigen::VectorXd::Random(ng);
  constraint_data->Gx = Eigen::MatrixXd::Random(ng, ndx);
  constraint_data->Gu = Eigen::MatrixXd::Random(ng, nu);
  constraint_data->Hx = Eigen::MatrixXd::Random(nh, ndx);
  constraint_data->Hu = Eigen::MatrixXd::Random(nh, nu);

  // check that the data has been shared
  BOOST_CHECK(action_data->g.isApprox(constraint_data->g, 1e-9));
  BOOST_CHECK(action_data->h.isApprox(constraint_data->h, 1e-9));
  BOOST_CHECK(action_data->Gx.isApprox(constraint_data->Gx, 1e-9));
  BOOST_CHECK(action_data->Gu.isApprox(constraint_data->Gu, 1e-9));
  BOOST_CHECK(action_data->Hx.isApprox(constraint_data->Hx, 1e-9));
  BOOST_CHECK(action_data->Hu.isApprox(constraint_data->Hu, 1e-9));

  // let's now resize the data
  ng /= 2;
  nh /= 2;
  constraint_data->resize(&action_model, action_data.get());

  // check that the shared data has been resized
  BOOST_CHECK(action_data->g.isApprox(constraint_data->g, 1e-9));
  BOOST_CHECK(action_data->h.isApprox(constraint_data->h, 1e-9));
  BOOST_CHECK(action_data->Gx.isApprox(constraint_data->Gx, 1e-9));
  BOOST_CHECK(action_data->Gu.isApprox(constraint_data->Gu, 1e-9));
  BOOST_CHECK(action_data->Hx.isApprox(constraint_data->Hx, 1e-9));
  BOOST_CHECK(action_data->Hu.isApprox(constraint_data->Hu, 1e-9));
}

//----------------------------------------------------------------------------//

void register_unit_tests(StateModelTypes::Type state_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_ConstraintModelManager"
            << "_" << state_type;
  std::cout << "Running " << test_name.str() << std::endl;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  ts->add(BOOST_TEST_CASE(boost::bind(&test_constructor, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_addConstraint, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_addConstraint_error_message, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_removeConstraint, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_removeConstraint_error_message, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calc, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_calcDiff, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_get_constraints, state_type)));
  ts->add(BOOST_TEST_CASE(boost::bind(&test_shareMemory, state_type)));
  framework::master_test_suite().add(ts);
}

bool init_function() {
  register_unit_tests(StateModelTypes::StateMultibody_TalosArm);
  register_unit_tests(StateModelTypes::StateMultibody_HyQ);
  register_unit_tests(StateModelTypes::StateMultibody_Talos);
  return true;
}

int main(int argc, char** argv) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
