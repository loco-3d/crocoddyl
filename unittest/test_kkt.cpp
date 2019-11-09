///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/included/unit_test.hpp>
#include <boost/bind.hpp>
#include <Eigen/Dense>

#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/solver-base.hpp"
#include "crocoddyl/core/solvers/kkt.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/actions/unicycle.hpp"
#include "crocoddyl/core/numdiff/state.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"


using namespace boost::unit_test;

//____________________________________________________________________________//

void test_shooting_problem(crocoddyl::ShootingProblem& problem) {
  // test my understanding
  const long unsigned int T = problem.get_T();
  crocoddyl::ActionModelAbstract* model_zero = problem.get_runningModels()[0];
  const unsigned int nx_ac = model_zero->get_state().get_nx();
  const unsigned int ndx_ac = model_zero->get_state().get_ndx();
  const unsigned int nu_ac = model_zero->get_nu();

  for (long unsigned int t = 0; t < T; ++t) {
    crocoddyl::ActionModelAbstract* model_i = problem.get_runningModels()[t];
    BOOST_CHECK_EQUAL(model_i->get_state().get_nx(), nx_ac);
    BOOST_CHECK_EQUAL(model_i->get_state().get_ndx(), ndx_ac);
    BOOST_CHECK_EQUAL(model_i->get_nu(), nu_ac);
  }
  crocoddyl::ActionModelAbstract* model_terminal = problem.get_terminalModel();
  BOOST_CHECK_EQUAL(model_terminal->get_state().get_nx(), nx_ac);
  BOOST_CHECK_EQUAL(model_terminal->get_state().get_ndx(), ndx_ac);
}
//____________________________________________________________________________//

void test_kkt_matrix_lqr() {
  unsigned int NX = 3;
  unsigned int NU = 2;
  unsigned int N = 2;


  Eigen::VectorXd x0;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  std::vector<crocoddyl::ActionModelAbstract *> runningModels;
  crocoddyl::ActionModelAbstract *terminalModel;
  x0 = Eigen::VectorXd::Zero(NX);

  // Creating the action models and warm point for the LQR system
  for (unsigned int i = 0; i < N; ++i)
  {
    crocoddyl::ActionModelAbstract *model_i = new crocoddyl::ActionModelLQR(NX, NU);
    runningModels.push_back(model_i);
    xs.push_back(x0);
    us.push_back(Eigen::VectorXd::Zero(NU));
  }
  xs.push_back(x0);
  terminalModel = new crocoddyl::ActionModelLQR(NX, NU);

  crocoddyl::ShootingProblem problem(x0, runningModels, terminalModel);
  crocoddyl::SolverKKT kkt(problem);
  std::vector<crocoddyl::CallbackAbstract *> cbs;
  cbs.push_back(new crocoddyl::CallbackVerbose());
  kkt.setCallbacks(cbs);

  kkt.setCandidate(xs,us,false);
  kkt.computeDirection(true); 
  // check if kkt matrix is invertible 
  std::cout<<"kkt invertible "<<kkt.get_kkt().fullPivLu().isInvertible()<<std::endl;

  // compute condition number for kkt matrix 
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(kkt.get_kkt());
  double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
  std::cout<<"kkt condition number "<<cond<<std::endl;

  // solve the kkt system by hand 
  std::cout<<"kkt matrix\n"<<kkt.get_kkt()<<std::endl;
  Eigen::MatrixXd invKKT = kkt.get_kkt().inverse(); 
  std::cout<<"inverse kkt matrix\n"<<invKKT<<std::endl;
  std::cout<<"solution \n"<<invKKT*(-kkt.get_kktref())<<std::endl;

  // print derivatives to make sure they are filled correctly in the kkt matrix 
  for (unsigned int i = 0; i < N; ++i) {
    boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = problem.running_datas_[i];

    std::cout<<"Lxx at "<<i<<"\n"<<d->get_Lxx()<<std::endl;
    std::cout<<"Luu at "<<i<<"\n"<<d->get_Luu()<<std::endl;
    std::cout<<"Lxu at "<<i<<"\n"<<d->get_Lxu()<<std::endl;
    std::cout<<"Fx at "<<i<<"\n"<<d->get_Fx()<<std::endl;
    std::cout<<"Fu at "<<i<<"\n"<<d->get_Fu()<<std::endl;

    
  }
  boost::shared_ptr<crocoddyl::ActionDataAbstract>& df = problem.terminal_data_;
  std::cout<<"Lxx at "<<N<<"\n"<<df->get_Lxx()<<std::endl;
  std::cout<<"Fx at "<<N<<"\n"<<df->get_Fx()<<std::endl;


  std::cout<<"kktref vector\n"<<kkt.get_kktref()<<std::endl;


  for (unsigned int i = 0; i < N; ++i) {
    boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = problem.running_datas_[i];
    std::cout<<"Lx at "<<i<<"\n"<<d->get_Lx()<<std::endl;
    std::cout<<"Lu at "<<i<<"\n"<<d->get_Lu()<<std::endl;
  }

  std::cout<<"Lx at "<<N<<"\n"<<df->get_Lx()<<std::endl;
  std::cout<<"PrimalDual vector\n"<<kkt.get_primaldual()<<std::endl;

}


//____________________________________________________________________________//

void test_kkt_matrix_unicycle() {
  unsigned int N = 2;  // number of nodes

  Eigen::VectorXd x0;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  std::vector<crocoddyl::ActionModelAbstract*> runningModels;
  crocoddyl::ActionModelAbstract* terminalModel;
  x0 = Eigen::Vector3d(1., 0., 0.);

  // Creating the action models and warm point for the unicycle system
  for (unsigned int i = 0; i < N; ++i) {
    crocoddyl::ActionModelAbstract* model_i = new crocoddyl::ActionModelUnicycle();
    runningModels.push_back(model_i);
    xs.push_back(x0);
    us.push_back(Eigen::Vector2d::Zero());
  }
  xs.push_back(x0);
  terminalModel = new crocoddyl::ActionModelUnicycle();

  // Formulating the optimal control problem
  crocoddyl::ShootingProblem problem(x0, runningModels, terminalModel);

  crocoddyl::SolverKKT kkt(problem);

  std::vector<crocoddyl::CallbackAbstract *> cbs;
  cbs.push_back(new crocoddyl::CallbackVerbose());
  kkt.setCallbacks(cbs);

  kkt.setCandidate(xs,us,false);
  kkt.computeDirection(true); 

  std::cout<<"kkt invertible "<<kkt.get_kkt().fullPivLu().isInvertible()<<std::endl;


  Eigen::JacobiSVD<Eigen::MatrixXd> svd(kkt.get_kkt());
  double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
  std::cout<<"kkt condition number "<<cond<<std::endl;
  std::cout<<"kkt matrix\n"<<kkt.get_kkt()<<std::endl;

  Eigen::MatrixXd invKKT = kkt.get_kkt().inverse(); 

  std::cout<<"inverse kkt matrix\n"<<invKKT<<std::endl;

  std::cout<<"solution \n"<<invKKT*(-kkt.get_kktref())<<std::endl;


  for (unsigned int i = 0; i < N; ++i) {
    boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = problem.running_datas_[i];

    std::cout<<"Lxx at "<<i<<"\n"<<d->get_Lxx()<<std::endl;
    std::cout<<"Luu at "<<i<<"\n"<<d->get_Luu()<<std::endl;
    std::cout<<"Lxu at "<<i<<"\n"<<d->get_Lxu()<<std::endl;
    std::cout<<"Fx at "<<i<<"\n"<<d->get_Fx()<<std::endl;
    std::cout<<"Fu at "<<i<<"\n"<<d->get_Fu()<<std::endl;

    
  }
  boost::shared_ptr<crocoddyl::ActionDataAbstract>& df = problem.terminal_data_;
  std::cout<<"Lxx at "<<N<<"\n"<<df->get_Lxx()<<std::endl;
  std::cout<<"Fx at "<<N<<"\n"<<df->get_Fx()<<std::endl;
  // std::cout<<"Luu at "<<N<<"\n"<<df->get_Luu()<<std::endl;


  std::cout<<"kktref vector\n"<<kkt.get_kktref()<<std::endl;


  for (unsigned int i = 0; i < N; ++i) {
    boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = problem.running_datas_[i];

    std::cout<<"Lx at "<<i<<"\n"<<d->get_Lx()<<std::endl;
    std::cout<<"Lu at "<<i<<"\n"<<d->get_Lu()<<std::endl;


    
  }

  std::cout<<"Lx at "<<N<<"\n"<<df->get_Lx()<<std::endl;

  std::cout<<"PrimalDual vector\n"<<kkt.get_primaldual()<<std::endl;

}




//____________________________________________________________________________//

void test_kkt_unicycle(crocoddyl::ShootingProblem& problem) {
  const long unsigned int T = problem.get_T();
  crocoddyl::ActionModelAbstract* model_zero = problem.get_runningModels()[0];
  const unsigned int nx_ac = model_zero->get_state().get_nx();
  const unsigned int ndx_ac = model_zero->get_state().get_ndx();
  const unsigned int nu_ac = model_zero->get_nu();

  const unsigned int nx_val = (unsigned int)(T + 1) * nx_ac;
  const unsigned int ndx_val = (unsigned int)(T + 1) * ndx_ac;
  const unsigned int nu_val = (unsigned int)T * nu_ac;

  crocoddyl::SolverKKT kkt(problem);

  Eigen::VectorXd x0;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  x0 = Eigen::Vector3d(1., 0., 0.);

  // Creating the action models and warm point for the unicycle system
  for (unsigned int i = 0; i < T; ++i)
  {
    xs.push_back(x0);
    us.push_back(Eigen::Vector2d::Zero());
  }
  xs.push_back(x0);

  std::vector<crocoddyl::CallbackAbstract *> cbs;
  cbs.push_back(new crocoddyl::CallbackVerbose());
  kkt.setCallbacks(cbs);

  kkt.solve(xs, us, 3);

  // check trajectory dimensions
  BOOST_CHECK_EQUAL(kkt.get_us().size(), T);
  BOOST_CHECK_EQUAL(kkt.get_xs().size(), T + 1);
  BOOST_CHECK((kkt.get_xs()[0]-x0).isMuchSmallerThan(1.,1.e-9));
  // check problem dimensions nx, ndx, nu
  // BOOST_CHECK_EQUAL(kkt.get_nx(), nx_val);
  // BOOST_CHECK_EQUAL(kkt.get_ndx(), ndx_val);
  // BOOST_CHECK_EQUAL(kkt.get_nu(), nu_val);
  // check kkt matrix dimensions
  // BOOST_CHECK_EQUAL(kkt.get_kkt().rows(), 2 * ndx_val + nu_val);
  // BOOST_CHECK_EQUAL(kkt.get_kkt().cols(), 2 * ndx_val + nu_val);
  // BOOST_CHECK_EQUAL(kkt.get_kktref().rows(), 2 * ndx_val + nu_val);
}

//____________________________________________________________________________//

void test_ddp_unicycle(crocoddyl::ShootingProblem& problem) {
  const long unsigned int T = problem.get_T();
  crocoddyl::SolverDDP ddp(problem);

  Eigen::VectorXd x0;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  x0 = Eigen::Vector3d(1., 0., 0.);

  // Creating the action models and warm point for the unicycle system
  for (unsigned int i = 0; i < T; ++i)
  {
    xs.push_back(x0);
    us.push_back(Eigen::Vector2d::Zero());
  }
  xs.push_back(x0);

  std::vector<crocoddyl::CallbackAbstract *> cbs;
  cbs.push_back(new crocoddyl::CallbackVerbose());
  ddp.setCallbacks(cbs);
  ddp.solve(xs, us, 100);

  // check trajectory dimensions
  BOOST_CHECK_EQUAL(ddp.get_us().size(), T);
  BOOST_CHECK_EQUAL(ddp.get_xs().size(), T + 1);
}
//____________________________________________________________________________//

void test_ddp_lqr(){
  unsigned int NX = 6;
  unsigned int NU = 3;
  unsigned int N = 10;
  unsigned int MAXITER = 10;

  Eigen::VectorXd x0;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  std::vector<crocoddyl::ActionModelAbstract *> runningModels;
  crocoddyl::ActionModelAbstract *terminalModel;
  x0 = Eigen::VectorXd::Zero(NX);

  // Creating the action models and warm point for the LQR system
  for (unsigned int i = 0; i < N; ++i)
  {
    crocoddyl::ActionModelAbstract *model_i = new crocoddyl::ActionModelLQR(NX, NU);
    runningModels.push_back(model_i);
    xs.push_back(x0);
    us.push_back(Eigen::VectorXd::Zero(NU));
  }
  xs.push_back(x0);
  terminalModel = new crocoddyl::ActionModelLQR(NX, NU);

  crocoddyl::ShootingProblem problem(x0, runningModels, terminalModel);

  crocoddyl::SolverDDP ddp(problem);
  std::vector<crocoddyl::CallbackAbstract *> cbs;
  cbs.push_back(new crocoddyl::CallbackVerbose());
  ddp.setCallbacks(cbs);

  ddp.solve(xs, us, MAXITER);

  BOOST_CHECK_EQUAL(ddp.get_us().size(), N);
  BOOST_CHECK_EQUAL(ddp.get_xs().size(), N + 1);
}



//____________________________________________________________________________//

void test_kkt_lqr(){

  unsigned int NX = 6;
  unsigned int NU = 3;
  unsigned int N = 10;


  Eigen::VectorXd x0;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  std::vector<crocoddyl::ActionModelAbstract *> runningModels;
  crocoddyl::ActionModelAbstract *terminalModel;
  x0 = Eigen::VectorXd::Zero(NX);

  // Creating the action models and warm point for the LQR system
  for (unsigned int i = 0; i < N; ++i)
  {
    crocoddyl::ActionModelAbstract *model_i = new crocoddyl::ActionModelLQR(NX, NU);
    runningModels.push_back(model_i);
    xs.push_back(x0);
    us.push_back(Eigen::VectorXd::Zero(NU));
  }
  xs.push_back(x0);
  terminalModel = new crocoddyl::ActionModelLQR(NX, NU);

  crocoddyl::ShootingProblem problem(x0, runningModels, terminalModel);
  crocoddyl::SolverKKT kkt(problem);
  std::vector<crocoddyl::CallbackAbstract *> cbs;
  cbs.push_back(new crocoddyl::CallbackVerbose());
  kkt.setCallbacks(cbs);
  kkt.solve(xs, us, 3);
  BOOST_CHECK_EQUAL(kkt.get_us().size(), N);
  BOOST_CHECK_EQUAL(kkt.get_xs().size(), N + 1);
  BOOST_CHECK((kkt.get_xs()[0]-x0).isMuchSmallerThan(1.,1.e-9));
}


//____________________________________________________________________________//

void register_state_vector_unit_tests() {
  unsigned int N = 50;  // number of nodes

  Eigen::VectorXd x0;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  std::vector<crocoddyl::ActionModelAbstract*> runningModels;
  crocoddyl::ActionModelAbstract* terminalModel;
  x0 = Eigen::Vector3d(1., 0., 0.);

  // Creating the action models and warm point for the unicycle system
  for (unsigned int i = 0; i < N; ++i) {
    crocoddyl::ActionModelAbstract* model_i = new crocoddyl::ActionModelUnicycle();
    runningModels.push_back(model_i);
    xs.push_back(x0);
    us.push_back(Eigen::Vector2d::Zero());
  }
  xs.push_back(x0);
  terminalModel = new crocoddyl::ActionModelUnicycle();

  // Formulating the optimal control problem
  crocoddyl::ShootingProblem problem(x0, runningModels, terminalModel);
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_kkt_matrix_lqr)));
  // framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_kkt_matrix_unicycle)));
  // framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_shooting_problem, problem)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_ddp_unicycle, problem)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_ddp_lqr)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_kkt_unicycle, problem)));
  framework::master_test_suite().add(BOOST_TEST_CASE(boost::bind(&test_kkt_lqr)));
}

//____________________________________________________________________________//

//____________________________________________________________________________//

bool init_function() {
  register_state_vector_unit_tests();
  return true;
}

//____________________________________________________________________________//

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
