///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#include <boost/test/included/unit_test.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"

using namespace boost::unit_test;

class ActionModelLQRIdentity: public crocoddyl::ActionModelLQR
{
public:
  ActionModelLQRIdentity(const unsigned int& nx, const unsigned int& nu, bool drift_free)
    : crocoddyl::ActionModelLQR(nx, nu, drift_free) {
    Fx_ = Eigen::MatrixXd::Identity(nx, nx);
    Fu_ = Eigen::MatrixXd::Identity(nx, nu);
    f0_ = Eigen::VectorXd::Ones(nx);
    Lxx_ = Eigen::MatrixXd::Identity(nx, nx);
    Lxu_ = Eigen::MatrixXd::Identity(nx, nu);
    Luu_ = Eigen::MatrixXd::Identity(nu, nu);
    lx_ = Eigen::VectorXd::Ones(nx);
    lu_ = Eigen::VectorXd::Ones(nu);
  }
};

/**
 * @brief This small struct allows to build a crocoddyl::ShootingProblem
 * from an ActionModelLQRIdentity
 * while managing the creation and deletion of the memory inside the unit tests.
 */
struct ShootingProblemFactory{
  /**
   * @brief Construct a new ShootingProblemFactory object
   * 
   * @param nx is the dimension of the state
   * @param nu is the dimension of teh control 
   * @param drift_free Is a ActionLQR parameter
   * @param horizon_size
   */
  ShootingProblemFactory(int nx, int nu, bool drift_free,
                                           unsigned horizon_size)
  {
    running_models.clear();
    for(unsigned i = 0 ; i < horizon_size ; ++i)
    {
      running_models.push_back(new ActionModelLQRIdentity(nx, nu, drift_free));
    }
    terminal_model = new ActionModelLQRIdentity (nx, nu, drift_free);
    x0 = terminal_model->get_state()->rand();
    shooting_problem = 
      new crocoddyl::ShootingProblem(x0, running_models, terminal_model);
  }

  /**
   * @brief Destroy the ShootingProblemFactory object
   */
  ~ShootingProblemFactory()
  {
    for(unsigned i = 0 ; i < running_models.size() ; ++i)
    {
      if(running_models[i] != NULL)
      {
        delete running_models[i];
        running_models[i] = NULL;
      }
    }
    running_models.clear();
    if(terminal_model != NULL)
    {
      delete terminal_model;
      terminal_model = NULL;
    }
    if(shooting_problem != NULL)
    {
      delete shooting_problem;
      shooting_problem = NULL;
    }
  }

  /**
   * @brief Get the shooting_problem object
   * 
   * @return crocoddyl::ShootingProblem& 
   */
  crocoddyl::ShootingProblem& get_shooting_problem()
  {
    return *shooting_problem;
  }

  std::vector<crocoddyl::ActionModelAbstract*> running_models; 
  ActionModelLQRIdentity* terminal_model;
  Eigen::VectorXd x0;
  crocoddyl::ShootingProblem* shooting_problem;
};

//____________________________________________________________________________//
//____________________________________________________________________________//

void test_trajectory_dimension(int nx, int nu, bool drift_free, unsigned horizon_size) {
  // Create the shooting problem
  ShootingProblemFactory factory(nx, nu, drift_free, horizon_size);
  crocoddyl::ShootingProblem& shooting_problem = factory.get_shooting_problem();

  // Compute the sum of all the state dimensions
  long int sum_state_dims = 0;
  for(unsigned i = 0 ; i < shooting_problem.get_runningModels().size() ; ++i)
  {
    Eigen::VectorXd x = shooting_problem.get_runningModels()[i]->get_state()->zero();
    sum_state_dims += x.size();
  }
  Eigen::VectorXd x = shooting_problem.get_terminalModel()->get_state()->zero();
  sum_state_dims += x.size();

  int sum_model_state_dims = 0;
  for(unsigned i = 0 ; i < shooting_problem.get_runningModels().size() ; ++i)
  {
    sum_model_state_dims += shooting_problem.get_runningModels()[i]->get_nx();
  }
  sum_model_state_dims += shooting_problem.get_terminalModel()->get_nx();

  BOOST_CHECK( sum_model_state_dims == sum_state_dims );
}

//____________________________________________________________________________//
//____________________________________________________________________________//

void test_control_dimension(int nx, int nu, bool drift_free, unsigned horizon_size) {
  // Create the shooting problem
  ShootingProblemFactory factory(nx, nu, drift_free, horizon_size);
  crocoddyl::ShootingProblem& shooting_problem = factory.get_shooting_problem();

  long int sum_control_dims = 0;
  for(unsigned i = 0 ; i < shooting_problem.get_runningModels().size() ; ++i)
  {
    sum_control_dims += shooting_problem.get_runningModels()[i]->get_nu();
  }
  sum_control_dims += shooting_problem.get_terminalModel()->get_nu();
  
  long int sum_control_dims_from_construction = nu * (horizon_size + 1);

  BOOST_CHECK( sum_control_dims == sum_control_dims_from_construction );
}

//____________________________________________________________________________//
//____________________________________________________________________________//

void test_get_T(int nx, int nu, bool drift_free, unsigned horizon_size) {
  // Create the shooting problem
  ShootingProblemFactory factory(nx, nu, drift_free, horizon_size);
  crocoddyl::ShootingProblem& shooting_problem = factory.get_shooting_problem();

  BOOST_CHECK( shooting_problem.get_T() == horizon_size );
}

//____________________________________________________________________________//
//____________________________________________________________________________//

void test_get_x0(int nx, int nu, bool drift_free, unsigned horizon_size) {
  // Create the shooting problem
  ShootingProblemFactory factory(nx, nu, drift_free, horizon_size);
  crocoddyl::ShootingProblem& shooting_problem = factory.get_shooting_problem();

  double tol = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
  BOOST_CHECK(( shooting_problem.get_x0() - factory.x0).isMuchSmallerThan(1.0, tol));
}

//____________________________________________________________________________//
//____________________________________________________________________________//

void test_data_dim(int nx, int nu, bool drift_free, unsigned horizon_size) {
  // Create the shooting problem
  ShootingProblemFactory factory(nx, nu, drift_free, horizon_size);
  crocoddyl::ShootingProblem& shooting_problem = factory.get_shooting_problem();

  for(unsigned i = 0 ; i < shooting_problem.get_runningDatas().size() ; ++i)
  {
    BOOST_CHECK( nx == shooting_problem.get_runningDatas()[i]->xnext.size());
    BOOST_CHECK( nx*nx == shooting_problem.get_runningDatas()[i]->Fx.size());
    BOOST_CHECK( nx*nu == shooting_problem.get_runningDatas()[i]->Fu.size());
    BOOST_CHECK( nx == shooting_problem.get_runningDatas()[i]->Lx.size());
    BOOST_CHECK( nu == shooting_problem.get_runningDatas()[i]->Lu.size());
    BOOST_CHECK( nx*nx == shooting_problem.get_runningDatas()[i]->Lxx.size());
    BOOST_CHECK( nx*nu == shooting_problem.get_runningDatas()[i]->Lxu.size());
    BOOST_CHECK( nu*nu == shooting_problem.get_runningDatas()[i]->Luu.size());
    BOOST_CHECK( 0 == shooting_problem.get_runningDatas()[i]->r.size());
    BOOST_CHECK( 0 == shooting_problem.get_runningDatas()[i]->Rx.size());
    BOOST_CHECK( 0 == shooting_problem.get_runningDatas()[i]->Ru.size());
  }

  BOOST_CHECK( nx == shooting_problem.get_terminalData()->xnext.size());
  BOOST_CHECK( nx*nx == shooting_problem.get_terminalData()->Fx.size());
  BOOST_CHECK( nx*nu == shooting_problem.get_terminalData()->Fu.size());
  BOOST_CHECK( nx == shooting_problem.get_terminalData()->Lx.size());
  BOOST_CHECK( nu == shooting_problem.get_terminalData()->Lu.size());
  BOOST_CHECK( nx*nx == shooting_problem.get_terminalData()->Lxx.size());
  BOOST_CHECK( nx*nu == shooting_problem.get_terminalData()->Lxu.size());
  BOOST_CHECK( nu*nu == shooting_problem.get_terminalData()->Luu.size());
  BOOST_CHECK( 0 == shooting_problem.get_terminalData()->r.size());
  BOOST_CHECK( 0 == shooting_problem.get_terminalData()->Rx.size());
  BOOST_CHECK( 0 == shooting_problem.get_terminalData()->Ru.size());
}

//____________________________________________________________________________//
//____________________________________________________________________________//

void test_calc() {
  /**
   * @TODO TestShootingProblem::test_calc to be improved, I just want to check
   * here that all costs and xnext have been computed properly.
   */

  std::cout << "Create the shooting problem" << std::endl;
  // Create the shooting problem
  int nx = 2;
  int nu = 1;
  bool drift_free = true;
  unsigned horizon_size = 3;
  ShootingProblemFactory factory(nx, nu, drift_free, horizon_size);
  crocoddyl::ShootingProblem& shooting_problem = factory.get_shooting_problem();

  std::vector<Eigen::VectorXd> x_vec;
  std::vector<Eigen::VectorXd> u_vec;
  for(unsigned i = 0 ; i < horizon_size + 1 ; ++i)
  {
    Eigen::VectorXd x, u;
    x.resize(nx);
    x.fill(1.0);
    u.resize(nu);
    u.fill(2.0);
    x_vec.push_back(x);
    u_vec.push_back(u);
  }

  shooting_problem.calc(x_vec, u_vec);
  BOOST_CHECK(shooting_problem.get_runningDatas()[0]->cost == 9);
  BOOST_CHECK(shooting_problem.get_runningDatas()[1]->cost == 9);
  BOOST_CHECK(shooting_problem.get_runningDatas()[2]->cost == 9);
  BOOST_CHECK(shooting_problem.get_terminalData()->cost == 3);

  double tol = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
  BOOST_CHECK( (shooting_problem.get_runningDatas()[0]->xnext - 
    (Eigen::VectorXd(2) << 3, 1).finished()).isMuchSmallerThan(1.0, tol) );
  BOOST_CHECK( (shooting_problem.get_runningDatas()[1]->xnext - 
    (Eigen::VectorXd(2) << 3, 1).finished()).isMuchSmallerThan(1.0, tol) );
  BOOST_CHECK( (shooting_problem.get_runningDatas()[2]->xnext - 
    (Eigen::VectorXd(2) << 3, 1).finished()).isMuchSmallerThan(1.0, tol) );
  BOOST_CHECK( (shooting_problem.get_terminalData()->xnext - 
    (Eigen::VectorXd(2) << 1, 1).finished()).isMuchSmallerThan(1.0, tol) );
  
}

//____________________________________________________________________________//
//____________________________________________________________________________//

void register_action_model_lqr_unit_tests() {
  int nx = 80;
  int nu = 40;
  bool drift_free = true;
  unsigned horizon_size = 3;

  // register the tests
  framework::master_test_suite().add(
    BOOST_TEST_CASE(boost::bind(&test_trajectory_dimension,
      nx, nu, drift_free, horizon_size ))
  );

  framework::master_test_suite().add(
    BOOST_TEST_CASE(boost::bind(&test_trajectory_dimension,
      nx, nu, drift_free, horizon_size ))
  );

  framework::master_test_suite().add(
    BOOST_TEST_CASE(boost::bind(&test_get_T,
      nx, nu, drift_free, horizon_size ))
  );

  framework::master_test_suite().add(
    BOOST_TEST_CASE(boost::bind(&test_get_x0,
      nx, nu, drift_free, horizon_size ))
  );

  framework::master_test_suite().add(
    BOOST_TEST_CASE(boost::bind(&test_data_dim,
      nx, nu, drift_free, horizon_size ))
  );
  
  framework::master_test_suite().add(
    BOOST_TEST_CASE(&test_calc)
  );
  
}

//____________________________________________________________________________//

bool init_function() {
  // Here we test the state_vector
  register_action_model_lqr_unit_tests();
  return true;
}

//____________________________________________________________________________//

int main(int argc, char* argv[]) { return ::boost::unit_test::unit_test_main(&init_function, argc, argv); }
