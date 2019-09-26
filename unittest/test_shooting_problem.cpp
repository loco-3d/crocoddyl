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

/**
 * @brief This class is an extremely simple lqr with identity matrices
 */
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
template <typename ActionModelLQRType>
class ShootingProblemFactory{
public:
  /**
   * @brief Construct a new ShootingProblemFactory object
   * 
   * @tparam ActionModelLQRType 
   * @param nx is the dimension of the state
   * @param nu is the dimension of teh control 
   * @param drift_free Is a ActionLQR parameter
   * @param horizon_size
   */
  ShootingProblemFactory(int nx, int nu, bool drift_free,
                         unsigned horizon_size)
  {
    running_models_.clear();
    for(unsigned i = 0 ; i < horizon_size ; ++i)
    {
      running_models_.push_back(new ActionModelLQRType(nx, nu, drift_free));
    }
    terminal_model_ = new ActionModelLQRType (nx, nu, drift_free);
    x0_ = terminal_model_->get_state().rand();
    x0_.fill(1.0);
    shooting_problem_ = 
      new crocoddyl::ShootingProblem(x0_, running_models_, terminal_model_);
  }

  /**
   * @brief Destroy the ShootingProblemFactory object
   */
  ~ShootingProblemFactory()
  {
    for(unsigned i = 0 ; i < running_models_.size() ; ++i)
    {
      if(running_models_[i] != NULL)
      {
        delete running_models_[i];
        running_models_[i] = NULL;
      }
    }
    running_models_.clear();
    if(terminal_model_ != NULL)
    {
      delete terminal_model_;
      terminal_model_ = NULL;
    }
    if(shooting_problem_ != NULL)
    {
      delete shooting_problem_;
      shooting_problem_ = NULL;
    }
  }

  const Eigen::VectorXd& get_x0() const
  {
    return x0_;
  }

  Eigen::VectorXd& get_x0()
  {
    return x0_;
  }

  /**
   * @brief Get the shooting_problem object
   * 
   * @return crocoddyl::ShootingProblem& 
   */
  crocoddyl::ShootingProblem& get_shooting_problem()
  {
    return *shooting_problem_;
  }
private:
  std::vector<crocoddyl::ActionModelAbstract*> running_models_; 
  ActionModelLQRType* terminal_model_;
  Eigen::VectorXd x0_;
  crocoddyl::ShootingProblem* shooting_problem_;
};

//____________________________________________________________________________//
//____________________________________________________________________________//

void test_trajectory_dimension(int nx, int nu, bool drift_free, unsigned horizon_size) {
  // Create the shooting problem
  ShootingProblemFactory<crocoddyl::ActionModelLQR> factory(
    nx, nu, drift_free, horizon_size);
  crocoddyl::ShootingProblem& shooting_problem = factory.get_shooting_problem();

  // Compute the sum of all the state dimensions
  long int sum_state_dims = 0;
  for(unsigned i = 0 ; i < shooting_problem.get_runningModels().size() ; ++i)
  {
    Eigen::VectorXd x = shooting_problem.get_runningModels()[i]->get_state().zero();
    sum_state_dims += x.size();
  }
  Eigen::VectorXd x = shooting_problem.get_terminalModel()->get_state().zero();
  sum_state_dims += x.size();

  int sum_model_state_dims = 0;
  for(unsigned i = 0 ; i < shooting_problem.get_runningModels().size() ; ++i)
  {
    sum_model_state_dims += shooting_problem.get_runningModels()[i]->get_state().get_nx();
  }
  sum_model_state_dims += shooting_problem.get_terminalModel()->get_state().get_nx();

  BOOST_CHECK( sum_model_state_dims == sum_state_dims );
}

//____________________________________________________________________________//
//____________________________________________________________________________//

void test_control_dimension(int nx, int nu, bool drift_free, unsigned horizon_size) {
  // Create the shooting problem
  ShootingProblemFactory<crocoddyl::ActionModelLQR> factory(
    nx, nu, drift_free, horizon_size);
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
  ShootingProblemFactory<crocoddyl::ActionModelLQR> factory(
    nx, nu, drift_free, horizon_size);
  crocoddyl::ShootingProblem& shooting_problem = factory.get_shooting_problem();

  BOOST_CHECK( shooting_problem.get_T() == horizon_size );
}

//____________________________________________________________________________//
//____________________________________________________________________________//

void test_get_x0(int nx, int nu, bool drift_free, unsigned horizon_size) {
  // Create the shooting problem
  ShootingProblemFactory<crocoddyl::ActionModelLQR> factory(
    nx, nu, drift_free, horizon_size);
  crocoddyl::ShootingProblem& shooting_problem = factory.get_shooting_problem();

  double tol = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
  BOOST_CHECK(( shooting_problem.get_x0() - factory.get_x0()).isMuchSmallerThan(1.0, tol));
}

//____________________________________________________________________________//
//____________________________________________________________________________//

void test_data_dim(int nx, int nu, bool drift_free, unsigned horizon_size) {
  // Create the shooting problem
  ShootingProblemFactory<crocoddyl::ActionModelLQR> factory(
    nx, nu, drift_free, horizon_size);
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
}

//____________________________________________________________________________//
//____________________________________________________________________________//

void test_calc() {
  /**
   * @TODO TestShootingProblem::test_calc to be improved, I just want to check
   * here that all costs and xnext have been computed properly.
   */

  // Create the shooting problem
  int nx = 2;
  int nu = 1;
  bool drift_free = true;
  unsigned horizon_size = 3;
  ShootingProblemFactory<ActionModelLQRIdentity> factory(
    nx, nu, drift_free, horizon_size);
  crocoddyl::ShootingProblem& shooting_problem = factory.get_shooting_problem();

  std::vector<Eigen::VectorXd> x_vec(horizon_size + 1, Eigen::VectorXd(nx));
  std::vector<Eigen::VectorXd> u_vec(horizon_size, Eigen::VectorXd(nu));
  for(unsigned i = 0 ; i < x_vec.size() ; ++i){x_vec[i].fill(1.0);}
  for(unsigned i = 0 ; i < u_vec.size() ; ++i){u_vec[i].fill(2.0);}


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

void test_calc_diff(int nx, int nu, bool drift_free, unsigned horizon_size) {
  /**
   * @TODO TestShootingProblem::test_calc_diff to be improved, I just want to check
   * here that all costs and xnext have been computed properly.
   */

  // Create the shooting problems
  ShootingProblemFactory<crocoddyl::ActionModelLQR> factory(
    nx, nu, drift_free, horizon_size);
  ShootingProblemFactory<crocoddyl::ActionModelLQR> factory2(
    nx, nu, drift_free, horizon_size);
  crocoddyl::ShootingProblem& shooting_problem = factory.get_shooting_problem();
  crocoddyl::ShootingProblem& shooting_problem2 = factory.get_shooting_problem();

  std::vector<Eigen::VectorXd> x_vec(horizon_size + 1, Eigen::VectorXd(nx));
  std::vector<Eigen::VectorXd> u_vec(horizon_size, Eigen::VectorXd(nu));
  for(unsigned i = 0 ; i < x_vec.size() ; ++i){x_vec[i].fill(1.0);}
  for(unsigned i = 0 ; i < u_vec.size() ; ++i){u_vec[i].fill(2.0);}

  shooting_problem.calc(x_vec, u_vec);
  shooting_problem.calcDiff(x_vec, u_vec);

  BOOST_CHECK(shooting_problem.get_runningDatas()[0]->cost ==
              shooting_problem2.get_runningDatas()[0]->cost);
  BOOST_CHECK(shooting_problem.get_runningDatas()[1]->cost ==
              shooting_problem2.get_runningDatas()[1]->cost);
  BOOST_CHECK(shooting_problem.get_runningDatas()[2]->cost ==
              shooting_problem2.get_runningDatas()[2]->cost);
  BOOST_CHECK(shooting_problem.get_terminalData()->cost == 
              shooting_problem2.get_terminalData()->cost);

  double tol = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
  BOOST_CHECK( (shooting_problem.get_runningDatas()[0]->xnext - 
    shooting_problem2.get_runningDatas()[0]->xnext).isMuchSmallerThan(1.0, tol) );
  BOOST_CHECK( (shooting_problem.get_runningDatas()[1]->xnext - 
    shooting_problem2.get_runningDatas()[1]->xnext).isMuchSmallerThan(1.0, tol) );
  BOOST_CHECK( (shooting_problem.get_runningDatas()[2]->xnext - 
    shooting_problem2.get_runningDatas()[2]->xnext).isMuchSmallerThan(1.0, tol) );
  BOOST_CHECK( (shooting_problem.get_terminalData()->xnext -
    shooting_problem2.get_terminalData()->xnext).isMuchSmallerThan(1.0, tol) );
}

//____________________________________________________________________________//
//____________________________________________________________________________//

void test_rollout() {
  /**
   * @TODO TestShootingProblem::test_calc_diff to be improved, I just want to check
   * here that all costs and xnext have been computed properly.
   */

  // Create the shooting problems
  // Create the shooting problem
  int nx = 2;
  int nu = 1;
  bool drift_free = true;
  unsigned horizon_size = 3;
  ShootingProblemFactory<ActionModelLQRIdentity> factory(
    nx, nu, drift_free, horizon_size);
  crocoddyl::ShootingProblem& shooting_problem = factory.get_shooting_problem();

  std::vector<Eigen::VectorXd> x_vec(horizon_size + 1, Eigen::VectorXd(nx));
  std::vector<Eigen::VectorXd> u_vec(horizon_size, Eigen::VectorXd(nu));
  for(unsigned i = 0 ; i < x_vec.size() ; ++i){x_vec[i].fill(1.0);}
  for(unsigned i = 0 ; i < u_vec.size() ; ++i){u_vec[i].fill(2.0);}

  shooting_problem.rollout(u_vec, x_vec);
  
  std::cout << shooting_problem.get_terminalData()->cost
            << " ; "
            << shooting_problem.get_terminalData()->xnext.transpose()
            << std::endl;

  BOOST_CHECK(shooting_problem.get_runningDatas()[0]->cost == 9);
  BOOST_CHECK(shooting_problem.get_runningDatas()[1]->cost == 19);
  BOOST_CHECK(shooting_problem.get_runningDatas()[2]->cost == 33);
  BOOST_CHECK(shooting_problem.get_terminalData()->cost == 
              shooting_problem.get_runningDatas().back()->cost);

  double tol = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
  BOOST_CHECK( (shooting_problem.get_runningDatas()[0]->xnext - 
    (Eigen::VectorXd(2) << 3, 1).finished()).isMuchSmallerThan(1.0, tol) );
  BOOST_CHECK( (shooting_problem.get_runningDatas()[1]->xnext - 
    (Eigen::VectorXd(2) << 5, 1).finished()).isMuchSmallerThan(1.0, tol) );
  BOOST_CHECK( (shooting_problem.get_runningDatas()[2]->xnext - 
    (Eigen::VectorXd(2) << 7, 1).finished()).isMuchSmallerThan(1.0, tol) );
  BOOST_CHECK( (shooting_problem.get_terminalData()->xnext - 
    shooting_problem.get_runningDatas().back()->xnext).isMuchSmallerThan(1.0, tol) );
}

//____________________________________________________________________________//
//____________________________________________________________________________//

void test_rollout_us() {
  /**
   * @TODO TestShootingProblem::test_calc_diff to be improved, I just want to check
   * here that all costs and xnext have been computed properly.
   */

  // Create the shooting problems
  // Create the shooting problem
  int nx = 2;
  int nu = 1;
  bool drift_free = true;
  unsigned horizon_size = 3;
  ShootingProblemFactory<ActionModelLQRIdentity> factory(
    nx, nu, drift_free, horizon_size);
  crocoddyl::ShootingProblem& shooting_problem = factory.get_shooting_problem();

  std::vector<Eigen::VectorXd> x_vec(horizon_size + 1, Eigen::VectorXd(nx));
  std::vector<Eigen::VectorXd> u_vec(horizon_size, Eigen::VectorXd(nu));
  for(unsigned i = 0 ; i < x_vec.size() ; ++i){x_vec[i].fill(1.0);}
  for(unsigned i = 0 ; i < u_vec.size() ; ++i){u_vec[i].fill(2.0);}

  shooting_problem.rollout_us(u_vec);
  
  BOOST_CHECK(shooting_problem.get_runningDatas()[0]->cost == 9);
  BOOST_CHECK(shooting_problem.get_runningDatas()[1]->cost == 19);
  BOOST_CHECK(shooting_problem.get_runningDatas()[2]->cost == 33);
  BOOST_CHECK(shooting_problem.get_terminalData()->cost == 
              shooting_problem.get_runningDatas().back()->cost);

  double tol = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());
  BOOST_CHECK( (shooting_problem.get_runningDatas()[0]->xnext - 
    (Eigen::VectorXd(2) << 3, 1).finished()).isMuchSmallerThan(1.0, tol) );
  BOOST_CHECK( (shooting_problem.get_runningDatas()[1]->xnext - 
    (Eigen::VectorXd(2) << 5, 1).finished()).isMuchSmallerThan(1.0, tol) );
  BOOST_CHECK( (shooting_problem.get_runningDatas()[2]->xnext - 
    (Eigen::VectorXd(2) << 7, 1).finished()).isMuchSmallerThan(1.0, tol) );
  BOOST_CHECK( (shooting_problem.get_terminalData()->xnext - 
    shooting_problem.get_runningDatas().back()->xnext).isMuchSmallerThan(1.0, tol) );
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
  
  framework::master_test_suite().add(
    BOOST_TEST_CASE(boost::bind(&test_calc_diff,
    nx, nu, drift_free, horizon_size ))
  );

  framework::master_test_suite().add(
    BOOST_TEST_CASE(&test_rollout)
  );

  framework::master_test_suite().add(
    BOOST_TEST_CASE(&test_rollout_us)
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
