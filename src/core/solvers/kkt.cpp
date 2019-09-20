///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


#include "crocoddyl/core/solvers/kkt.hpp"

namespace crocoddyl {

SolverKKT::SolverKKT(ShootingProblem& problem)
    : SolverAbstract(problem),
      regfactor_(10.),
      regmin_(1e-9),
      regmax_(1e9),
      cost_try_(0.),
      th_grad_(1e-12),
      th_step_(0.5),
      was_feasible_(false) {
  allocateData();

  const unsigned int& n_alphas = 10;
  alphas_.resize(n_alphas);
  for (unsigned int n = 0; n < n_alphas; ++n) {
    alphas_[n] = 1. / pow(2., (double)n);
  }
}

SolverKKT::~SolverKKT() {}

bool SolverKKT::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                      const unsigned int& maxiter, const bool& is_feasible, const double& reginit) {
               throw(std::runtime_error("SolverKKT::solve() this method is not implemented yet"));
             }

void SolverKKT::computeDirection(const bool& recalc){
  throw(std::runtime_error("SolverKKT::computeDirection() this method is not implemented yet"));
}

double SolverKKT::tryStep(const double& steplength){
    throw(std::runtime_error("SolverKKT::tryStep() this method is not implemented yet"));

}

double SolverKKT::stoppingCriteria(){
    throw(std::runtime_error("SolverKKT::stoppingCriteria() this method is not implemented yet"));

}

const Eigen::Vector2d& SolverKKT::expectedImprovement(){
      throw(std::runtime_error("SolverKKT::stoppingCriteria() this method is not implemented yet"));
}


double SolverKKT::calc(){
  const long unsigned int& T = problem_.get_T();
  // problem calc diff 
  cost_ = problem_.calcDiff(xs_, us_);
  // some indices 
  int ix = 0; 
  int iu = 0; 
  //offset on constraint xnext = f(x,u) due to x0 = ref.
  const unsigned int cx0 = problem_.get_runningModels()[0]->get_ndx(); 
  // fill diagonals of dynamics gradient as identity 
  // block starts,ends, rows,columns
  kkt_.block(ndx_+nu_,0,ndx_,ndx_) = Eigen::MatrixXd::Identity(ndx_,ndx_); 
  // loop over models and fill out kkt matrix 
  for (long unsigned int t = 0; t < T; ++t) {
    ActionModelAbstract* m = problem_.running_models_[t];
    boost::shared_ptr<ActionDataAbstract>& d = problem_.running_datas_[t];
    const unsigned int ndxi = m->get_ndx();
    const unsigned int nui = m->get_nu();
    // compute gap at initial state 
    if(t==0){
      m->get_state()->diff(problem_.get_x0(), xs_[0], kktref_.segment(ndx_+nu_, ndxi));
    }
    // fill the kkt matrix 
    // hessian
    kkt_.block(ix,ix,ndxi,ndxi) = d->get_Lxx();
    kkt_.block(ix,ndx_+iu,ndxi,nui) = d->get_Lxu();
    kkt_.block(ndx_+iu,ix,nui,ndxi) = d->get_Lxu().transpose();
    kkt_.block(ndx_+iu,ndx_+iu,nui,nui) = d->get_Luu(); 
    // jacobian 
    kkt_.block(ndx_+nu_+cx0+ix, ix, ndxi, ndxi) = - d->get_Fx();
    kkt_.block(ndx_+nu_+cx0+ix,ndx_+iu, ndxi, nui) = - d->get_Fu();
    // fill kkt refernce 
    kktref_.segment(ix, ndxi) = d->get_Lx();
    kktref_.segment(ndx_+iu, nui) = d->get_Lu();
    // constraint value = x_guess - x_ref = diff(x_ref,x_guess)
    m->get_state()->diff(d->get_xnext(), xs_[t + 1], kktref_.segment(ndx_+nu_+cx0+ix, ndxi));
    // increment node index for state and control 
    ix += ndxi; 
    iu += nui;
  }
  // do terminal model 
  boost::shared_ptr<ActionDataAbstract>& df = problem_.terminal_data_;
  const unsigned int ndxf = problem_.terminal_model_->get_ndx();
  kkt_.block(ix,ix,ndxf,ndxf) = df->get_Lxx();
  kktref_.segment(ix, ndxf) = df->get_Lx();
  // jacobian transpose 
  kkt_.block(0,ndx_,ndx_+nu_,ndx_) = kkt_.block(ndx_, 0, ndx_,ndx_+nu_).transpose();
  // add regularization 
  if (!std::isnan(xreg_)) {
    kkt_.block(0,0, ndx_,ndx_).diagonal() += Eigen::VectorXd::Constant(ndx_, xreg_);
  }
  if (!std::isnan(ureg_)) {
    kkt_.block(ndx_,ndx_, nu_,nu_).diagonal() += Eigen::VectorXd::Constant(nu_, ureg_);
  }
  // return cost 
   return cost_;

}


void SolverKKT::computePrimalDual(){
  throw(std::runtime_error("SolverKKT::computePrimalDual() this method is not implemented yet"));
}



void SolverKKT::allocateData() {
  const long unsigned int& T = problem_.get_T();
 
  gaps_.resize(T + 1);

  xs_try_.resize(T + 1);
  us_try_.resize(T);
  dx_.resize(T);
  nx_ = 0; 
  ndx_ = 0;
  nu_ = 0; 


  for (long unsigned int t = 0; t < T; ++t) {
    ActionModelAbstract* model = problem_.running_models_[t];
    const int& nx = model->get_nx();
    const int& ndx = model->get_ndx();
    const int& nu = model->get_nu();

    if (t == 0) {
      xs_try_[t] = problem_.get_x0();
    } else {
      xs_try_[t] = Eigen::VectorXd::Constant(nx, NAN);
    }
    us_try_[t] = Eigen::VectorXd::Constant(nu, NAN);
    dx_[t] = Eigen::VectorXd::Zero(ndx);
    nx_ += nx; 
    ndx_ += ndx;
    nu_ += nu;   
  }
  // add terminal model 
  ActionModelAbstract* model = problem_.get_terminalModel();
  nx_ +=  model->get_nx();
  ndx_ +=  model->get_ndx();

  // set dimensions for kkt matrix and kkt_ref vector 
  kkt_.resize(2*ndx_+nu_, 2*ndx_+nu_);
  kkt_.setZero(); 
  kktref_.resize(2*ndx_+nu_);
  kktref_.setZero();



}

const int& SolverKKT::get_nx() const { return nx_; }
const int& SolverKKT::get_ndx() const { return ndx_; }
const int& SolverKKT::get_nu() const { return nu_; } 
const Eigen::MatrixXd& SolverKKT::get_kkt() const {return kkt_;}
const Eigen::VectorXd& SolverKKT::get_kktref() const{return kktref_;}









}  // namespace crocoddyl