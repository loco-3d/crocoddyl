///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include <iostream>
#ifdef WITH_MULTITHREADING
#include <omp.h>
#define NUM_THREADS WITH_NTHREADS
#endif  // WITH_MULTITHREADING

namespace crocoddyl {

ShootingProblem::ShootingProblem(const Eigen::VectorXd& x0,
                                 const std::vector<boost::shared_ptr<ActionModelAbstract> >& running_models,
                                 boost::shared_ptr<ActionModelAbstract> terminal_model)
    : cost_(0.), T_(running_models.size()), x0_(x0), terminal_model_(terminal_model), running_models_(running_models) {
  if (static_cast<std::size_t>(x0.size()) != running_models_[0]->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x0 has wrong dimension (it should be " +
                        std::to_string(running_models_[0]->get_state()->get_nx()) + ")");
  }
  allocateData();
}

ShootingProblem::~ShootingProblem() {}

double ShootingProblem::calc(const std::vector<Eigen::VectorXd>& xs, const std::vector<Eigen::VectorXd>& us) {
  if (xs.size() != T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "xs has wrong dimension (it should be " + std::to_string(T_ + 1) + ")");
  }
  if (us.size() != T_) {
    throw_pretty("Invalid argument: "
                 << "us has wrong dimension (it should be " + std::to_string(T_) + ")");
  }

  cost_ = 0;
  for (std::size_t i = 0; i < T_; ++i) {
    const boost::shared_ptr<ActionModelAbstract>& model = running_models_[i];
    const boost::shared_ptr<ActionDataAbstract>& data = running_datas_[i];
    const Eigen::VectorXd& x = xs[i];
    const Eigen::VectorXd& u = us[i];
    model->calc(data, x, u);
  }
  terminal_model_->calc(terminal_data_, xs.back());
  
  for (std::size_t i = 0; i < T_; ++i) {
    cost_ += running_datas_[i]->cost;
  }  
  cost_ += terminal_data_->cost;
  return cost_;
}

double ShootingProblem::calcDiff(const std::vector<Eigen::VectorXd>& xs, const std::vector<Eigen::VectorXd>& us,
                                 const bool& recalc) {
  if (xs.size() != T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "xs has wrong dimension (it should be " + std::to_string(T_ + 1) + ")");
  }
  if (us.size() != T_) {
    throw_pretty("Invalid argument: "
                 << "us has wrong dimension (it should be " + std::to_string(T_) + ")");
  }

  std::size_t i;

#ifdef WITH_MULTITHREADING
  omp_set_num_threads(NUM_THREADS);
#endif

  
  if (recalc) {
#ifdef WITH_MULTITHREADING
#pragma omp parallel for
#endif
    for (i = 0; i < T_; ++i) {
      running_models_[i]->calc(running_datas_[i], xs[i], us[i]);
    }
    terminal_model_->calc(terminal_data_, xs.back());
  }
  
#ifdef WITH_MULTITHREADING
#pragma omp parallel for
#endif
  for (i = 0; i < T_; ++i) {
    running_models_[i]->calcDiff(running_datas_[i], xs[i], us[i], false);
  }
  terminal_model_->calcDiff(terminal_data_, xs.back(), false);

  cost_ = 0;
  for (std::size_t i = 0; i < T_; ++i) {
    cost_ += running_datas_[i]->cost;
  }
  cost_ += terminal_data_->cost;

  return cost_;
}

void ShootingProblem::rollout(const std::vector<Eigen::VectorXd>& us, std::vector<Eigen::VectorXd>& xs) {
  if (xs.size() != T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "xs has wrong dimension (it should be " + std::to_string(T_ + 1) + ")");
  }
  if (us.size() != T_) {
    throw_pretty("Invalid argument: "
                 << "us has wrong dimension (it should be " + std::to_string(T_) + ")");
  }

  xs[0] = x0_;
  for (std::size_t i = 0; i < T_; ++i) {
    const boost::shared_ptr<ActionModelAbstract>& model = running_models_[i];
    const boost::shared_ptr<ActionDataAbstract>& data = running_datas_[i];
    const Eigen::VectorXd& x = xs[i];
    const Eigen::VectorXd& u = us[i];

    model->calc(data, x, u);
    xs[i + 1] = data->xnext;
  }
  terminal_model_->calc(terminal_data_, xs.back());
}

std::vector<Eigen::VectorXd> ShootingProblem::rollout_us(const std::vector<Eigen::VectorXd>& us) {
  std::vector<Eigen::VectorXd> xs;
  xs.resize(T_ + 1);
  rollout(us, xs);
  return xs;
}

const std::size_t& ShootingProblem::get_T() const { return T_; }

const Eigen::VectorXd& ShootingProblem::get_x0() const { return x0_; }

void ShootingProblem::allocateData() {
  for (std::size_t i = 0; i < T_; ++i) {
    const boost::shared_ptr<ActionModelAbstract>& model = running_models_[i];
    running_datas_.push_back(model->createData());
  }
  terminal_data_ = terminal_model_->createData();
}

const std::vector<boost::shared_ptr<ActionModelAbstract> >& ShootingProblem::get_runningModels() const {
  return running_models_;
}

const boost::shared_ptr<ActionModelAbstract>& ShootingProblem::get_terminalModel() const { return terminal_model_; }

const std::vector<boost::shared_ptr<ActionDataAbstract> >& ShootingProblem::get_runningDatas() const {
  return running_datas_;
}

const boost::shared_ptr<ActionDataAbstract>& ShootingProblem::get_terminalData() const { return terminal_data_; }

void ShootingProblem::set_x0(const Eigen::VectorXd& x0_in) {
  if (x0_in.size() != x0_.size()) {
    throw_pretty("Invalid argument: "
                 << "invalid size of x0 provided.");
  }
  x0_ = x0_in;
}

void ShootingProblem::set_runningModels(const std::vector<boost::shared_ptr<ActionModelAbstract> >& models) {
  T_ = models.size();
  running_models_ = models;
  running_datas_.clear();
  for (std::size_t i = 0; i < T_; ++i) {
    const boost::shared_ptr<ActionModelAbstract>& model = running_models_[i];
    running_datas_.push_back(model->createData());
  }
}

void ShootingProblem::set_terminalModel(boost::shared_ptr<ActionModelAbstract> model) {
  terminal_model_ = model;
  terminal_data_ = terminal_model_->createData();
}

}  // namespace crocoddyl
