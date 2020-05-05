///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifdef WITH_MULTITHREADING
#include <omp.h>
#define NUM_THREADS WITH_NTHREADS
#endif  // WITH_MULTITHREADING

namespace crocoddyl {

template <typename Scalar>
ShootingProblemTpl<Scalar>::ShootingProblemTpl(
    const VectorXs& x0, const std::vector<boost::shared_ptr<ActionModelAbstract> >& running_models,
    boost::shared_ptr<ActionModelAbstract> terminal_model)
    : cost_(Scalar(0.)),
      T_(running_models.size()),
      x0_(x0),
      terminal_model_(terminal_model),
      running_models_(running_models) {
  if (static_cast<std::size_t>(x0.size()) != running_models_[0]->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x0 has wrong dimension (it should be " +
                        std::to_string(running_models_[0]->get_state()->get_nx()) + ")");
  }
  allocateData();
}

template <typename Scalar>
ShootingProblemTpl<Scalar>::~ShootingProblemTpl() {}

template <typename Scalar>
Scalar ShootingProblemTpl<Scalar>::calc(const std::vector<VectorXs>& xs, const std::vector<VectorXs>& us) {
  if (xs.size() != T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "xs has wrong dimension (it should be " + std::to_string(T_ + 1) + ")");
  }
  if (us.size() != T_) {
    throw_pretty("Invalid argument: "
                 << "us has wrong dimension (it should be " + std::to_string(T_) + ")");
  }

#ifdef WITH_MULTITHREADING
#pragma omp parallel for
#endif
  for (std::size_t i = 0; i < T_; ++i) {
    running_models_[i]->calc(running_datas_[i], xs[i], us[i]);
  }
  terminal_model_->calc(terminal_data_, xs.back());

  cost_ = Scalar(0.);
  for (std::size_t i = 0; i < T_; ++i) {
    cost_ += running_datas_[i]->cost;
  }
  cost_ += terminal_data_->cost;
  return cost_;
}

template <typename Scalar>
Scalar ShootingProblemTpl<Scalar>::calcDiff(const std::vector<VectorXs>& xs, const std::vector<VectorXs>& us) {
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
#pragma omp parallel for
#endif
  for (i = 0; i < T_; ++i) {
    running_models_[i]->calcDiff(running_datas_[i], xs[i], us[i]);
  }
  terminal_model_->calcDiff(terminal_data_, xs.back());

  cost_ = Scalar(0.);
  for (std::size_t i = 0; i < T_; ++i) {
    cost_ += running_datas_[i]->cost;
  }
  cost_ += terminal_data_->cost;

  return cost_;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::rollout(const std::vector<VectorXs>& us, std::vector<VectorXs>& xs) {
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
    const VectorXs& x = xs[i];
    const VectorXs& u = us[i];

    model->calc(data, x, u);
    xs[i + 1] = data->xnext;
  }
  terminal_model_->calc(terminal_data_, xs.back());
}

template <typename Scalar>
std::vector<typename MathBaseTpl<Scalar>::VectorXs> ShootingProblemTpl<Scalar>::rollout_us(
    const std::vector<VectorXs>& us) {
  std::vector<VectorXs> xs;
  xs.resize(T_ + 1);
  rollout(us, xs);
  return xs;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::updateModel(std::size_t i, boost::shared_ptr<ActionModelAbstract> model) {
  if (i > T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "i is bigger than the allocated horizon (it should be lower than " + std::to_string(T_) + ")");
  }
  if (i == T_ + 1) {
    set_terminalModel(model);
  } else {
    running_models_[i] = model;
    running_datas_[i] = model->createData();
  }
}

template <typename Scalar>
const std::size_t& ShootingProblemTpl<Scalar>::get_T() const {
  return T_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ShootingProblemTpl<Scalar>::get_x0() const {
  return x0_;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::allocateData() {
  for (std::size_t i = 0; i < T_; ++i) {
    const boost::shared_ptr<ActionModelAbstract>& model = running_models_[i];
    running_datas_.push_back(model->createData());
  }
  terminal_data_ = terminal_model_->createData();
}

template <typename Scalar>
const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > >&
ShootingProblemTpl<Scalar>::get_runningModels() const {
  return running_models_;
}

template <typename Scalar>
const boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> >& ShootingProblemTpl<Scalar>::get_terminalModel()
    const {
  return terminal_model_;
}

template <typename Scalar>
const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > >&
ShootingProblemTpl<Scalar>::get_runningDatas() const {
  return running_datas_;
}

template <typename Scalar>
const boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> >& ShootingProblemTpl<Scalar>::get_terminalData()
    const {
  return terminal_data_;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::set_x0(const VectorXs& x0_in) {
  if (x0_in.size() != x0_.size()) {
    throw_pretty("Invalid argument: "
                 << "invalid size of x0 provided.");
  }
  x0_ = x0_in;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::set_runningModels(
    const std::vector<boost::shared_ptr<ActionModelAbstract> >& models) {
  T_ = models.size();
  running_models_ = models;
  running_datas_.clear();
  for (std::size_t i = 0; i < T_; ++i) {
    const boost::shared_ptr<ActionModelAbstract>& model = running_models_[i];
    running_datas_.push_back(model->createData());
  }
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::set_terminalModel(boost::shared_ptr<ActionModelAbstract> model) {
  terminal_model_ = model;
  terminal_data_ = terminal_model_->createData();
}

}  // namespace crocoddyl
