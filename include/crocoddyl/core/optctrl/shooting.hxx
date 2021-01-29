///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifdef CROCODDYL_WITH_MULTITHREADING
#include <omp.h>
#define NUM_THREADS CROCODDYL_WITH_NTHREADS
#endif // CROCODDYL_WITH_MULTITHREADING

namespace crocoddyl {

template <typename Scalar>
ShootingProblemTpl<Scalar>::ShootingProblemTpl(
    const VectorXs &x0,
    const std::vector<boost::shared_ptr<ActionModelAbstract>> &running_models,
    boost::shared_ptr<ActionModelAbstract> terminal_model)
    : cost_(Scalar(0.)), T_(running_models.size()), x0_(x0),
      terminal_model_(terminal_model), running_models_(running_models),
      nx_(running_models[0]->get_state()->get_nx()),
      ndx_(running_models[0]->get_state()->get_ndx()),
      nu_max_(running_models[0]->get_nu()) {
  for (std::size_t i = 1; i < T_; ++i) {
    const boost::shared_ptr<ActionModelAbstract> &model = running_models_[i];
    const std::size_t &nu = model->get_nu();
    if (nu_max_ < nu) {
      nu_max_ = nu;
    }
  }
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x0 has wrong dimension (it should be " +
                        std::to_string(nx_) + ")");
  }
  for (std::size_t i = 1; i < T_; ++i) {
    const boost::shared_ptr<ActionModelAbstract> &model = running_models_[i];
    if (model->get_state()->get_nx() != nx_) {
      throw_pretty("Invalid argument: "
                   << "nx in " << i
                   << " node is not consistent with the other nodes")
    }
    if (model->get_state()->get_ndx() != ndx_) {
      throw_pretty("Invalid argument: "
                   << "ndx in " << i
                   << " node is not consistent with the other nodes")
    }
  }
  if (terminal_model_->get_state()->get_nx() != nx_) {
    throw_pretty(
        "Invalid argument: "
        << "nx in terminal node is not consistent with the other nodes")
  }
  if (terminal_model_->get_state()->get_ndx() != ndx_) {
    throw_pretty(
        "Invalid argument: "
        << "ndx in terminal node is not consistent with the other nodes")
  }
  allocateData();
}

template <typename Scalar>
ShootingProblemTpl<Scalar>::ShootingProblemTpl(
    const VectorXs &x0,
    const std::vector<boost::shared_ptr<ActionModelAbstract>> &running_models,
    boost::shared_ptr<ActionModelAbstract> terminal_model,
    const std::vector<boost::shared_ptr<ActionDataAbstract>> &running_datas,
    boost::shared_ptr<ActionDataAbstract> terminal_data)
    : cost_(Scalar(0.)), T_(running_models.size()), x0_(x0),
      terminal_model_(terminal_model), terminal_data_(terminal_data),
      running_models_(running_models), running_datas_(running_datas),
      nx_(running_models[0]->get_state()->get_nx()),
      ndx_(running_models[0]->get_state()->get_ndx()),
      nu_max_(running_models[0]->get_nu()) {
  for (std::size_t i = 1; i < T_; ++i) {
    const boost::shared_ptr<ActionModelAbstract> &model = running_models_[i];
    const std::size_t &nu = model->get_nu();
    if (nu_max_ < nu) {
      nu_max_ = nu;
    }
  }
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "x0 has wrong dimension (it should be " +
                        std::to_string(nx_) + ")");
  }
  std::size_t Td = running_datas.size();
  if (Td != T_) {
    throw_pretty(
        "Invalid argument: "
        << "the number of running models and datas are not the same (" +
               std::to_string(T_) + " != " + std::to_string(Td) + ")")
  }
  for (std::size_t i = 0; i < T_; ++i) {
    const boost::shared_ptr<ActionModelAbstract> &model = running_models_[i];
    const boost::shared_ptr<ActionDataAbstract> &data = running_datas_[i];
    if (model->get_state()->get_nx() != nx_) {
      throw_pretty("Invalid argument: "
                   << "nx in " << i
                   << " node is not consistent with the other nodes")
    }
    if (model->get_state()->get_ndx() != ndx_) {
      throw_pretty("Invalid argument: "
                   << "ndx in " << i
                   << " node is not consistent with the other nodes")
    }
    if (!model->checkData(data)) {
      throw_pretty("Invalid argument: "
                   << "action data in " << i
                   << " node is not consistent with the action model")
    }
  }
  if (!terminal_model->checkData(terminal_data)) {
    throw_pretty("Invalid argument: "
                 << "terminal action data is not consistent with the terminal "
                    "action model")
  }
}

template <typename Scalar>
ShootingProblemTpl<Scalar>::ShootingProblemTpl(
    const ShootingProblemTpl<Scalar> &problem)
    : cost_(Scalar(0.)), T_(problem.get_T()), x0_(problem.get_x0()),
      terminal_model_(problem.get_terminalModel()),
      terminal_data_(problem.get_terminalData()),
      running_models_(problem.get_runningModels()),
      running_datas_(problem.get_runningDatas()), nx_(problem.get_nx()),
      ndx_(problem.get_ndx()), nu_max_(problem.get_nu_max()) {}

template <typename Scalar> ShootingProblemTpl<Scalar>::~ShootingProblemTpl() {}

template <typename Scalar>
Scalar ShootingProblemTpl<Scalar>::calc(const std::vector<VectorXs> &xs,
                                        const std::vector<VectorXs> &us) {
  if (xs.size() != T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "xs has wrong dimension (it should be " +
                        std::to_string(T_ + 1) + ")");
  }
  if (us.size() != T_) {
    throw_pretty("Invalid argument: "
                 << "us has wrong dimension (it should be " +
                        std::to_string(T_) + ")");
  }

#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for
#endif
  for (std::size_t i = 0; i < T_; ++i) {
    const std::size_t &nu = running_models_[i]->get_nu();
    if (nu != 0) {
      running_models_[i]->calc(running_datas_[i], xs[i], us[i].head(nu));
    } else {
      running_models_[i]->calc(running_datas_[i], xs[i]);
    }
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
Scalar ShootingProblemTpl<Scalar>::calcDiff(const std::vector<VectorXs> &xs,
                                            const std::vector<VectorXs> &us) {
  if (xs.size() != T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "xs has wrong dimension (it should be " +
                        std::to_string(T_ + 1) + ")");
  }
  if (us.size() != T_) {
    throw_pretty("Invalid argument: "
                 << "us has wrong dimension (it should be " +
                        std::to_string(T_) + ")");
  }

#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for
#endif
  for (std::size_t i = 0; i < T_; ++i) {
    if (running_models_[i]->get_nu() != 0) {
      const std::size_t &nu = running_models_[i]->get_nu();
      running_models_[i]->calcDiff(running_datas_[i], xs[i], us[i].head(nu));
    } else {
      running_models_[i]->calcDiff(running_datas_[i], xs[i]);
    }
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
void ShootingProblemTpl<Scalar>::rollout(const std::vector<VectorXs> &us,
                                         std::vector<VectorXs> &xs) {
  if (xs.size() != T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "xs has wrong dimension (it should be " +
                        std::to_string(T_ + 1) + ")");
  }
  if (us.size() != T_) {
    throw_pretty("Invalid argument: "
                 << "us has wrong dimension (it should be " +
                        std::to_string(T_) + ")");
  }

  xs[0] = x0_;
  for (std::size_t i = 0; i < T_; ++i) {
    const boost::shared_ptr<ActionModelAbstract> &model = running_models_[i];
    const boost::shared_ptr<ActionDataAbstract> &data = running_datas_[i];
    const VectorXs &x = xs[i];
    const std::size_t &nu = running_models_[i]->get_nu();
    if (model->get_nu() != 0) {
      const VectorXs &u = us[i];
      model->calc(data, x, u.head(nu));
    } else {
      model->calc(data, x);
    }
    xs[i + 1] = data->xnext;
  }
  terminal_model_->calc(terminal_data_, xs.back());
}

template <typename Scalar>
std::vector<typename MathBaseTpl<Scalar>::VectorXs>
ShootingProblemTpl<Scalar>::rollout_us(const std::vector<VectorXs> &us) {
  std::vector<VectorXs> xs;
  xs.resize(T_ + 1);
  rollout(us, xs);
  return xs;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::quasiStatic(std::vector<VectorXs> &us,
                                             const std::vector<VectorXs> &xs) {
  if (xs.size() != T_) {
    throw_pretty("Invalid argument: "
                 << "xs has wrong dimension (it should be " +
                        std::to_string(T_) + ")");
  }
  if (us.size() != T_) {
    throw_pretty("Invalid argument: "
                 << "us has wrong dimension (it should be " +
                        std::to_string(T_) + ")");
  }

#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for
#endif
  for (std::size_t i = 0; i < T_; ++i) {
    const std::size_t &nu = running_models_[i]->get_nu();
    running_models_[i]->quasiStatic(running_datas_[i], us[i].head(nu), xs[i]);
  }
}

template <typename Scalar>
std::vector<typename MathBaseTpl<Scalar>::VectorXs>
ShootingProblemTpl<Scalar>::quasiStatic_xs(const std::vector<VectorXs> &xs) {
  std::vector<VectorXs> us;
  us.resize(T_);
  for (std::size_t i = 0; i < T_; ++i) {
    us[i] = VectorXs::Zero(running_models_[i]->get_nu());
  }
  quasiStatic(us, xs);
  return us;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::circularAppend(
    boost::shared_ptr<ActionModelAbstract> model,
    boost::shared_ptr<ActionDataAbstract> data) {
  if (!model->checkData(data)) {
    throw_pretty("Invalid argument: "
                 << "action data is not consistent with the action model")
  }
  if (model->get_state()->get_nx() != nx_) {
    throw_pretty("Invalid argument: "
                 << "nx is not consistent with the other nodes")
  }
  if (model->get_state()->get_ndx() != ndx_) {
    throw_pretty("Invalid argument: "
                 << "ndx node is not consistent with the other nodes")
  }
  if (model->get_nu() > nu_max_) {
    throw_pretty("Invalid argument: "
                 << "nu node is greater than the maximum nu")
  }

  for (std::size_t i = 0; i < T_ - 1; ++i) {
    running_models_[i] = running_models_[i + 1];
    running_datas_[i] = running_datas_[i + 1];
  }
  running_models_.back() = model;
  running_datas_.back() = data;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::circularAppend(
    boost::shared_ptr<ActionModelAbstract> model) {
  if (model->get_state()->get_nx() != nx_) {
    throw_pretty("Invalid argument: "
                 << "nx is not consistent with the other nodes")
  }
  if (model->get_state()->get_ndx() != ndx_) {
    throw_pretty("Invalid argument: "
                 << "ndx node is not consistent with the other nodes")
  }
  if (model->get_nu() > nu_max_) {
    throw_pretty("Invalid argument: "
                 << "nu node is greater than the maximum nu")
  }

  for (std::size_t i = 0; i < T_ - 1; ++i) {
    running_models_[i] = running_models_[i + 1];
    running_datas_[i] = running_datas_[i + 1];
  }
  running_models_.back() = model;
  running_datas_.back() = model->createData();
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::updateNode(
    std::size_t i, boost::shared_ptr<ActionModelAbstract> model,
    boost::shared_ptr<ActionDataAbstract> data) {
  if (i >= T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "i is bigger than the allocated horizon (it should be less "
                    "than or equal to " +
                        std::to_string(T_ + 1) + ")");
  }
  if (!model->checkData(data)) {
    throw_pretty("Invalid argument: "
                 << "action data is not consistent with the action model")
  }
  if (model->get_state()->get_nx() != nx_) {
    throw_pretty("Invalid argument: "
                 << "nx is not consistent with the other nodes")
  }
  if (model->get_state()->get_ndx() != ndx_) {
    throw_pretty("Invalid argument: "
                 << "ndx node is not consistent with the other nodes")
  }
  if (model->get_nu() > nu_max_) {
    throw_pretty("Invalid argument: "
                 << "nu node is greater than the maximum nu")
  }

  if (i == T_) {
    terminal_model_ = model;
    terminal_data_ = data;
  } else {
    running_models_[i] = model;
    running_datas_[i] = data;
  }
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::updateModel(
    std::size_t i, boost::shared_ptr<ActionModelAbstract> model) {
  if (i >= T_ + 1) {
    throw_pretty(
        "Invalid argument: "
        << "i is bigger than the allocated horizon (it should be lower than " +
               std::to_string(T_ + 1) + ")");
  }
  if (model->get_state()->get_nx() != nx_) {
    throw_pretty("Invalid argument: "
                 << "nx is not consistent with the other nodes")
  }
  if (model->get_state()->get_ndx() != ndx_) {
    throw_pretty("Invalid argument: "
                 << "ndx is not consistent with the other nodes")
  }
  if (model->get_nu() > nu_max_) {
    throw_pretty("Invalid argument: "
                 << "nu node is greater than the maximum nu")
  }

  if (i == T_) {
    terminal_model_ = model;
    terminal_data_ = terminal_model_->createData();
  } else {
    running_models_[i] = model;
    running_datas_[i] = model->createData();
  }
}

template <typename Scalar>
const std::size_t &ShootingProblemTpl<Scalar>::get_T() const {
  return T_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs &
ShootingProblemTpl<Scalar>::get_x0() const {
  return x0_;
}

template <typename Scalar> void ShootingProblemTpl<Scalar>::allocateData() {
  for (std::size_t i = 0; i < T_; ++i) {
    const boost::shared_ptr<ActionModelAbstract> &model = running_models_[i];
    running_datas_.push_back(model->createData());
  }
  terminal_data_ = terminal_model_->createData();
}

template <typename Scalar>
const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>>>
    &ShootingProblemTpl<Scalar>::get_runningModels() const {
  return running_models_;
}

template <typename Scalar>
const boost::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>> &
ShootingProblemTpl<Scalar>::get_terminalModel() const {
  return terminal_model_;
}

template <typename Scalar>
const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>> &
ShootingProblemTpl<Scalar>::get_runningDatas() const {
  return running_datas_;
}

template <typename Scalar>
const boost::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> &
ShootingProblemTpl<Scalar>::get_terminalData() const {
  return terminal_data_;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::set_x0(const VectorXs &x0_in) {
  if (x0_in.size() != x0_.size()) {
    throw_pretty("Invalid argument: "
                 << "invalid size of x0 provided.");
  }
  x0_ = x0_in;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::set_runningModels(
    const std::vector<boost::shared_ptr<ActionModelAbstract>> &models) {
  for (std::size_t i = 0; i < T_; ++i) {
    const boost::shared_ptr<ActionModelAbstract> &model = running_models_[i];
    if (model->get_state()->get_nx() != nx_) {
      throw_pretty("Invalid argument: "
                   << "nx in " << i
                   << " node is not consistent with the other nodes")
    }
    if (model->get_state()->get_ndx() != ndx_) {
      throw_pretty("Invalid argument: "
                   << "ndx in " << i
                   << " node is not consistent with the other nodes")
    }
    if (model->get_nu() > nu_max_) {
      throw_pretty("Invalid argument: "
                   << "nu node is greater than the maximum nu")
    }
  }

  T_ = models.size();
  running_models_.clear();
  running_datas_.clear();
  for (std::size_t i = 0; i < T_; ++i) {
    const boost::shared_ptr<ActionModelAbstract> &model = running_models_[i];
    running_datas_.push_back(model->createData());
  }
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::set_terminalModel(
    boost::shared_ptr<ActionModelAbstract> model) {
  if (model->get_state()->get_nx() != nx_) {
    throw_pretty("Invalid argument: "
                 << "nx is not consistent with the other nodes")
  }
  if (model->get_state()->get_ndx() != ndx_) {
    throw_pretty("Invalid argument: "
                 << "ndx is not consistent with the other nodes")
  }
  terminal_model_ = model;
  terminal_data_ = terminal_model_->createData();
}

template <typename Scalar>
const std::size_t &ShootingProblemTpl<Scalar>::get_nx() const {
  return nx_;
}

template <typename Scalar>
const std::size_t &ShootingProblemTpl<Scalar>::get_ndx() const {
  return ndx_;
}

template <typename Scalar>
const std::size_t &ShootingProblemTpl<Scalar>::get_nu_max() const {
  return nu_max_;
}

} // namespace crocoddyl
