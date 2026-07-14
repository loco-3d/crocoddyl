///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ParameterManagerTpl<Scalar>::ParameterManagerTpl(
    std::shared_ptr<StateAbstract> state)
    : state_(state), np_(0), np_action_(0), np_dynamics_(0) {
  if (state_ == nullptr) {
    throw_pretty("Invalid argument: state is null");
  }
}

template <typename Scalar>
ParameterManagerTpl<Scalar>::ParameterManagerTpl(
    const ParameterManagerTpl& other)
    : state_(other.state_),
      np_(other.np_),
      np_action_(other.np_action_),
      np_dynamics_(other.np_dynamics_),
      active_set_(other.active_set_),
      inactive_set_(other.inactive_set_) {
  for (typename ParameterContainer::const_iterator it =
           other.action_params_.begin();
       it != other.action_params_.end(); ++it) {
    action_params_[it->first] = std::allocate_shared<ParameterItem>(
        Eigen::aligned_allocator<ParameterItem>(), *it->second);
  }
  for (typename ParameterContainer::const_iterator it =
           other.dynamics_params_.begin();
       it != other.dynamics_params_.end(); ++it) {
    dynamics_params_[it->first] = std::allocate_shared<ParameterItem>(
        Eigen::aligned_allocator<ParameterItem>(), *it->second);
  }
}

template <typename Scalar>
void ParameterManagerTpl<Scalar>::addParam(
    const std::string& name, std::shared_ptr<ActionModelParamsAbstract> param,
    const bool active) {
  addParamItem(name, param, &action_params_, active);
}

template <typename Scalar>
void ParameterManagerTpl<Scalar>::addParam(
    const std::string& name, std::shared_ptr<DynamicsParamsAbstract> param,
    const bool active) {
  addParamItem(name, param, &dynamics_params_, active);
}

template <typename Scalar>
void ParameterManagerTpl<Scalar>::addParamItem(
    const std::string& name, std::shared_ptr<ParamsAbstract> param,
    ParameterContainer* container, const bool active) {
  if (action_params_.find(name) != action_params_.end() ||
      dynamics_params_.find(name) != dynamics_params_.end()) {
    std::cout << "Warning: we couldn't add the " << name
              << " parameter item, it already existed." << std::endl;
    return;
  }
  if (param == nullptr) {
    throw_pretty("Invalid argument: parameter model is null");
  }
  if (param->get_state() == nullptr ||
      param->get_state()->get_nx() != state_->get_nx() ||
      param->get_state()->get_ndx() != state_->get_ndx() ||
      param->get_state()->get_nv() != state_->get_nv()) {
    throw_pretty(
        "Invalid argument: parameter state is not compatible with the "
        "manager state");
  }
  const std::shared_ptr<ParameterItem> item =
      std::allocate_shared<ParameterItem>(
          Eigen::aligned_allocator<ParameterItem>(), name, param, active);
  (*container)[name] = item;
  if (active) {
    updateDimensions(item, +1, container == &action_params_);
  }
  addToSets(name, active);
}

template <typename Scalar>
void ParameterManagerTpl<Scalar>::removeParam(const std::string& name) {
  typename ParameterContainer::iterator it = action_params_.find(name);
  if (it != action_params_.end()) {
    if (it->second->active_) {
      updateDimensions(it->second, -1, true);
    }
    active_set_.erase(name);
    inactive_set_.erase(name);
    action_params_.erase(it);
    return;
  }
  it = dynamics_params_.find(name);
  if (it != dynamics_params_.end()) {
    if (it->second->active_) {
      updateDimensions(it->second, -1, false);
    }
    active_set_.erase(name);
    inactive_set_.erase(name);
    dynamics_params_.erase(it);
    return;
  }
  std::cout << "Warning: we couldn't remove the " << name
            << " parameter item, it doesn't exist." << std::endl;
}

template <typename Scalar>
void ParameterManagerTpl<Scalar>::changeParamStatus(const std::string& name,
                                                    bool active) {
  typename ParameterContainer::iterator it = action_params_.find(name);
  if (it != action_params_.end()) {
    const std::shared_ptr<ParameterItem>& item = it->second;
    if (item->active_ != active) {
      updateDimensions(item, active ? +1 : -1, true);
      item->active_ = active;
      addToSets(name, active);
    }
    return;
  }
  it = dynamics_params_.find(name);
  if (it != dynamics_params_.end()) {
    const std::shared_ptr<ParameterItem>& item = it->second;
    if (item->active_ != active) {
      updateDimensions(item, active ? +1 : -1, false);
      item->active_ = active;
      addToSets(name, active);
    }
    return;
  }
  std::cout << "Warning: we couldn't change the status of the " << name
            << " parameter item, it doesn't exist." << std::endl;
}

template <typename Scalar>
bool ParameterManagerTpl<Scalar>::getParamStatus(
    const std::string& name) const {
  typename ParameterContainer::const_iterator it = action_params_.find(name);
  if (it != action_params_.end()) {
    return it->second->active_;
  }
  it = dynamics_params_.find(name);
  if (it != dynamics_params_.end()) {
    return it->second->active_;
  }
  std::cout << "Warning: we couldn't get the status of the " << name
            << " parameter item, it doesn't exist." << std::endl;
  return false;
}

template <typename Scalar>
void ParameterManagerTpl<Scalar>::update(
    const std::shared_ptr<ParameterDataManager>& data,
    const Eigen::Ref<const VectorXs>& p) const {
  assertDataIsConsistent(data);
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty("Invalid argument: p has wrong dimension (it should be "
                 << np_ << ")");
  }
  data->params->p = p;

  std::size_t offset = 0;
  typename ParameterContainer::const_iterator model_it = action_params_.begin();
  typename ParameterDataManager::ParameterDataContainer::const_iterator
      data_it = data->action_params.begin();
  for (; model_it != action_params_.end(); ++model_it, ++data_it) {
    const std::shared_ptr<ParameterItem>& item = model_it->second;
    const std::size_t np = assertItemDataIsConsistent(
        model_it->first, item, data_it->first, data_it->second, true);
    if (item->active_) {
      item->param_->update(data_it->second, p.segment(offset, np));
      offset += np;
    }
  }
  model_it = dynamics_params_.begin();
  data_it = data->dynamics_params.begin();
  for (; model_it != dynamics_params_.end(); ++model_it, ++data_it) {
    const std::shared_ptr<ParameterItem>& item = model_it->second;
    const std::size_t np = assertItemDataIsConsistent(
        model_it->first, item, data_it->first, data_it->second, false);
    if (item->active_) {
      item->param_->update(data_it->second, p.segment(offset, np));
      offset += np;
    }
  }
}

template <typename Scalar>
void ParameterManagerTpl<Scalar>::calcDiff_action(
    const std::shared_ptr<ParameterDataManager>& data,
    const std::shared_ptr<ActionDataAbstract>& action_data,
    const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) const {
  assertDataIsConsistent(data);
  if (action_data == nullptr) {
    throw_pretty("Invalid argument: action data is null");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be "
                 << state_->get_nx() << ")");
  }
  if (u.size() != action_data->Fu.cols()) {
    throw_pretty("Invalid argument: u has wrong dimension (it should be "
                 << action_data->Fu.cols() << ")");
  }
  data->params->dx_dp.setZero();

  std::size_t offset = 0;
  typename ParameterContainer::const_iterator model_it = action_params_.begin();
  typename ParameterDataManager::ParameterDataContainer::const_iterator
      data_it = data->action_params.begin();
  for (; model_it != action_params_.end(); ++model_it, ++data_it) {
    const std::shared_ptr<ParameterItem>& item = model_it->second;
    const std::size_t np = assertItemDataIsConsistent(
        model_it->first, item, data_it->first, data_it->second, true);
    if (item->active_) {
      const std::shared_ptr<ActionModelParamsAbstract> param =
          std::static_pointer_cast<ActionModelParamsAbstract>(item->param_);
      param->computeParamSensitivity(action_data, data_it->second, x, u);
      data->params->dx_dp.middleCols(offset, np) = data_it->second->dx_dp;
      offset += np;
    }
  }
}

template <typename Scalar>
void ParameterManagerTpl<Scalar>::calcDiff_dynamics(
    const std::shared_ptr<ParameterDataManager>& data,
    const std::shared_ptr<DynamicsDataAbstract>& dynamics_data,
    const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) const {
  assertDataIsConsistent(data);
  if (dynamics_data == nullptr) {
    throw_pretty("Invalid argument: dynamics data is null");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: x has wrong dimension (it should be "
                 << state_->get_nx() << ")");
  }
  if (u.size() != dynamics_data->Fu.cols()) {
    throw_pretty("Invalid argument: u has wrong dimension (it should be "
                 << dynamics_data->Fu.cols() << ")");
  }
  data->params->dtau_dp.setZero();

  std::size_t offset = 0;
  typename ParameterContainer::const_iterator model_it =
      dynamics_params_.begin();
  typename ParameterDataManager::ParameterDataContainer::const_iterator
      data_it = data->dynamics_params.begin();
  for (; model_it != dynamics_params_.end(); ++model_it, ++data_it) {
    const std::shared_ptr<ParameterItem>& item = model_it->second;
    const std::size_t np = assertItemDataIsConsistent(
        model_it->first, item, data_it->first, data_it->second, false);
    if (item->active_) {
      const std::shared_ptr<DynamicsParamsAbstract> param =
          std::static_pointer_cast<DynamicsParamsAbstract>(item->param_);
      param->computeJointTorqueRegressor(dynamics_data, data_it->second, x, u);
      data->params->dtau_dp.middleCols(offset, np) = data_it->second->dtau_dp;
      offset += np;
    }
  }
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs ParameterManagerTpl<Scalar>::zero()
    const {
  VectorXs p(np_);
  std::size_t offset = 0;
  for (typename ParameterContainer::const_iterator it = action_params_.begin();
       it != action_params_.end(); ++it) {
    if (it->second->active_) {
      const std::size_t np = it->second->param_->get_np();
      p.segment(offset, np) = it->second->param_->zero();
      offset += np;
    }
  }
  for (typename ParameterContainer::const_iterator it =
           dynamics_params_.begin();
       it != dynamics_params_.end(); ++it) {
    if (it->second->active_) {
      const std::size_t np = it->second->param_->get_np();
      p.segment(offset, np) = it->second->param_->zero();
      offset += np;
    }
  }
  return p;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs ParameterManagerTpl<Scalar>::rand()
    const {
  VectorXs p(np_);
  std::size_t offset = 0;
  for (typename ParameterContainer::const_iterator it = action_params_.begin();
       it != action_params_.end(); ++it) {
    if (it->second->active_) {
      const std::size_t np = it->second->param_->get_np();
      p.segment(offset, np) = it->second->param_->rand();
      offset += np;
    }
  }
  for (typename ParameterContainer::const_iterator it =
           dynamics_params_.begin();
       it != dynamics_params_.end(); ++it) {
    if (it->second->active_) {
      const std::size_t np = it->second->param_->get_np();
      p.segment(offset, np) = it->second->param_->rand();
      offset += np;
    }
  }
  return p;
}

template <typename Scalar>
std::shared_ptr<ParameterDataManagerTpl<Scalar> >
ParameterManagerTpl<Scalar>::createData() const {
  return std::allocate_shared<ParameterDataManager>(
      Eigen::aligned_allocator<ParameterDataManager>(), this);
}

template <typename Scalar>
template <typename NewScalar>
ParameterManagerTpl<NewScalar> ParameterManagerTpl<Scalar>::cast() const {
  typedef ParameterManagerTpl<NewScalar> ReturnType;
  typedef ActionModelParamsAbstractTpl<NewScalar> ActionModelParamsNew;
  typedef DynamicsParamsAbstractTpl<NewScalar> DynamicsParamsNew;

  ReturnType ret(state_->template cast<NewScalar>());
  for (typename ParameterContainer::const_iterator it = action_params_.begin();
       it != action_params_.end(); ++it) {
    const std::shared_ptr<ActionModelParamsNew> param =
        std::dynamic_pointer_cast<ActionModelParamsNew>(
            it->second->param_->template cast<NewScalar>());
    if (param == nullptr) {
      throw_pretty("Invalid call: parameter '"
                   << it->first
                   << "' is not an action parameter after casting");
    }
    ret.addParam(it->first, param, it->second->active_);
  }
  for (typename ParameterContainer::const_iterator it =
           dynamics_params_.begin();
       it != dynamics_params_.end(); ++it) {
    const std::shared_ptr<DynamicsParamsNew> param =
        std::dynamic_pointer_cast<DynamicsParamsNew>(
            it->second->param_->template cast<NewScalar>());
    if (param == nullptr) {
      throw_pretty("Invalid call: parameter '"
                   << it->first
                   << "' is not a dynamics parameter after casting");
    }
    ret.addParam(it->first, param, it->second->active_);
  }
  return ret;
}

template <typename Scalar>
const std::shared_ptr<StateAbstractTpl<Scalar> >&
ParameterManagerTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
std::size_t ParameterManagerTpl<Scalar>::get_np() const {
  return np_;
}

template <typename Scalar>
std::size_t ParameterManagerTpl<Scalar>::get_np_action() const {
  return np_action_;
}

template <typename Scalar>
std::size_t ParameterManagerTpl<Scalar>::get_np_dynamics() const {
  return np_dynamics_;
}

template <typename Scalar>
const typename ParameterManagerTpl<Scalar>::ParameterContainer&
ParameterManagerTpl<Scalar>::get_action_params() const {
  return action_params_;
}

template <typename Scalar>
const typename ParameterManagerTpl<Scalar>::ParameterContainer&
ParameterManagerTpl<Scalar>::get_dynamics_params() const {
  return dynamics_params_;
}

template <typename Scalar>
const typename ParameterManagerTpl<Scalar>::NameSet&
ParameterManagerTpl<Scalar>::get_active_set() const {
  return active_set_;
}

template <typename Scalar>
const typename ParameterManagerTpl<Scalar>::NameSet&
ParameterManagerTpl<Scalar>::get_inactive_set() const {
  return inactive_set_;
}

template <typename Scalar>
void ParameterManagerTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os,
                         const ParameterManagerTpl<Scalar>& model) {
  model.print(os);
  return os;
}

template <typename Scalar>
void ParameterManagerTpl<Scalar>::addToSets(const std::string& name,
                                            bool active) {
  if (active) {
    inactive_set_.erase(name);
    active_set_.insert(name);
  } else {
    active_set_.erase(name);
    inactive_set_.insert(name);
  }
}

template <typename Scalar>
void ParameterManagerTpl<Scalar>::updateDimensions(
    const std::shared_ptr<ParameterItem>& item, const int delta,
    const bool action) {
  const std::size_t np = item->param_->get_np();
  if (delta > 0) {
    np_ += np;
    if (action) {
      np_action_ += np;
    } else {
      np_dynamics_ += np;
    }
  } else if (delta < 0) {
    np_ -= np;
    if (action) {
      np_action_ -= np;
    } else {
      np_dynamics_ -= np;
    }
  }
}

template <typename Scalar>
void ParameterManagerTpl<Scalar>::assertDataIsConsistent(
    const std::shared_ptr<ParameterDataManager>& data) const {
  if (data == nullptr || data->params == nullptr) {
    throw_pretty("Invalid argument: parameter data is null");
  }
  if (data->parameter_data != data.get()) {
    throw_pretty("Invalid argument: parameter data has an invalid self-link");
  }
  if (data->params->np_action != np_action_ ||
      data->params->np_dynamics != np_dynamics_ || data->params->np != np_ ||
      static_cast<std::size_t>(data->params->p.size()) != np_ ||
      static_cast<std::size_t>(data->params->dx_dp.rows()) !=
          state_->get_ndx() ||
      static_cast<std::size_t>(data->params->dx_dp.cols()) != np_action_ ||
      static_cast<std::size_t>(data->params->dtau_dp.rows()) !=
          state_->get_nv() ||
      static_cast<std::size_t>(data->params->dtau_dp.cols()) != np_dynamics_) {
    throw_pretty(
        "Invalid argument: parameter data dimensions are stale. Resize after "
        "status changes or recreate after adding/removing models");
  }
  if (data->action_params.size() != action_params_.size() ||
      data->dynamics_params.size() != dynamics_params_.size()) {
    throw_pretty(
        "Invalid argument: parameter data names are stale. Recreate data "
        "after adding or removing models");
  }
}

template <typename Scalar>
std::size_t ParameterManagerTpl<Scalar>::assertItemDataIsConsistent(
    const std::string& name, const std::shared_ptr<ParameterItem>& item,
    const std::string& data_name,
    const std::shared_ptr<ParamsDataAbstract>& item_data,
    const bool action) const {
  const std::size_t np = item->param_->get_np();
  const std::size_t np_action = action ? np : 0;
  const std::size_t np_dynamics = action ? 0 : np;
  if (name != data_name || item_data == nullptr ||
      !item->param_->checkData(item_data) ||
      item_data->np_action != np_action ||
      item_data->np_dynamics != np_dynamics ||
      static_cast<std::size_t>(item_data->p.size()) != np ||
      static_cast<std::size_t>(item_data->dx_dp.rows()) != state_->get_ndx() ||
      static_cast<std::size_t>(item_data->dx_dp.cols()) != np_action ||
      static_cast<std::size_t>(item_data->dtau_dp.rows()) != state_->get_nv() ||
      static_cast<std::size_t>(item_data->dtau_dp.cols()) != np_dynamics) {
    throw_pretty("Invalid argument: " << (action ? "action" : "dynamics")
                                      << " parameter data for '" << name
                                      << "' is inconsistent");
  }
  return np;
}

template <typename Scalar>
ParameterDataManagerTpl<Scalar>::ParameterDataManagerTpl(
    const ParameterManager* const model)
    : Base(std::shared_ptr<ParamsDataAbstract>(), this) {
  if (model == nullptr) {
    throw_pretty("Invalid argument: parameter manager is null");
  }
  this->params = std::allocate_shared<ParamsDataAbstract>(
      Eigen::aligned_allocator<ParamsDataAbstract>(), model->get_state(),
      model->get_np_action(), model->get_np_dynamics());

  const typename ParameterManager::ParameterContainer& action_models =
      model->get_action_params();
  for (typename ParameterManager::ParameterContainer::const_iterator it =
           action_models.begin();
       it != action_models.end(); ++it) {
    const std::shared_ptr<ParamsDataAbstract> item_data =
        it->second->get_param()->createData();
    const std::size_t np = it->second->get_param()->get_np();
    if (item_data == nullptr ||
        !it->second->get_param()->checkData(item_data) ||
        item_data->np_action != np || item_data->np_dynamics != 0 ||
        static_cast<std::size_t>(item_data->p.size()) != np ||
        static_cast<std::size_t>(item_data->dx_dp.rows()) !=
            model->get_state()->get_ndx() ||
        static_cast<std::size_t>(item_data->dx_dp.cols()) != np ||
        static_cast<std::size_t>(item_data->dtau_dp.rows()) !=
            model->get_state()->get_nv() ||
        item_data->dtau_dp.cols() != 0) {
      throw_pretty("Invalid argument: action parameter model '"
                   << it->first << "' created inconsistent data");
    }
    action_params[it->first] = item_data;
    active_offsets_[it->first] = std::numeric_limits<std::size_t>::max();
  }
  const typename ParameterManager::ParameterContainer& dynamics_models =
      model->get_dynamics_params();
  for (typename ParameterManager::ParameterContainer::const_iterator it =
           dynamics_models.begin();
       it != dynamics_models.end(); ++it) {
    const std::shared_ptr<ParamsDataAbstract> item_data =
        it->second->get_param()->createData();
    const std::size_t np = it->second->get_param()->get_np();
    if (item_data == nullptr ||
        !it->second->get_param()->checkData(item_data) ||
        item_data->np_action != 0 || item_data->np_dynamics != np ||
        static_cast<std::size_t>(item_data->p.size()) != np ||
        static_cast<std::size_t>(item_data->dx_dp.rows()) !=
            model->get_state()->get_ndx() ||
        item_data->dx_dp.cols() != 0 ||
        static_cast<std::size_t>(item_data->dtau_dp.rows()) !=
            model->get_state()->get_nv() ||
        static_cast<std::size_t>(item_data->dtau_dp.cols()) != np) {
      throw_pretty("Invalid argument: dynamics parameter model '"
                   << it->first << "' created inconsistent data");
    }
    dynamics_params[it->first] = item_data;
    active_offsets_[it->first] = std::numeric_limits<std::size_t>::max();
  }
  refreshActiveLayout(model);
}

template <typename Scalar>
ParameterDataManagerTpl<Scalar>::ParameterDataManagerTpl(
    const ParameterDataManagerTpl& other)
    : Base(other.params, this),
      action_params(other.action_params),
      dynamics_params(other.dynamics_params),
      active_offsets_(other.active_offsets_) {}

template <typename Scalar>
void ParameterDataManagerTpl<Scalar>::resize(
    const ParameterManager* const model) {
  if (model == nullptr) {
    throw_pretty("Invalid argument: parameter manager is null");
  }
  if (this->params == nullptr) {
    throw_pretty("Invalid argument: aggregate parameter data is null");
  }
  if (this->parameter_data != this) {
    throw_pretty("Invalid argument: parameter data has an invalid self-link");
  }
  const typename ParameterManager::ParameterContainer& action_models =
      model->get_action_params();
  const typename ParameterManager::ParameterContainer& dynamics_models =
      model->get_dynamics_params();
  if (action_params.size() != action_models.size() ||
      dynamics_params.size() != dynamics_models.size()) {
    throw_pretty(
        "Invalid argument: parameter data names are stale. Recreate data "
        "after adding or removing models");
  }
  typename ParameterManager::ParameterContainer::const_iterator model_it =
      action_models.begin();
  typename ParameterDataContainer::const_iterator data_it =
      action_params.begin();
  for (; model_it != action_models.end(); ++model_it, ++data_it) {
    const std::size_t np = model_it->second->get_param()->get_np();
    if (model_it->first != data_it->first || data_it->second == nullptr ||
        !model_it->second->get_param()->checkData(data_it->second) ||
        data_it->second->np_action != np || data_it->second->np_dynamics != 0 ||
        static_cast<std::size_t>(data_it->second->p.size()) != np ||
        static_cast<std::size_t>(data_it->second->dx_dp.rows()) !=
            model->get_state()->get_ndx() ||
        static_cast<std::size_t>(data_it->second->dx_dp.cols()) != np ||
        static_cast<std::size_t>(data_it->second->dtau_dp.rows()) !=
            model->get_state()->get_nv() ||
        data_it->second->dtau_dp.cols() != 0) {
      throw_pretty(
          "Invalid argument: parameter data names are stale. Recreate data "
          "after adding or removing models");
    }
  }
  model_it = dynamics_models.begin();
  data_it = dynamics_params.begin();
  for (; model_it != dynamics_models.end(); ++model_it, ++data_it) {
    const std::size_t np = model_it->second->get_param()->get_np();
    if (model_it->first != data_it->first || data_it->second == nullptr ||
        !model_it->second->get_param()->checkData(data_it->second) ||
        data_it->second->np_action != 0 || data_it->second->np_dynamics != np ||
        static_cast<std::size_t>(data_it->second->p.size()) != np ||
        static_cast<std::size_t>(data_it->second->dx_dp.rows()) !=
            model->get_state()->get_ndx() ||
        data_it->second->dx_dp.cols() != 0 ||
        static_cast<std::size_t>(data_it->second->dtau_dp.rows()) !=
            model->get_state()->get_nv() ||
        static_cast<std::size_t>(data_it->second->dtau_dp.cols()) != np) {
      throw_pretty(
          "Invalid argument: parameter data names are stale. Recreate data "
          "after adding or removing models");
    }
  }
  this->params->resize(model->get_np_action(), model->get_np_dynamics());
  refreshActiveLayout(model);
}

template <typename Scalar>
void ParameterDataManagerTpl<Scalar>::refreshActiveLayout(
    const ParameterManager* const model) {
  const typename ParameterManager::ParameterContainer& action_models =
      model->get_action_params();
  const typename ParameterManager::ParameterContainer& dynamics_models =
      model->get_dynamics_params();
  if (active_offsets_.size() != action_models.size() + dynamics_models.size()) {
    throw_pretty(
        "Invalid argument: parameter data names are stale. Recreate "
        "data after adding or removing models");
  }
  for (typename ParameterManager::ParameterContainer::const_iterator it =
           action_models.begin();
       it != action_models.end(); ++it) {
    if (active_offsets_.find(it->first) == active_offsets_.end()) {
      throw_pretty(
          "Invalid argument: parameter data names are stale. Recreate "
          "data after adding or removing models");
    }
  }
  for (typename ParameterManager::ParameterContainer::const_iterator it =
           dynamics_models.begin();
       it != dynamics_models.end(); ++it) {
    if (active_offsets_.find(it->first) == active_offsets_.end()) {
      throw_pretty(
          "Invalid argument: parameter data names are stale. Recreate "
          "data after adding or removing models");
    }
  }

  const std::size_t inactive = std::numeric_limits<std::size_t>::max();
  std::size_t offset = 0;
  for (typename ParameterManager::ParameterContainer::const_iterator it =
           action_models.begin();
       it != action_models.end(); ++it) {
    if (it->second->get_active()) {
      active_offsets_.find(it->first)->second = offset;
      offset += it->second->get_param()->get_np();
    } else {
      active_offsets_.find(it->first)->second = inactive;
    }
  }
  for (typename ParameterManager::ParameterContainer::const_iterator it =
           dynamics_models.begin();
       it != dynamics_models.end(); ++it) {
    if (it->second->get_active()) {
      active_offsets_.find(it->first)->second = offset;
      offset += it->second->get_param()->get_np();
    } else {
      active_offsets_.find(it->first)->second = inactive;
    }
  }
  if (offset != model->get_np()) {
    throw_pretty("Invalid argument: parameter data layout is inconsistent");
  }
}

template <typename Scalar>
void ParameterDataManagerTpl<Scalar>::setZero() {
  if (this->params == nullptr) {
    throw_pretty("Invalid argument: aggregate parameter data is null");
  }
  this->params->setZero();
  for (typename ParameterDataContainer::iterator it = action_params.begin();
       it != action_params.end(); ++it) {
    if (it->second == nullptr) {
      throw_pretty("Invalid argument: action parameter data for '"
                   << it->first << "' is null");
    }
    it->second->setZero();
  }
  for (typename ParameterDataContainer::iterator it = dynamics_params.begin();
       it != dynamics_params.end(); ++it) {
    if (it->second == nullptr) {
      throw_pretty("Invalid argument: dynamics parameter data for '"
                   << it->first << "' is null");
    }
    it->second->setZero();
  }
}

}  // namespace crocoddyl
