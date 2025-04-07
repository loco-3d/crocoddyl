///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ImpulseModelMultipleTpl<Scalar>::ImpulseModelMultipleTpl(
    std::shared_ptr<StateMultibody> state)
    : state_(state), nc_(0), nc_total_(0) {}

template <typename Scalar>
ImpulseModelMultipleTpl<Scalar>::~ImpulseModelMultipleTpl() {}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::addImpulse(
    const std::string& name, std::shared_ptr<ImpulseModelAbstract> impulse,
    const bool active) {
  std::pair<typename ImpulseModelContainer::iterator, bool> ret =
      impulses_.insert(std::make_pair(
          name, std::make_shared<ImpulseItem>(name, impulse, active)));
  if (ret.second == false) {
    std::cerr << "Warning: we couldn't add the " << name
              << " impulse item, it already existed." << std::endl;
  } else if (active) {
    nc_ += impulse->get_nc();
    nc_total_ += impulse->get_nc();
    active_set_.insert(name);
  } else if (!active) {
    nc_total_ += impulse->get_nc();
    inactive_set_.insert(name);
  }
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::removeImpulse(const std::string& name) {
  typename ImpulseModelContainer::iterator it = impulses_.find(name);
  if (it != impulses_.end()) {
    nc_ -= it->second->impulse->get_nc();
    nc_total_ -= it->second->impulse->get_nc();
    impulses_.erase(it);
    active_set_.erase(name);
    inactive_set_.erase(name);
  } else {
    std::cerr << "Warning: we couldn't remove the " << name
              << " impulse item, it doesn't exist." << std::endl;
  }
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::changeImpulseStatus(
    const std::string& name, const bool active) {
  typename ImpulseModelContainer::iterator it = impulses_.find(name);
  if (it != impulses_.end()) {
    if (active && !it->second->active) {
      nc_ += it->second->impulse->get_nc();
      active_set_.insert(name);
      inactive_set_.erase(name);
    } else if (!active && it->second->active) {
      nc_ -= it->second->impulse->get_nc();
      inactive_set_.insert(name);
      active_set_.erase(name);
    }
    it->second->active = active;
  } else {
    std::cerr << "Warning: we couldn't change the status of the " << name
              << " impulse item, it doesn't exist." << std::endl;
  }
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::calc(
    const std::shared_ptr<ImpulseDataMultiple>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (data->impulses.size() != impulses_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of impulse datas and models");
  }

  std::size_t nc = 0;
  const std::size_t nv = state_->get_nv();
  typename ImpulseModelContainer::iterator it_m, end_m;
  typename ImpulseDataContainer::iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(),
      it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<ImpulseItem>& m_i = it_m->second;
    if (m_i->active) {
      const std::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the impulse name between model and data ("
                        << it_m->first << " != " << it_d->first << ")");

      m_i->impulse->calc(d_i, x);
      const std::size_t nc_i = m_i->impulse->get_nc();
      data->Jc.block(nc, 0, nc_i, nv) = d_i->Jc;
      nc += nc_i;
    }
  }
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::calcDiff(
    const std::shared_ptr<ImpulseDataMultiple>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (data->impulses.size() != impulses_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of impulse datas and models");
  }

  std::size_t nc = 0;
  const std::size_t nv = state_->get_nv();
  typename ImpulseModelContainer::iterator it_m, end_m;
  typename ImpulseDataContainer::iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(),
      it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<ImpulseItem>& m_i = it_m->second;
    if (m_i->active) {
      const std::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first,
                    "it doesn't match the impulse name between model and data ("
                        << it_m->first << " != " << it_d->first << ")");

      m_i->impulse->calcDiff(d_i, x);
      const std::size_t nc_i = m_i->impulse->get_nc();
      data->dv0_dq.block(nc, 0, nc_i, nv) = d_i->dv0_dq;
      nc += nc_i;
    }
  }
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::updateVelocity(
    const std::shared_ptr<ImpulseDataMultiple>& data,
    const VectorXs& vnext) const {
  if (static_cast<std::size_t>(vnext.size()) != state_->get_nv()) {
    throw_pretty(
        "Invalid argument: " << "vnext has wrong dimension (it should be " +
                                    std::to_string(state_->get_nv()) + ")");
  }
  data->vnext = vnext;
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::updateForce(
    const std::shared_ptr<ImpulseDataMultiple>& data, const VectorXs& force) {
  if (static_cast<std::size_t>(force.size()) != nc_) {
    throw_pretty(
        "Invalid argument: " << "force has wrong dimension (it should be " +
                                    std::to_string(nc_) + ")");
  }
  if (static_cast<std::size_t>(data->impulses.size()) != impulses_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of impulse datas and models");
  }

  for (ForceIterator it = data->fext.begin(); it != data->fext.end(); ++it) {
    *it = pinocchio::ForceTpl<Scalar>::Zero();
  }

  std::size_t nc = 0;
  typename ImpulseModelContainer::iterator it_m, end_m;
  typename ImpulseDataContainer::iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(),
      it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<ImpulseItem>& m_i = it_m->second;
    const std::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
    assert_pretty(it_m->first == it_d->first,
                  "it doesn't match the impulse name between data and model");
    if (m_i->active) {
      const std::size_t nc_i = m_i->impulse->get_nc();
      const Eigen::VectorBlock<const VectorXs, Eigen::Dynamic> force_i =
          force.segment(nc, nc_i);
      m_i->impulse->updateForce(d_i, force_i);
      const pinocchio::JointIndex joint =
          state_->get_pinocchio()->frames[d_i->frame].parentJoint;
      data->fext[joint] = d_i->fext;
      nc += nc_i;
    } else {
      m_i->impulse->setZeroForce(d_i);
    }
  }
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::updateVelocityDiff(
    const std::shared_ptr<ImpulseDataMultiple>& data,
    const MatrixXs& dvnext_dx) const {
  if (static_cast<std::size_t>(dvnext_dx.rows()) != state_->get_nv() ||
      static_cast<std::size_t>(dvnext_dx.cols()) != state_->get_ndx()) {
    throw_pretty(
        "Invalid argument: " << "dvnext_dx has wrong dimension (it should be " +
                                    std::to_string(state_->get_nv()) + "," +
                                    std::to_string(state_->get_ndx()) + ")");
  }
  data->dvnext_dx = dvnext_dx;
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::updateForceDiff(
    const std::shared_ptr<ImpulseDataMultiple>& data,
    const MatrixXs& df_dx) const {
  const std::size_t ndx = state_->get_ndx();
  if (static_cast<std::size_t>(df_dx.rows()) != nc_ ||
      static_cast<std::size_t>(df_dx.cols()) != ndx) {
    throw_pretty("Invalid argument: "
                 << "df_dx has wrong dimension (it should be " +
                        std::to_string(nc_) + "," + std::to_string(ndx) + ")");
  }
  if (static_cast<std::size_t>(data->impulses.size()) != impulses_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of impulse datas and models");
  }

  std::size_t nc = 0;
  typename ImpulseModelContainer::const_iterator it_m, end_m;
  typename ImpulseDataContainer::const_iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(),
      it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<ImpulseItem>& m_i = it_m->second;
    const std::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
    assert_pretty(it_m->first == it_d->first,
                  "it doesn't match the impulse name between data and model");
    if (m_i->active) {
      const std::size_t nc_i = m_i->impulse->get_nc();
      const Eigen::Block<const MatrixXs> df_dx_i =
          df_dx.block(nc, 0, nc_i, ndx);
      m_i->impulse->updateForceDiff(d_i, df_dx_i);
      nc += nc_i;
    } else {
      m_i->impulse->setZeroForceDiff(d_i);
    }
  }
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::updateRneaDiff(
    const std::shared_ptr<ImpulseDataMultiple>& data,
    pinocchio::DataTpl<Scalar>& pinocchio) const {
  if (static_cast<std::size_t>(data->impulses.size()) !=
      this->get_impulses().size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of impulse datas and models");
  }
  typename ImpulseModelContainer::const_iterator it_m, end_m;
  typename ImpulseDataContainer::const_iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(),
      it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const std::shared_ptr<ImpulseItem>& m_i = it_m->second;
    const std::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
    assert_pretty(it_m->first == it_d->first,
                  "it doesn't match the impulse name between data and model");
    if (m_i->active) {
      switch (m_i->impulse->get_type()) {
        case pinocchio::ReferenceFrame::LOCAL:
          break;
        case pinocchio::ReferenceFrame::WORLD:
        case pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED:
          pinocchio.dtau_dq += d_i->dtau_dq;
          break;
      }
    }
  }
}

template <typename Scalar>
std::shared_ptr<ImpulseDataMultipleTpl<Scalar> >
ImpulseModelMultipleTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return std::allocate_shared<ImpulseDataMultiple>(
      Eigen::aligned_allocator<ImpulseDataMultiple>(), this, data);
}

template <typename Scalar>
template <typename NewScalar>
ImpulseModelMultipleTpl<NewScalar> ImpulseModelMultipleTpl<Scalar>::cast()
    const {
  typedef ImpulseModelMultipleTpl<NewScalar> ReturnType;
  typedef StateMultibodyTpl<NewScalar> StateType;
  typedef ImpulseItemTpl<NewScalar> ImpulseType;
  ReturnType ret(
      std::make_shared<StateType>(state_->template cast<NewScalar>()));
  typename ImpulseModelContainer::const_iterator it_m, end_m;
  for (it_m = impulses_.begin(), end_m = impulses_.end(); it_m != end_m;
       ++it_m) {
    const std::string name = it_m->first;
    const ImpulseType& m_i = it_m->second->template cast<NewScalar>();
    ret.addImpulse(name, m_i.impulse, m_i.active);
  }
  return ret;
}

template <typename Scalar>
const std::shared_ptr<StateMultibodyTpl<Scalar> >&
ImpulseModelMultipleTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const typename ImpulseModelMultipleTpl<Scalar>::ImpulseModelContainer&
ImpulseModelMultipleTpl<Scalar>::get_impulses() const {
  return impulses_;
}

template <typename Scalar>
std::size_t ImpulseModelMultipleTpl<Scalar>::get_nc() const {
  return nc_;
}

template <typename Scalar>
std::size_t ImpulseModelMultipleTpl<Scalar>::get_nc_total() const {
  return nc_total_;
}

template <typename Scalar>
const std::set<std::string>& ImpulseModelMultipleTpl<Scalar>::get_active_set()
    const {
  return active_set_;
}

template <typename Scalar>
const std::set<std::string>& ImpulseModelMultipleTpl<Scalar>::get_inactive_set()
    const {
  return inactive_set_;
}

template <typename Scalar>
bool ImpulseModelMultipleTpl<Scalar>::getImpulseStatus(
    const std::string& name) const {
  typename ImpulseModelContainer::const_iterator it = impulses_.find(name);
  if (it != impulses_.end()) {
    return it->second->active;
  } else {
    std::cerr << "Warning: we couldn't get the status of the " << name
              << " impulse item, it doesn't exist." << std::endl;
    return false;
  }
}

template <class Scalar>
std::ostream& operator<<(std::ostream& os,
                         const ImpulseModelMultipleTpl<Scalar>& model) {
  const auto& active = model.get_active_set();
  const auto& inactive = model.get_inactive_set();
  os << "ImpulseModelMultiple:" << std::endl;
  os << "  Active:" << std::endl;
  for (std::set<std::string>::const_iterator it = active.begin();
       it != active.end(); ++it) {
    const std::shared_ptr<
        typename ImpulseModelMultipleTpl<Scalar>::ImpulseItem>& impulse_item =
        model.get_impulses().find(*it)->second;
    if (it != --active.end()) {
      os << "    " << *it << ": " << *impulse_item << std::endl;
    } else {
      os << "    " << *it << ": " << *impulse_item << std::endl;
    }
  }
  os << "  Inactive:" << std::endl;
  for (std::set<std::string>::const_iterator it = inactive.begin();
       it != inactive.end(); ++it) {
    const std::shared_ptr<
        typename ImpulseModelMultipleTpl<Scalar>::ImpulseItem>& impulse_item =
        model.get_impulses().find(*it)->second;
    if (it != --inactive.end()) {
      os << "    " << *it << ": " << *impulse_item << std::endl;
    } else {
      os << "    " << *it << ": " << *impulse_item;
    }
  }
  return os;
}

}  // namespace crocoddyl
