///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"

namespace crocoddyl {

template <typename Scalar>
ImpulseModelMultipleTpl<Scalar>::ImpulseModelMultipleTpl(boost::shared_ptr<StateMultibody> state)
    : state_(state), ni_(0), ni_total_(0) {}

template <typename Scalar>
ImpulseModelMultipleTpl<Scalar>::~ImpulseModelMultipleTpl() {}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::addImpulse(const std::string& name,
                                                 boost::shared_ptr<ImpulseModelAbstract> impulse, bool active) {
  std::pair<typename ImpulseModelContainer::iterator, bool> ret =
      impulses_.insert(std::make_pair(name, boost::make_shared<ImpulseItem>(name, impulse, active)));
  if (ret.second == false) {
    std::cout << "Warning: we couldn't add the " << name << " impulse item, it already existed." << std::endl;
  } else if (active) {
    ni_ += impulse->get_ni();
    ni_total_ += impulse->get_ni();
    std::vector<std::string>::iterator it =
        std::lower_bound(active_.begin(), active_.end(), name, std::less<std::string>());
    active_.insert(it, name);
  } else if (!active) {
    ni_total_ += impulse->get_ni();
    std::vector<std::string>::iterator it =
        std::lower_bound(inactive_.begin(), inactive_.end(), name, std::less<std::string>());
    inactive_.insert(it, name);
  }
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::removeImpulse(const std::string& name) {
  typename ImpulseModelContainer::iterator it = impulses_.find(name);
  if (it != impulses_.end()) {
    ni_ -= it->second->impulse->get_ni();
    ni_total_ -= it->second->impulse->get_ni();
    impulses_.erase(it);
    active_.erase(std::remove(active_.begin(), active_.end(), name), active_.end());
    inactive_.erase(std::remove(inactive_.begin(), inactive_.end(), name), inactive_.end());
  } else {
    std::cout << "Warning: we couldn't remove the " << name << " impulse item, it doesn't exist." << std::endl;
  }
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::changeImpulseStatus(const std::string& name, bool active) {
  typename ImpulseModelContainer::iterator it = impulses_.find(name);
  if (it != impulses_.end()) {
    if (active && !it->second->active) {
      ni_ += it->second->impulse->get_ni();
      std::vector<std::string>::iterator it =
          std::lower_bound(active_.begin(), active_.end(), name, std::less<std::string>());
      active_.insert(it, name);
      inactive_.erase(std::remove(inactive_.begin(), inactive_.end(), name), inactive_.end());
    } else if (!active && it->second->active) {
      ni_ -= it->second->impulse->get_ni();
      active_.erase(std::remove(active_.begin(), active_.end(), name), active_.end());
      std::vector<std::string>::iterator it =
          std::lower_bound(inactive_.begin(), inactive_.end(), name, std::less<std::string>());
      inactive_.insert(it, name);
    }
    it->second->active = active;
  } else {
    std::cout << "Warning: we couldn't change the status of the " << name << " impulse item, it doesn't exist."
              << std::endl;
  }
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::calc(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                           const Eigen::Ref<const VectorXs>& x) {
  if (data->impulses.size() != impulses_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of impulse datas and models");
  }

  std::size_t ni = 0;
  const std::size_t& nv = state_->get_nv();
  typename ImpulseModelContainer::iterator it_m, end_m;
  typename ImpulseDataContainer::iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(), it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const boost::shared_ptr<ImpulseItem>& m_i = it_m->second;
    if (m_i->active) {
      const boost::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first, "it doesn't match the impulse name between model and data ("
                                                    << it_m->first << " != " << it_d->first << ")");

      m_i->impulse->calc(d_i, x);
      const std::size_t& ni_i = m_i->impulse->get_ni();
      data->Jc.block(ni, 0, ni_i, nv) = d_i->Jc;
      ni += ni_i;
    }
  }
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::calcDiff(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                               const Eigen::Ref<const VectorXs>& x) {
  if (data->impulses.size() != impulses_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of impulse datas and models");
  }

  std::size_t ni = 0;
  const std::size_t& nv = state_->get_nv();
  typename ImpulseModelContainer::iterator it_m, end_m;
  typename ImpulseDataContainer::iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(), it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const boost::shared_ptr<ImpulseItem>& m_i = it_m->second;
    if (m_i->active) {
      const boost::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
      assert_pretty(it_m->first == it_d->first, "it doesn't match the impulse name between model and data ("
                                                    << it_m->first << " != " << it_d->first << ")");

      m_i->impulse->calcDiff(d_i, x);
      const std::size_t& ni_i = m_i->impulse->get_ni();
      data->dv0_dq.block(ni, 0, ni_i, nv) = d_i->dv0_dq;
      ni += ni_i;
    }
  }
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::updateVelocity(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                                     const VectorXs& vnext) const {
  if (static_cast<std::size_t>(vnext.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "vnext has wrong dimension (it should be " + std::to_string(state_->get_nv()) + ")");
  }
  data->vnext = vnext;
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::updateForce(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                                  const VectorXs& force) {
  if (static_cast<std::size_t>(force.size()) != ni_) {
    throw_pretty("Invalid argument: "
                 << "force has wrong dimension (it should be " + std::to_string(ni_) + ")");
  }
  if (static_cast<std::size_t>(data->impulses.size()) != impulses_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of impulse datas and models");
  }

  for (ForceIterator it = data->fext.begin(); it != data->fext.end(); ++it) {
    *it = pinocchio::ForceTpl<Scalar>::Zero();
  }

  std::size_t ni = 0;
  typename ImpulseModelContainer::iterator it_m, end_m;
  typename ImpulseDataContainer::iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(), it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const boost::shared_ptr<ImpulseItem>& m_i = it_m->second;
    const boost::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
    assert_pretty(it_m->first == it_d->first, "it doesn't match the impulse name between data and model");
    if (m_i->active) {
      const std::size_t& ni_i = m_i->impulse->get_ni();
      const Eigen::VectorBlock<const VectorXs, Eigen::Dynamic> force_i = force.segment(ni, ni_i);
      m_i->impulse->updateForce(d_i, force_i);
      data->fext[d_i->joint] = d_i->f;
      ni += ni_i;
    } else {
      m_i->impulse->setZeroForce(d_i);
    }
  }
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::updateVelocityDiff(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                                         const MatrixXs& dvnext_dx) const {
  if (static_cast<std::size_t>(dvnext_dx.rows()) != state_->get_nv() ||
      static_cast<std::size_t>(dvnext_dx.cols()) != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "dvnext_dx has wrong dimension (it should be " + std::to_string(state_->get_nv()) + "," +
                        std::to_string(state_->get_ndx()) + ")");
  }
  data->dvnext_dx = dvnext_dx;
}

template <typename Scalar>
void ImpulseModelMultipleTpl<Scalar>::updateForceDiff(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                                      const MatrixXs& df_dx) const {
  const std::size_t& ndx = state_->get_ndx();
  if (static_cast<std::size_t>(df_dx.rows()) != ni_ || static_cast<std::size_t>(df_dx.cols()) != ndx) {
    throw_pretty("Invalid argument: "
                 << "df_dx has wrong dimension (it should be " + std::to_string(ni_) + "," + std::to_string(ndx) +
                        ")");
  }
  if (static_cast<std::size_t>(data->impulses.size()) != impulses_.size()) {
    throw_pretty("Invalid argument: "
                 << "it doesn't match the number of impulse datas and models");
  }

  std::size_t ni = 0;
  typename ImpulseModelContainer::const_iterator it_m, end_m;
  typename ImpulseDataContainer::const_iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(), it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const boost::shared_ptr<ImpulseItem>& m_i = it_m->second;
    const boost::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
    assert_pretty(it_m->first == it_d->first, "it doesn't match the impulse name between data and model");
    if (m_i->active) {
      const std::size_t& ni_i = m_i->impulse->get_ni();
      const Eigen::Block<const MatrixXs> df_dx_i = df_dx.block(ni, 0, ni_i, ndx);
      m_i->impulse->updateForceDiff(d_i, df_dx_i);
      ni += ni_i;
    } else {
      m_i->impulse->setZeroForceDiff(d_i);
    }
  }
}

template <typename Scalar>
boost::shared_ptr<ImpulseDataMultipleTpl<Scalar> > ImpulseModelMultipleTpl<Scalar>::createData(
    pinocchio::DataTpl<Scalar>* const data) {
  return boost::allocate_shared<ImpulseDataMultiple>(Eigen::aligned_allocator<ImpulseDataMultiple>(), this, data);
}

template <typename Scalar>
const boost::shared_ptr<StateMultibodyTpl<Scalar> >& ImpulseModelMultipleTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const typename ImpulseModelMultipleTpl<Scalar>::ImpulseModelContainer& ImpulseModelMultipleTpl<Scalar>::get_impulses()
    const {
  return impulses_;
}

template <typename Scalar>
const std::size_t& ImpulseModelMultipleTpl<Scalar>::get_ni() const {
  return ni_;
}

template <typename Scalar>
const std::size_t& ImpulseModelMultipleTpl<Scalar>::get_ni_total() const {
  return ni_total_;
}

template <typename Scalar>
const std::vector<std::string>& ImpulseModelMultipleTpl<Scalar>::get_active() const {
  return active_;
}

template <typename Scalar>
const std::vector<std::string>& ImpulseModelMultipleTpl<Scalar>::get_inactive() const {
  return inactive_;
}

template <typename Scalar>
bool ImpulseModelMultipleTpl<Scalar>::getImpulseStatus(const std::string& name) const {
  typename ImpulseModelContainer::const_iterator it = impulses_.find(name);
  if (it != impulses_.end()) {
    return it->second->active;
  } else {
    std::cout << "Warning: we couldn't get the status of the " << name << " impulse item, it doesn't exist."
              << std::endl;
    return false;
  }
}

}  // namespace crocoddyl
