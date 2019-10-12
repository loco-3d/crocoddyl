///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"

namespace crocoddyl {

ImpulseModelMultiple::ImpulseModelMultiple(boost::shared_ptr<StateMultibody> state) : state_(state), ni_(0) {}

ImpulseModelMultiple::~ImpulseModelMultiple() {}

void ImpulseModelMultiple::addImpulse(const std::string& name, ImpulseModelAbstract* const impulse) {
  std::pair<ImpulseModelContainer::iterator, bool> ret =
      impulses_.insert(std::make_pair(name, ImpulseItem(name, impulse)));
  if (ret.second == false) {
    std::cout << "Warning: this impulse item already existed, we cannot add it" << std::endl;
  } else {
    ni_ += impulse->get_ni();
  }
}

void ImpulseModelMultiple::removeImpulse(const std::string& name) {
  ImpulseModelContainer::iterator it = impulses_.find(name);
  if (it != impulses_.end()) {
    ni_ -= it->second.impulse->get_ni();
    impulses_.erase(it);
  } else {
    std::cout << "Warning: this impulse item doesn't exist, we cannot remove it" << std::endl;
  }
}

void ImpulseModelMultiple::calc(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                const Eigen::Ref<const Eigen::VectorXd>& x) {
  assert(static_cast<std::size_t>(data->impulses.size()) == impulses_.size() &&
         "it doesn't match the number of impulse datas and models");
  std::size_t ni = 0;

  const std::size_t& nv = state_->get_nv();
  ImpulseModelContainer::iterator it_m, end_m;
  ImpulseDataContainer::iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(), it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ImpulseItem& m_i = it_m->second;
    boost::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
    assert(it_m->first == it_d->first && "it doesn't match the impulse name between data and model");

    m_i.impulse->calc(d_i, x);
    const std::size_t& ni_i = m_i.impulse->get_ni();
    data->Jc.block(ni, 0, ni_i, nv) = d_i->Jc;
    ni += ni_i;
  }
}

void ImpulseModelMultiple::calcDiff(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                    const Eigen::Ref<const Eigen::VectorXd>& x, const bool& recalc) {
  assert(static_cast<std::size_t>(data->impulses.size()) == impulses_.size() &&
         "it doesn't match the number of impulse datas and models");
  if (recalc) {
    calc(data, x);
  }
  std::size_t ni = 0;

  const std::size_t& nv = state_->get_nv();
  ImpulseModelContainer::iterator it_m, end_m;
  ImpulseDataContainer::iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(), it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ImpulseItem& m_i = it_m->second;
    boost::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
    assert(it_m->first == it_d->first && "it doesn't match the impulse name between data and model");

    m_i.impulse->calcDiff(d_i, x, false);
    const std::size_t& ni_i = m_i.impulse->get_ni();
    data->dv0_dq.block(ni, 0, ni_i, nv) = d_i->dv0_dq;
    ni += ni_i;
  }
}

void ImpulseModelMultiple::updateVelocity(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                          const Eigen::VectorXd& vnext) const {
  assert(static_cast<std::size_t>(vnext.rows()) == state_->get_nv() && "vnext has wrong dimension");

  data->vnext = vnext;
}

void ImpulseModelMultiple::updateForce(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                       const Eigen::VectorXd& force) {
  assert(static_cast<std::size_t>(force.size()) == ni_ && "force has wrong dimension, it should be ni vector");
  assert(static_cast<std::size_t>(data->impulses.size()) == impulses_.size() &&
         "it doesn't match the number of impulse datas and models");
  std::size_t ni = 0;

  for (ForceIterator it = data->fext.begin(); it != data->fext.end(); ++it) {
    *it = pinocchio::Force::Zero();
  }

  ImpulseModelContainer::iterator it_m, end_m;
  ImpulseDataContainer::iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(), it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ImpulseItem& m_i = it_m->second;
    boost::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
    assert(it_m->first == it_d->first && "it doesn't match the impulse name between data and model");

    const std::size_t& ni_i = m_i.impulse->get_ni();
    const Eigen::VectorBlock<const Eigen::VectorXd, Eigen::Dynamic> force_i = force.segment(ni, ni_i);
    m_i.impulse->updateForce(d_i, force_i);
    data->fext[d_i->joint] = d_i->f;
    ni += ni_i;
  }
}

void ImpulseModelMultiple::updateVelocityDiff(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                              const Eigen::MatrixXd& dvnext_dx) const {
  assert((static_cast<std::size_t>(dvnext_dx.rows()) == state_->get_nv() &&
          static_cast<std::size_t>(dvnext_dx.cols()) == state_->get_ndx()) &&
         "dvnext_dx has wrong dimension");

  data->dvnext_dx = dvnext_dx;
}

void ImpulseModelMultiple::updateForceDiff(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                           const Eigen::MatrixXd& df_dq) const {
  const std::size_t& nv = state_->get_nv();
  assert((static_cast<std::size_t>(df_dq.rows()) == ni_ && static_cast<std::size_t>(df_dq.cols()) == nv) &&
         "df_dq has wrong dimension");
  assert(data->impulses.size() == impulses_.size() && "it doesn't match the number of impulse datas and models");
  std::size_t ni = 0;

  ImpulseModelContainer::const_iterator it_m, end_m;
  ImpulseDataContainer::const_iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(), it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ImpulseItem& m_i = it_m->second;
    const boost::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
    assert(it_m->first == it_d->first && "it doesn't match the impulse name between data and model");

    const std::size_t& ni_i = m_i.impulse->get_ni();
    const Eigen::Block<const Eigen::MatrixXd> df_dq_i = df_dq.block(ni, 0, ni_i, nv);
    m_i.impulse->updateForceDiff(d_i, df_dq_i);
    ni += ni_i;
  }
}

boost::shared_ptr<ImpulseDataMultiple> ImpulseModelMultiple::createData(pinocchio::Data* const data) {
  return boost::make_shared<ImpulseDataMultiple>(this, data);
}

const boost::shared_ptr<StateMultibody>& ImpulseModelMultiple::get_state() const { return state_; }

const ImpulseModelMultiple::ImpulseModelContainer& ImpulseModelMultiple::get_impulses() const { return impulses_; }

const std::size_t& ImpulseModelMultiple::get_ni() const { return ni_; }

}  // namespace crocoddyl
