///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"

namespace crocoddyl {

ImpulseModelMultiple::ImpulseModelMultiple(StateMultibody& state) : state_(state), ni_(0) {}

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
  assert(data->impulses.size() == impulses_.size() && "it doesn't match the number of impulse datas and models");
  unsigned int ni = 0;

  unsigned int const& nv = state_.get_nv();
  ImpulseModelContainer::iterator it_m, end_m;
  ImpulseDataContainer::iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(), it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ImpulseItem& m_i = it_m->second;
    boost::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
    assert(it_m->first == it_d->first && "it doesn't match the impulse name between data and model");

    m_i.impulse->calc(d_i, x);
    unsigned int const& ni_i = m_i.impulse->get_ni();
    data->Jc.block(ni, 0, ni_i, nv) = d_i->Jc;
    ni += ni_i;
  }
}

void ImpulseModelMultiple::calcDiff(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                    const Eigen::Ref<const Eigen::VectorXd>& x, const bool& recalc) {
  assert(data->impulses.size() == impulses_.size() && "it doesn't match the number of impulse datas and models");
  if (recalc) {
    calc(data, x);
  }
  unsigned int ni = 0;

  unsigned int const& nv = state_.get_nv();
  ImpulseModelContainer::iterator it_m, end_m;
  ImpulseDataContainer::iterator it_d, end_d;
  for (it_m = impulses_.begin(), end_m = impulses_.end(), it_d = data->impulses.begin(), end_d = data->impulses.end();
       it_m != end_m || it_d != end_d; ++it_m, ++it_d) {
    const ImpulseItem& m_i = it_m->second;
    boost::shared_ptr<ImpulseDataAbstract>& d_i = it_d->second;
    assert(it_m->first == it_d->first && "it doesn't match the impulse name between data and model");

    m_i.impulse->calcDiff(d_i, x, false);
    unsigned int const& ni_i = m_i.impulse->get_ni();
    data->Vq.block(ni, 0, ni_i, nv) = d_i->Vq;
    ni += ni_i;
  }
}

void ImpulseModelMultiple::updateLagrangian(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                            const Eigen::VectorXd& lambda) {
  assert(lambda.size() == ni_ && "lambda has wrong dimension, it should be ni vector");
  assert(data->impulses.size() == impulses_.size() && "it doesn't match the number of impulse datas and models");
  unsigned int ni = 0;

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

    unsigned int const& ni_i = m_i.impulse->get_ni();
    m_i.impulse->updateLagrangian(d_i, lambda.segment(ni, ni_i));
    data->fext[d_i->joint] = d_i->f;
    ni += ni_i;
  }
}

void ImpulseModelMultiple::updateImpulseVelocity(const boost::shared_ptr<ImpulseDataMultiple>& data,
                                                 const Eigen::VectorXd& vnext) const {
  data->vnext = vnext;
}

boost::shared_ptr<ImpulseDataMultiple> ImpulseModelMultiple::createData(pinocchio::Data* const data) {
  return boost::make_shared<ImpulseDataMultiple>(this, data);
}

StateMultibody& ImpulseModelMultiple::get_state() const { return state_; }

const ImpulseModelMultiple::ImpulseModelContainer& ImpulseModelMultiple::get_impulses() const { return impulses_; }

const unsigned int& ImpulseModelMultiple::get_ni() const { return ni_; }

}  // namespace crocoddyl
