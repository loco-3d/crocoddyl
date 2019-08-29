///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_IMPULSES_MULTIPLE_IMPULSES_HPP_
#define CROCODDYL_MULTIBODY_IMPULSES_MULTIPLE_IMPULSES_HPP_

#include <string>
#include <map>
#include <utility>
#include "crocoddyl/multibody/impulse-base.hpp"

namespace crocoddyl {

struct ImpulseItem {
  ImpulseItem() {}
  ImpulseItem(const std::string& name, ImpulseModelAbstract* impulse) : name(name), impulse(impulse) {}

  std::string name;
  ImpulseModelAbstract* impulse;
};

struct ImpulseDataMultiple;  // forward declaration

class ImpulseModelMultiple {
 public:
  typedef std::map<std::string, ImpulseItem> ImpulseModelContainer;
  typedef std::map<std::string, boost::shared_ptr<ImpulseDataAbstract> > ImpulseDataContainer;
  typedef pinocchio::container::aligned_vector<pinocchio::Force>::iterator ForceIterator;

  ImpulseModelMultiple(StateMultibody& state);
  ~ImpulseModelMultiple();

  void addImpulse(const std::string& name, ImpulseModelAbstract* const impulse);
  void removeImpulse(const std::string& name);

  void calc(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const bool& recalc = true);
  void updateLagrangian(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::VectorXd& lambda);
  boost::shared_ptr<ImpulseDataMultiple> createData(pinocchio::Data* const data);

  StateMultibody& get_state() const;
  const ImpulseModelContainer& get_impulses() const;
  const unsigned int& get_ni() const;

 private:
  StateMultibody& state_;
  ImpulseModelContainer impulses_;
  unsigned int ni_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::VectorXd& x) { calc(data, x); }

  void calcDiff_wrap(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::VectorXd& x,
                     const bool& recalc = true) {
    calcDiff(data, x, recalc);
  }

#endif
};

struct ImpulseDataMultiple : ImpulseDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ImpulseDataMultiple(Model* const model, pinocchio::Data* const data)
      : ImpulseDataAbstract(model, data), fext(model->get_state().get_pinocchio().njoints, pinocchio::Force::Zero()) {
    for (ImpulseModelMultiple::ImpulseModelContainer::const_iterator it = model->get_impulses().begin();
         it != model->get_impulses().end(); ++it) {
      const ImpulseItem& item = it->second;
      impulses.insert(std::make_pair(item.name, item.impulse->createData(data)));
    }
  }

  ImpulseModelMultiple::ImpulseDataContainer impulses;
  pinocchio::container::aligned_vector<pinocchio::Force> fext;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_IMPULSES_MULTIPLE_IMPULSES_HPP_
