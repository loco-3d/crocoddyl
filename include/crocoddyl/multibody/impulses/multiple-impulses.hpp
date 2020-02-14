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
  ImpulseItem(const std::string& name, boost::shared_ptr<ImpulseModelAbstract> impulse)
      : name(name), impulse(impulse) {}

  std::string name;
  boost::shared_ptr<ImpulseModelAbstract> impulse;
};

struct ImpulseDataMultiple;  // forward declaration

class ImpulseModelMultiple {
 public:
  typedef std::map<std::string, ImpulseItem> ImpulseModelContainer;
  typedef std::map<std::string, boost::shared_ptr<ImpulseDataAbstract> > ImpulseDataContainer;
  typedef pinocchio::container::aligned_vector<pinocchio::Force>::iterator ForceIterator;

  explicit ImpulseModelMultiple(boost::shared_ptr<StateMultibody> state);
  ~ImpulseModelMultiple();

  void addImpulse(const std::string& name, boost::shared_ptr<ImpulseModelAbstract> impulse);
  void removeImpulse(const std::string& name);

  void calc(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::Ref<const Eigen::VectorXd>& x);

  void updateVelocity(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::VectorXd& vnext) const;
  void updateForce(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::VectorXd& force);
  boost::shared_ptr<ImpulseDataMultiple> createData(pinocchio::Data* const data);
  void updateVelocityDiff(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::MatrixXd& dvnext_dx) const;
  void updateForceDiff(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::MatrixXd& df_dq) const;

  const boost::shared_ptr<StateMultibody>& get_state() const;
  const ImpulseModelContainer& get_impulses() const;
  const std::size_t& get_ni() const;

 private:
  boost::shared_ptr<StateMultibody> state_;
  ImpulseModelContainer impulses_;
  std::size_t ni_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::VectorXd& x) { calc(data, x); }

  void calcDiff_wrap(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::VectorXd& x) {
    calcDiff(data, x);
  }

#endif
};

struct ImpulseDataMultiple : ImpulseDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ImpulseDataMultiple(Model* const model, pinocchio::Data* const data)
      : ImpulseDataAbstract(model, data),
        vnext(model->get_state()->get_nv()),
        dvnext_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        fext(model->get_state()->get_pinocchio().njoints, pinocchio::Force::Zero()) {
    vnext.fill(0);
    dvnext_dx.fill(0);
    for (ImpulseModelMultiple::ImpulseModelContainer::const_iterator it = model->get_impulses().begin();
         it != model->get_impulses().end(); ++it) {
      const ImpulseItem& item = it->second;
      impulses.insert(std::make_pair(item.name, item.impulse->createData(data)));
    }
  }

  Eigen::VectorXd vnext;
  Eigen::MatrixXd dvnext_dx;
  ImpulseModelMultiple::ImpulseDataContainer impulses;
  pinocchio::container::aligned_vector<pinocchio::Force> fext;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_IMPULSES_MULTIPLE_IMPULSES_HPP_
