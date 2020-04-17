///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_IMPULSES_MULTIPLE_IMPULSES_HPP_
#define CROCODDYL_MULTIBODY_IMPULSES_MULTIPLE_IMPULSES_HPP_

#include <string>
#include <map>
#include <utility>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
struct ImpulseItemTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ImpulseModelAbstractTpl<Scalar> ImpulseModelAbstract;

  ImpulseItemTpl() {}
  ImpulseItemTpl(const std::string& name, boost::shared_ptr<ImpulseModelAbstract> impulse, bool active = true)
      : name(name), impulse(impulse), active(active) {}

  std::string name;
  boost::shared_ptr<ImpulseModelAbstract> impulse;
  bool active;
};

template <typename _Scalar>
class ImpulseModelMultipleTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ImpulseDataAbstractTpl<Scalar> ImpulseDataAbstract;
  typedef ImpulseDataMultipleTpl<Scalar> ImpulseDataMultiple;
  typedef ImpulseModelAbstractTpl<Scalar> ImpulseModelAbstract;

  typedef ImpulseItemTpl<Scalar> ImpulseItem;

  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  typedef std::map<std::string, boost::shared_ptr<ImpulseItem> > ImpulseModelContainer;
  typedef std::map<std::string, boost::shared_ptr<ImpulseDataAbstract> > ImpulseDataContainer;
  typedef typename pinocchio::container::aligned_vector<pinocchio::ForceTpl<Scalar> >::iterator ForceIterator;

  explicit ImpulseModelMultipleTpl(boost::shared_ptr<StateMultibody> state);
  ~ImpulseModelMultipleTpl();

  void addImpulse(const std::string& name, boost::shared_ptr<ImpulseModelAbstract> impulse, bool active = true);
  void removeImpulse(const std::string& name);
  void changeImpulseStatus(const std::string& name, bool active);

  void calc(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::Ref<const VectorXs>& x);
  void calcDiff(const boost::shared_ptr<ImpulseDataMultiple>& data, const Eigen::Ref<const VectorXs>& x);

  void updateVelocity(const boost::shared_ptr<ImpulseDataMultiple>& data, const VectorXs& vnext) const;
  void updateForce(const boost::shared_ptr<ImpulseDataMultiple>& data, const VectorXs& force);
  boost::shared_ptr<ImpulseDataMultiple> createData(pinocchio::DataTpl<Scalar>* const data);
  void updateVelocityDiff(const boost::shared_ptr<ImpulseDataMultiple>& data, const MatrixXs& dvnext_dx) const;
  void updateForceDiff(const boost::shared_ptr<ImpulseDataMultiple>& data, const MatrixXs& df_dq) const;

  const boost::shared_ptr<StateMultibody>& get_state() const;
  const ImpulseModelContainer& get_impulses() const;
  const std::size_t& get_ni() const;
  const std::size_t& get_ni_total() const;
  const std::vector<std::string>& get_active() const;
  bool getImpulseStatus(const std::string& name) const;

 private:
  boost::shared_ptr<StateMultibody> state_;
  ImpulseModelContainer impulses_;
  std::size_t ni_;
  std::size_t ni_total_;
  std::vector<std::string> active_;
};

template <typename _Scalar>
struct ImpulseDataMultipleTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ImpulseModelMultipleTpl<Scalar> ImpulseModelMultiple;
  typedef ImpulseItemTpl<Scalar> ImpulseItem;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ImpulseDataMultipleTpl(Model<Scalar>* const model, pinocchio::DataTpl<Scalar>* const data)
      : Jc(model->get_ni_total(), model->get_state()->get_nv()),
        dv0_dq(model->get_ni_total(), model->get_state()->get_nv()),
        vnext(model->get_state()->get_nv()),
        dvnext_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        fext(model->get_state()->get_pinocchio()->njoints, pinocchio::ForceTpl<Scalar>::Zero()) {
    Jc.setZero();
    dv0_dq.setZero();
    vnext.setZero();
    dvnext_dx.setZero();
    for (typename ImpulseModelMultiple::ImpulseModelContainer::const_iterator it = model->get_impulses().begin();
         it != model->get_impulses().end(); ++it) {
      const boost::shared_ptr<ImpulseItem>& item = it->second;
      impulses.insert(std::make_pair(item->name, item->impulse->createData(data)));
    }
  }

  MatrixXs Jc;
  MatrixXs dv0_dq;
  VectorXs vnext;
  MatrixXs dvnext_dx;
  typename ImpulseModelMultiple::ImpulseDataContainer impulses;
  pinocchio::container::aligned_vector<pinocchio::ForceTpl<Scalar> > fext;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/impulses/multiple-impulses.hxx"

#endif  // CROCODDYL_MULTIBODY_IMPULSES_MULTIPLE_IMPULSES_HPP_
