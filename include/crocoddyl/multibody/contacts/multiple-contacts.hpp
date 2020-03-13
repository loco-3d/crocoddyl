///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_
#define CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_

#include <string>
#include <map>
#include <utility>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/contact-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
struct ContactItemTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  ContactItemTpl() {}
  ContactItemTpl(const std::string& name, boost::shared_ptr<ContactModelAbstractTpl<Scalar> > contact)
      : name(name), contact(contact) {}

  std::string name;
  boost::shared_ptr<ContactModelAbstractTpl<Scalar> > contact;
};

template <typename _Scalar>
class ContactModelMultipleTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ContactDataAbstractTpl<Scalar> ContactDataAbstract;
  typedef ContactDataMultipleTpl<Scalar> ContactDataMultiple;
  typedef ContactModelAbstractTpl<Scalar> ContactModelAbstract;

  typedef ContactItemTpl<Scalar> ContactItem;

  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  typedef std::map<std::string, ContactItem> ContactModelContainer;
  typedef std::map<std::string, boost::shared_ptr<ContactDataAbstract> > ContactDataContainer;
  typedef typename pinocchio::container::aligned_vector<pinocchio::ForceTpl<Scalar> >::iterator ForceIterator;

  ContactModelMultipleTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& nu);
  ContactModelMultipleTpl(boost::shared_ptr<StateMultibody> state);
  ~ContactModelMultipleTpl();

  void addContact(const std::string& name, boost::shared_ptr<ContactModelAbstract> contact);
  void removeContact(const std::string& name);

  void calc(const boost::shared_ptr<ContactDataMultiple>& data, const Eigen::Ref<const VectorXs>& x);
  void calcDiff(const boost::shared_ptr<ContactDataMultiple>& data, const Eigen::Ref<const VectorXs>& x);

  void updateAcceleration(const boost::shared_ptr<ContactDataMultiple>& data, const VectorXs& dv) const;
  void updateForce(const boost::shared_ptr<ContactDataMultiple>& data, const VectorXs& force);
  void updateAccelerationDiff(const boost::shared_ptr<ContactDataMultiple>& data, const MatrixXs& ddv_dx) const;
  void updateForceDiff(const boost::shared_ptr<ContactDataMultiple>& data, const MatrixXs& df_dx,
                       const MatrixXs& df_du) const;
  boost::shared_ptr<ContactDataMultiple> createData(pinocchio::DataTpl<Scalar>* const data);

  const boost::shared_ptr<StateMultibody>& get_state() const;
  const ContactModelContainer& get_contacts() const;
  const std::size_t& get_nc() const;
  const std::size_t& get_nu() const;

 private:
  boost::shared_ptr<StateMultibody> state_;
  ContactModelContainer contacts_;
  std::size_t nc_;
  std::size_t nu_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<ContactDataMultiple>& data, const VectorXs& x) { calc(data, x); }

  void calcDiff_wrap(const boost::shared_ptr<ContactDataMultiple>& data, const VectorXs& x) { calcDiff(data, x); }

#endif
};

template <typename _Scalar>
struct ContactDataMultipleTpl : ContactDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactDataAbstractTpl<Scalar> Base;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef ContactItemTpl<Scalar> ContactItem;

  typedef typename MathBase::Vector2s Vector2s;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ContactDataMultipleTpl(Model<Scalar>* const model, pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data),
        dv(model->get_state()->get_nv()),
        ddv_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        fext(model->get_state()->get_pinocchio()->njoints, pinocchio::ForceTpl<Scalar>::Zero()) {
    dv.setZero();
    ddv_dx.setZero();
    for (typename ContactModelMultiple::ContactModelContainer::const_iterator it = model->get_contacts().begin();
         it != model->get_contacts().end(); ++it) {
      const ContactItem& item = it->second;
      contacts.insert(std::make_pair(item.name, item.contact->createData(data)));
    }
  }

  VectorXs dv;
  MatrixXs ddv_dx;
  typename ContactModelMultiple::ContactDataContainer contacts;
  pinocchio::container::aligned_vector<pinocchio::ForceTpl<Scalar> > fext;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/contacts/multiple-contacts.hxx"

#endif  // CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_
