///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_
#define CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_

#include <string>
#include <map>
#include <utility>
#include "crocoddyl/multibody/contact-base.hpp"

namespace crocoddyl {

struct ContactItem {
  ContactItem() {}
  ContactItem(const std::string& name, ContactModelAbstract* contact) : name(name), contact(contact) {}

  std::string name;
  ContactModelAbstract* contact;
};

struct ContactDataMultiple;  // forward declaration

class ContactModelMultiple {
 public:
  typedef std::map<std::string, ContactItem> ContactModelContainer;
  typedef std::map<std::string, boost::shared_ptr<ContactDataAbstract> > ContactDataContainer;
  typedef pinocchio::container::aligned_vector<pinocchio::Force>::iterator ForceIterator;

  ContactModelMultiple(boost::shared_ptr<StateMultibody> state, const std::size_t& nu);
  ContactModelMultiple(boost::shared_ptr<StateMultibody> state);
  ~ContactModelMultiple();

  void addContact(const std::string& name, ContactModelAbstract* const contact);
  void removeContact(const std::string& name);

  void calc(const boost::shared_ptr<ContactDataMultiple>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<ContactDataMultiple>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const bool& recalc = true);

  void updateAcceleration(const boost::shared_ptr<ContactDataMultiple>& data, const Eigen::VectorXd& dv) const;
  void updateForce(const boost::shared_ptr<ContactDataMultiple>& data, const Eigen::VectorXd& force);
  void updateAccelerationDiff(const boost::shared_ptr<ContactDataMultiple>& data, const Eigen::MatrixXd& ddv_dx) const;
  void updateForceDiff(const boost::shared_ptr<ContactDataMultiple>& data, const Eigen::MatrixXd& df_dx,
                       const Eigen::MatrixXd& df_du) const;
  boost::shared_ptr<ContactDataMultiple> createData(pinocchio::Data* const data);

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
  void calc_wrap(const boost::shared_ptr<ContactDataMultiple>& data, const Eigen::VectorXd& x) { calc(data, x); }

  void calcDiff_wrap(const boost::shared_ptr<ContactDataMultiple>& data, const Eigen::VectorXd& x,
                     const bool& recalc = true) {
    calcDiff(data, x, recalc);
  }

#endif
};

struct ContactDataMultiple : ContactDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ContactDataMultiple(Model* const model, pinocchio::Data* const data)
      : ContactDataAbstract(model, data),
        dv(model->get_state()->get_nv()),
        ddv_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        fext(model->get_state()->get_pinocchio().njoints, pinocchio::Force::Zero()) {
    dv.fill(0);
    ddv_dx.fill(0);
    for (ContactModelMultiple::ContactModelContainer::const_iterator it = model->get_contacts().begin();
         it != model->get_contacts().end(); ++it) {
      const ContactItem& item = it->second;
      contacts.insert(std::make_pair(item.name, item.contact->createData(data)));
    }
  }

  Eigen::VectorXd dv;
  Eigen::MatrixXd ddv_dx;
  ContactModelMultiple::ContactDataContainer contacts;
  pinocchio::container::aligned_vector<pinocchio::Force> fext;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_
