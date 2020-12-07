///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
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
  typedef ContactModelAbstractTpl<Scalar> ContactModelAbstract;

  ContactItemTpl() {}
  ContactItemTpl(const std::string& name, boost::shared_ptr<ContactModelAbstract> contact, bool active = true)
      : name(name), contact(contact), active(active) {}

  std::string name;
  boost::shared_ptr<ContactModelAbstract> contact;
  bool active;
};

/**
 * @brief Define a stack of contact models
 *
 * The contact models can be defined with active and inactive status. The idea behind this design choice is to be able
 * to create a mechanism that allocates the entire data needed for the computations. Then, there are designed routines
 * that update the only active contacts.
 */
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

  typedef std::map<std::string, boost::shared_ptr<ContactItem> > ContactModelContainer;
  typedef std::map<std::string, boost::shared_ptr<ContactDataAbstract> > ContactDataContainer;
  typedef typename pinocchio::container::aligned_vector<pinocchio::ForceTpl<Scalar> >::iterator ForceIterator;

  /**
   * @brief Initialize the multi-contact model
   *
   * @param[in] state  Multibody state
   * @param[in] nu     Dimension of control vector
   */
  ContactModelMultipleTpl(boost::shared_ptr<StateMultibody> state, std::size_t nu);

  /**
   * @brief Initialize the multi-contact model
   *
   * @param[in] state  Multibody state
   */
  ContactModelMultipleTpl(boost::shared_ptr<StateMultibody> state);
  ~ContactModelMultipleTpl();

  /**
   * @brief Add contact item
   *
   * Note that the memory is allocated for inactive contacts as well.
   *
   * @param[in] name     Contact name
   * @param[in] contact  Contact model
   * @param[in] active   Contact status (active by default)
   */
  void addContact(const std::string& name, boost::shared_ptr<ContactModelAbstract> contact, bool active = true);

  /**
   * @brief Remove contact item
   *
   * @param[in] name  Contact name
   */
  void removeContact(const std::string& name);

  /**
   * @brief Change the contact status
   *
   * @param[in] name     Contact name
   * @param[in] active   Contact status (True for active)
   */
  void changeContactStatus(const std::string& name, bool active);

  /**
   * @brief Compute the contact Jacobian and contact acceleration
   *
   * @param[in] data  Multi-contact data
   * @param[in] x     State vector \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  void calc(const boost::shared_ptr<ContactDataMultiple>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the contact holonomic constraint
   *
   * @param[in] data  Multi-contact data
   * @param[in] x     State vector \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  void calcDiff(const boost::shared_ptr<ContactDataMultiple>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Update the constrained system acceleration
   *
   * @param[in] data  Multi-contact data
   * @param[in] dv    Constrained system acceleration \f$\dot{\mathbf{v}}\in\mathbb{R}^{nv}\f$
   */
  void updateAcceleration(const boost::shared_ptr<ContactDataMultiple>& data, const VectorXs& dv) const;

  /**
   * @brief Update the spatial force defined in frame coordinate
   *
   * @param[in] data   Multi-contact data
   * @param[in] force  Spatial force defined in frame coordinate
   * \f${}^o\underline{\boldsymbol{\lambda}}_c\in\mathbb{R}^{nc}\f$
   */
  void updateForce(const boost::shared_ptr<ContactDataMultiple>& data, const VectorXs& force);

  /**
   * @brief Update the Jacobian of the constrained system acceleration
   *
   * @param[in] data    Multi-contact data
   * @param[in] ddv_dx  Jacobian of the system acceleration in generalized coordinates
   * \f$\frac{\partial\dot{\mathbf{v}}}{\partial\mathbf{x}}\in\mathbb{R}^{nv\times ndx}\f$
   */
  void updateAccelerationDiff(const boost::shared_ptr<ContactDataMultiple>& data, const MatrixXs& ddv_dx) const;

  /**
   * @brief Update the Jacobian of the spatial force defined in frame coordinate
   *
   * @param[in] data   Multi-contact data
   * @param[in] df_dx  Jacobian of the spatial impulse defined in frame coordinate
   * \f$\frac{\partial{}^o\underline{\boldsymbol{\lambda}}_c}{\partial\mathbf{x}}\in\mathbb{R}^{nc\times{ndx}}\f$
   * @param[in] df_du  Jacobian of the spatial impulse defined in frame coordinate
   * \f$\frac{\partial{}^o\underline{\boldsymbol{\lambda}}_c}{\partial\mathbf{u}}\in\mathbb{R}^{nc\times{nu}}\f$
   */
  void updateForceDiff(const boost::shared_ptr<ContactDataMultiple>& data, const MatrixXs& df_dx,
                       const MatrixXs& df_du) const;

  /**
   * @brief Create the multi-contact data
   *
   * @param[in] data  Pinocchio data
   * @return the multi-contact data.
   */
  boost::shared_ptr<ContactDataMultiple> createData(pinocchio::DataTpl<Scalar>* const data);

  /**
   * @brief Return the multibody state
   */
  const boost::shared_ptr<StateMultibody>& get_state() const;

  /**
   * @brief Return the contact models
   */
  const ContactModelContainer& get_contacts() const;

  /**
   * @brief Return the dimension of active contacts
   */
  std::size_t get_nc() const;

  /**
   * @brief Return the dimension of all contacts
   */
  std::size_t get_nc_total() const;

  /**
   * @brief Return the dimension of control vector
   */
  std::size_t get_nu() const;

  /**
   * @brief Return the names of the active contacts
   */
  const std::vector<std::string>& get_active() const;

  /**
   * @brief Return the names of the inactive contacts
   */
  const std::vector<std::string>& get_inactive() const;

  /**
   * @brief Return the status of a given contact name
   */
  bool getContactStatus(const std::string& name) const;

 private:
  boost::shared_ptr<StateMultibody> state_;
  ContactModelContainer contacts_;
  std::size_t nc_;
  std::size_t nc_total_;
  std::size_t nu_;
  std::vector<std::string> active_;
  std::vector<std::string> inactive_;
};

/**
 * @brief Define the multi-contact data
 *
 * \sa ContactModelMultipleTpl
 */
template <typename _Scalar>
struct ContactDataMultipleTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef ContactItemTpl<Scalar> ContactItem;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialized a multi-contact data
   *
   * @param[in] model  Multi-contact model
   * @param[in] data   Pinocchio data
   */
  template <template <typename Scalar> class Model>
  ContactDataMultipleTpl(Model<Scalar>* const model, pinocchio::DataTpl<Scalar>* const data)
      : Jc(model->get_nc_total(), model->get_state()->get_nv()),
        a0(model->get_nc_total()),
        da0_dx(model->get_nc_total(), model->get_state()->get_ndx()),
        dv(model->get_state()->get_nv()),
        ddv_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        fext(model->get_state()->get_pinocchio()->njoints, pinocchio::ForceTpl<Scalar>::Zero()) {
    Jc.setZero();
    a0.setZero();
    da0_dx.setZero();
    dv.setZero();
    ddv_dx.setZero();
    for (typename ContactModelMultiple::ContactModelContainer::const_iterator it = model->get_contacts().begin();
         it != model->get_contacts().end(); ++it) {
      const boost::shared_ptr<ContactItem>& item = it->second;
      contacts.insert(std::make_pair(item->name, item->contact->createData(data)));
    }
  }

  MatrixXs Jc;  //!< Contact Jacobian in frame coordinate \f$\mathbf{J}_c\in\mathbb{R}^{nc_{total}\times{nv}}\f$
                //!< (memory defined for active and inactive contacts)
  VectorXs a0;  //!< Desired spatial contact acceleration in frame coordinate
                //!< \f$\underline{\mathbf{a}}_0\in\mathbb{R}^{nc_{total}}\f$ (memory defined for active and inactive
                //!< contacts)
  MatrixXs
      da0_dx;  //!< Jacobian of the desired spatial contact acceleration in frame coordinate
               //!< \f$\frac{\partial\underline{\mathbf{a}}_0}{\partial\mathbf{x}}\in\mathbb{R}^{nc_{total}\times{ndx}}\f$
               //!< (memory defined for active and inactive contacts)
  VectorXs
      dv;  //!< Constrained system acceleration in generalized coordinates \f$\dot{\mathbf{v}}\in\mathbb{R}^{nv}\f$
  MatrixXs ddv_dx;  //!< Jacobian of the system acceleration in generalized coordinates
                    //!< \f$\frac{\partial\dot{\mathbf{v}}}{\partial\mathbf{x}}\in\mathbb{R}^{nv\times ndx}\f$
  typename ContactModelMultiple::ContactDataContainer contacts;  //!< Stack of contact data
  pinocchio::container::aligned_vector<pinocchio::ForceTpl<Scalar> >
      fext;  //!< External spatial forces in body coordinates
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/contacts/multiple-contacts.hxx"

#endif  // CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_
