///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_
#define CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_

#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
struct ContactItemTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ContactModelAbstractTpl<Scalar> ContactModelAbstract;

  ContactItemTpl() {}
  ContactItemTpl(const std::string& name,
                 std::shared_ptr<ContactModelAbstract> contact,
                 const bool active = true)
      : name(name), contact(contact), active(active) {}

  template <typename NewScalar>
  ContactItemTpl<NewScalar> cast() const {
    typedef ContactItemTpl<NewScalar> ReturnType;
    ReturnType ret(name, contact->template cast<NewScalar>(), active);
    return ret;
  }

  /**
   * @brief Print information on the contact item
   */
  friend std::ostream& operator<<(std::ostream& os,
                                  const ContactItemTpl<Scalar>& model) {
    os << "{" << *model.contact << "}";
    return os;
  }

  std::string name;
  std::shared_ptr<ContactModelAbstract> contact;
  bool active;
};

/**
 * @brief Define a stack of contact models
 *
 * The contact models can be defined with active and inactive status. The idea
 * behind this design choice is to be able to create a mechanism that allocates
 * the entire data needed for the computations. Then, there are designed
 * routines that update the only active contacts.
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

  typedef std::map<std::string, std::shared_ptr<ContactItem> >
      ContactModelContainer;
  typedef std::map<std::string, std::shared_ptr<ContactDataAbstract> >
      ContactDataContainer;
  typedef typename pinocchio::container::aligned_vector<
      pinocchio::ForceTpl<Scalar> >::iterator ForceIterator;

  /**
   * @brief Initialize the multi-contact model
   *
   * @param[in] state  Multibody state
   * @param[in] nu     Dimension of control vector
   */
  ContactModelMultipleTpl(std::shared_ptr<StateMultibody> state,
                          const std::size_t nu);

  /**
   * @brief Initialize the multi-contact model
   *
   * @param[in] state  Multibody state
   */
  ContactModelMultipleTpl(std::shared_ptr<StateMultibody> state);
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
  void addContact(const std::string& name,
                  std::shared_ptr<ContactModelAbstract> contact,
                  const bool active = true);

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
  void changeContactStatus(const std::string& name, const bool active);

  /**
   * @brief Compute the contact Jacobian and contact acceleration
   *
   * @param[in] data  Multi-contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  void calc(const std::shared_ptr<ContactDataMultiple>& data,
            const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the contact holonomic constraint
   *
   * @param[in] data  Multi-contact data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  void calcDiff(const std::shared_ptr<ContactDataMultiple>& data,
                const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Update the constrained system acceleration
   *
   * @param[in] data  Multi-contact data
   * @param[in] dv    Constrained system acceleration
   * \f$\dot{\mathbf{v}}\in\mathbb{R}^{nv}\f$
   */
  void updateAcceleration(const std::shared_ptr<ContactDataMultiple>& data,
                          const VectorXs& dv) const;

  /**
   * @brief Update the spatial force defined in frame coordinate
   *
   * @param[in] data   Multi-contact data
   * @param[in] force  Spatial force defined in frame coordinate
   * \f${}^o\underline{\boldsymbol{\lambda}}_c\in\mathbb{R}^{nc}\f$
   */
  void updateForce(const std::shared_ptr<ContactDataMultiple>& data,
                   const VectorXs& force);

  /**
   * @brief Update the Jacobian of the constrained system acceleration
   *
   * @param[in] data    Multi-contact data
   * @param[in] ddv_dx  Jacobian of the system acceleration in generalized
   * coordinates
   * \f$\frac{\partial\dot{\mathbf{v}}}{\partial\mathbf{x}}\in\mathbb{R}^{nv\times
   * ndx}\f$
   */
  void updateAccelerationDiff(const std::shared_ptr<ContactDataMultiple>& data,
                              const MatrixXs& ddv_dx) const;

  /**
   * @brief Update the Jacobian of the spatial force defined in frame coordinate
   *
   * @param[in] data   Multi-contact data
   * @param[in] df_dx  Jacobian of the spatial impulse defined in frame
   * coordinate
   * \f$\frac{\partial{}^o\underline{\boldsymbol{\lambda}}_c}{\partial\mathbf{x}}\in\mathbb{R}^{nc\times{ndx}}\f$
   * @param[in] df_du  Jacobian of the spatial impulse defined in frame
   * coordinate
   * \f$\frac{\partial{}^o\underline{\boldsymbol{\lambda}}_c}{\partial\mathbf{u}}\in\mathbb{R}^{nc\times{nu}}\f$
   */
  void updateForceDiff(const std::shared_ptr<ContactDataMultiple>& data,
                       const MatrixXs& df_dx, const MatrixXs& df_du) const;

  /**
   * @brief Update the RNEA derivatives dtau_dq by adding the skew term
   * (necessary for contacts expressed in LOCAL_WORLD_ALIGNED / WORLD)
   *
   * To learn more about the computation of the contact derivatives in different
   * frames see https://hal.science/hal-03758989/document.
   *
   * @param[in] data       Multi-contact data
   * @param[in] pinocchio  Pinocchio data
   */
  void updateRneaDiff(const std::shared_ptr<ContactDataMultiple>& data,
                      pinocchio::DataTpl<Scalar>& pinocchio) const;

  /**
   * @brief Create the multi-contact data
   *
   * @param[in] data  Pinocchio data
   * @return the multi-contact data.
   */
  std::shared_ptr<ContactDataMultiple> createData(
      pinocchio::DataTpl<Scalar>* const data);

  /**
   * @brief Cast the multi-contact model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ContactModelMultipleTpl<NewScalar> A multi-contact model with the
   * new scalar type.
   */
  template <typename NewScalar>
  ContactModelMultipleTpl<NewScalar> cast() const;

  /**
   * @brief Return the multibody state
   */
  const std::shared_ptr<StateMultibody>& get_state() const;

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
   * @brief Return the names of the set of active contacts
   */
  const std::set<std::string>& get_active_set() const;

  /**
   * @brief Return the names of the set of inactive contacts
   */
  const std::set<std::string>& get_inactive_set() const;

  /**
   * @brief Return the status of a given contact name
   */
  bool getContactStatus(const std::string& name) const;

  /**
   * @brief Return the type of contact computation
   *
   * True for all contacts, otherwise false for active contacts
   */
  bool getComputeAllContacts() const;

  /**
   * @brief Set the tyoe of contact computation: all or active contacts
   *
   * @param status  Type of contact computation (true for all contacts and false
   * for active contacts)
   */
  void setComputeAllContacts(const bool status);

  /**
   * @brief Print information on the contact models
   */
  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os,
                                  const ContactModelMultipleTpl<Scalar>& model);

 private:
  std::shared_ptr<StateMultibody> state_;
  ContactModelContainer contacts_;
  std::size_t nc_;
  std::size_t nc_total_;
  std::size_t nu_;
  std::set<std::string> active_set_;
  std::set<std::string> inactive_set_;
  bool compute_all_contacts_;
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
  ContactDataMultipleTpl(Model<Scalar>* const model,
                         pinocchio::DataTpl<Scalar>* const data)
      : Jc(model->get_nc_total(), model->get_state()->get_nv()),
        a0(model->get_nc_total()),
        da0_dx(model->get_nc_total(), model->get_state()->get_ndx()),
        dv(model->get_state()->get_nv()),
        ddv_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        fext(model->get_state()->get_pinocchio()->njoints,
             pinocchio::ForceTpl<Scalar>::Zero()) {
    Jc.setZero();
    a0.setZero();
    da0_dx.setZero();
    dv.setZero();
    ddv_dx.setZero();
    for (typename ContactModelMultiple::ContactModelContainer::const_iterator
             it = model->get_contacts().begin();
         it != model->get_contacts().end(); ++it) {
      const std::shared_ptr<ContactItem>& item = it->second;
      contacts.insert(
          std::make_pair(item->name, item->contact->createData(data)));
    }
  }

  MatrixXs Jc;  //!< Contact Jacobian in frame coordinate
                //!< \f$\mathbf{J}_c\in\mathbb{R}^{nc_{total}\times{nv}}\f$
                //!< (memory defined for active and inactive contacts)
  VectorXs a0;  //!< Desired spatial contact acceleration in frame coordinate
                //!< \f$\underline{\mathbf{a}}_0\in\mathbb{R}^{nc_{total}}\f$
                //!< (memory defined for active and inactive contacts)
  MatrixXs
      da0_dx;  //!< Jacobian of the desired spatial contact acceleration in
               //!< frame coordinate
               //!< \f$\frac{\partial\underline{\mathbf{a}}_0}{\partial\mathbf{x}}\in\mathbb{R}^{nc_{total}\times{ndx}}\f$
               //!< (memory defined for active and inactive contacts)
  VectorXs dv;  //!< Constrained system acceleration in generalized coordinates
                //!< \f$\dot{\mathbf{v}}\in\mathbb{R}^{nv}\f$
  MatrixXs
      ddv_dx;  //!< Jacobian of the system acceleration in generalized
               //!< coordinates
               //!< \f$\frac{\partial\dot{\mathbf{v}}}{\partial\mathbf{x}}\in\mathbb{R}^{nv\times
               //!< ndx}\f$
  typename ContactModelMultiple::ContactDataContainer
      contacts;  //!< Stack of contact data
  pinocchio::container::aligned_vector<pinocchio::ForceTpl<Scalar> >
      fext;  //!< External spatial forces in body coordinates
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/contacts/multiple-contacts.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ContactItemTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ContactModelMultipleTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ContactDataMultipleTpl)

#endif  // CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_
