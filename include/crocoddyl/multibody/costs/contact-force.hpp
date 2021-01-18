///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTACT_FORCE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTACT_FORCE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

/**
 * @brief Define a contact force cost function
 *
 * This cost function defines a residual vector \f$\mathbf{r}=\boldsymbol{\lambda}-\boldsymbol{\lambda}^*\f$,
 * where \f$\boldsymbol{\lambda}, \boldsymbol{\lambda}^*\f$ are the current and reference spatial forces, respetively.
 * The current spatial forces \f$\boldsymbol{\lambda}\in\mathbb{R}^{nc}\f$is computed by
 * `DifferentialActionModelContactFwdDynamicsTpl`, with `nc` as the dimension of the contact.
 *
 * Both cost and residual derivatives are computed analytically, where th force vector \f$\boldsymbol{\lambda}\f$ and
 * its derivatives \f$\left(\frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{x}},
 * \frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{u}}\right)\f$ are computed by
 * `DifferentialActionModelContactFwdDynamicsTpl`. These values are stored in a shared data (i.e.
 * DataCollectorContactTpl). Note that this cost function cannot be used with other action models.
 * For the computation of the cost Hessian, we use the Gauss-Newton approximation, e.g.
 * \f$\mathbf{l_{xu}} = \mathbf{l_{x}}^T \mathbf{l_{u}} \f$.
 *
 * \sa DifferentialActionModelContactFwdDynamicsTpl, DataCollectorContactTpl, ActivationModelAbstractTpl
 */
template <typename _Scalar>
class CostModelContactForceTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataContactForceTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef FrameForceTpl<Scalar> FrameForce;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the contact force cost model
   *
   * Note that the `nr`, defined in the activation model, has to be lower / equals than 6.
   *
   * @param[in] state       Multibody state
   * @param[in] activation  Activation model
   * @param[in] fref        Reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
   * @param[in] nu          Dimension of control vector
   */
  CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                           boost::shared_ptr<ActivationModelAbstract> activation, const FrameForce& fref,
                           const std::size_t& nu);

  /**
   * @brief Initialize the contact force cost model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`. Note that the `nr`, defined in the activation
   * model, has to be lower / equals than 6.
   *
   * @param[in] state       Multibody state
   * @param[in] activation  Activation model
   * @param[in] fref        Reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
   */
  CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                           boost::shared_ptr<ActivationModelAbstract> activation, const FrameForce& fref);

  /**
   * @brief Initialize the contact force cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * Note that the `nr`, defined in the activation model, has to be lower / equals than 6.
   *
   * @param[in] state  Multibody state
   * @param[in] fref   Reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
   * @param[in] nr     Dimension of residual vector
   * @param[in] nu     Dimension of control vector
   */
  CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state, const FrameForce& fref, const std::size_t& nr,
                           const std::size_t& nu);

  /**
   * @brief Initialize the contact force cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$), and `nu`
   * is obtained from `StateAbstractTpl::get_nv()`. Note that the `nr`, defined in the activation model, has to be
   * lower / equals than 6.
   *
   * @param[in] state  Multibody state
   * @param[in] fref   Reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
   * @param[in] nr     Dimension of residual vector
   */
  CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state, const FrameForce& fref, const std::size_t& nr);

  /**
   * @brief Initialize the contact force cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$), and `nr`
   * and `nu` are equals to 6 and `StateAbstractTpl::get_nv()`, respectively.
   *
   * @param[in] state  Multibody state
   * @param[in] fref   Reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
   */
  CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state, const FrameForce& fref);
  virtual ~CostModelContactForceTpl();

  /**
   * @brief Compute the contact force cost
   *
   * The force vector is computed by DifferentialActionModelContactFwdDynamicsTpl and stored in
   * DataCollectorContactTpl.
   *
   * @param[in] data  Contact force data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the contact force cost
   *
   * The force derivatives are computed by DifferentialActionModelContactFwdDynamicsTpl and stored in
   * DataCollectorContactTpl.
   *
   * @param[in] data  Contact force data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the contact force cost data
   *
   * @param[in] data  shared data (it should be of type DataCollectorContactTpl)
   * @return the cost data.
   */
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  DEPRECATED("Use set_reference<FrameForceTpl<Scalar> >()", void set_fref(const FrameForce& fref));
  DEPRECATED("Use get_reference<FrameForceTpl<Scalar> >()", const FrameForce& get_fref() const);

 protected:
  /**
   * @brief Return the reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Modify the reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv) const;

  using Base::activation_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 protected:
  FrameForce fref_;  //!< Reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
};

template <typename _Scalar>
struct CostDataContactForceTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef FrameForceTpl<Scalar> FrameForce;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  CostDataContactForceTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), Arr_Ru(model->get_activation()->get_nr(), model->get_state()->get_nv()) {
    Arr_Ru.setZero();
    contact_type = ContactUndefined;

    // Check that proper shared data has been passed
    DataCollectorContactTpl<Scalar>* d = dynamic_cast<DataCollectorContactTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorContact");
    }

    // Avoids data casting at runtime
    FrameForce fref = model->template get_reference<FrameForce>();
    const boost::shared_ptr<StateMultibody>& state = boost::static_pointer_cast<StateMultibody>(model->get_state());
    std::string frame_name = state->get_pinocchio()->frames[fref.id].name;
    bool found_contact = false;
    for (typename ContactModelMultiple::ContactDataContainer::iterator it = d->contacts->contacts.begin();
         it != d->contacts->contacts.end(); ++it) {
      if (it->second->frame == fref.id) {
        ContactData3DTpl<Scalar>* d3d = dynamic_cast<ContactData3DTpl<Scalar>*>(it->second.get());
        if (d3d != NULL) {
          contact_type = Contact3D;
          if (model->get_activation()->get_nr() != 3) {
            throw_pretty("Domain error: nr isn't defined as 3 in the activation model for the 3d contact in " +
                         frame_name);
          }
          found_contact = true;
          contact = it->second;
          break;
        }
        ContactData6DTpl<Scalar>* d6d = dynamic_cast<ContactData6DTpl<Scalar>*>(it->second.get());
        if (d6d != NULL) {
          contact_type = Contact6D;
          if (model->get_activation()->get_nr() != 6) {
            throw_pretty("Domain error: nr isn't defined as 6 in the activation model for the 3d contact in " +
                         frame_name);
          }
          found_contact = true;
          contact = it->second;
          break;
        }
        throw_pretty("Domain error: there isn't defined at least a 3d contact for " + frame_name);
        break;
      }
    }
    if (!found_contact) {
      throw_pretty("Domain error: there isn't defined contact data for " + frame_name);
    }
  }

  boost::shared_ptr<ContactDataAbstractTpl<Scalar> > contact;
  MatrixXs Arr_Ru;
  ContactType contact_type;
  using Base::activation;
  using Base::cost;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/contact-force.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTACT_FORCE_HPP_
