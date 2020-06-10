///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTACT_FORCE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTACT_FORCE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

enum ContactType { Contact3D, Contact6D, Undefined };

/**
 * @brief Define a force cost function
 *
 * It builds a cost function to track a desired force of a given frame placement, i.e. the cost residual is defined as:
 * \f{equation*}{
 *   \mathbf{r} = \boldsymbol{\lambda}_i - \boldsymbol{\lambda}_i^{desired}
 * \f}
 * where \f$\boldsymbol{\lambda}_i\in\mathbb{R}^{nc}\f$ is the contact force of the frame \f$i\f$, and
 * \f$\boldsymbol{\lambda}_i^{desired}\in\mathbb{R}^{nc}\f$ is the reference force value, and \f$nc\f$ defines the
 * dimension of contact force. The cost is computed from the residual vector through an user defined action model.
 * Additionally, the reference force vector is defined using FrameForceTpl even for cased where \f$nc <
 * 6\f$.
 *
 * The force vector and its derivatives are computed by DifferentialActionModelContactFwdDynamicsTpl. These values are
 * stored in a shared data (i.e. DataCollectorContactTpl). Note that this cost function cannot be used with other
 * action models.
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
  typedef ActivationModelQuadTpl<Scalar> ActivationModelQuad;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef FrameForceTpl<Scalar> FrameForce;
  typedef typename MathBase::Vector6s Vector6s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the contact force cost model
   *
   * Note that the `nr`, defined in the activation model, has to be lower / equals than 6.
   *
   * @param[in] state       Multibody state
   * @param[in] activation  Activation model
   * @param[in] fref        Reference contact force
   * @param[in] nu          Dimension of control vector
   */
  CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                           boost::shared_ptr<ActivationModelAbstract> activation, const FrameForce& fref,
                           const std::size_t& nu);

  /**
   * @brief Initialize the contact force cost model
   *
   * For this case the default nu is equals to `state->get_nv()`. Note that the `nr`, defined in the activation model,
   * has to be lower / equals than 6.
   *
   * @param[in] state       Multibody state
   * @param[in] activation  Activation model
   * @param[in] fref        Reference contact force
   */
  CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                           boost::shared_ptr<ActivationModelAbstract> activation, const FrameForce& fref);

  /**
   * @brief Initialize the contact force cost model
   *
   * For this case the default activation model is quadratic, i.e. `ActivationModelQuadTpl(nr)`.
   * Note that the `nr`, defined in the activation model, has to be lower / equals than 6.
   *
   * @param[in] state       Multibody state
   * @param[in] fref        Reference contact force
   * @param[in] nr          Dimension of residual vector
   * @param[in] nu          Dimension of control vector
   */
  CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state, const FrameForce& fref, const std::size_t& nr,
                           const std::size_t& nu);

  /**
   * @brief Initialize the contact force cost model
   *
   * For this case the default activation model is quadratic, i.e. `ActivationModelQuadTpl(nr)`, and `nu` is equals to
   * `state->get_nv()`. Note that the `nr`, defined in the activation model, has to be lower / equals than 6.
   *
   * @param[in] state       Multibody state
   * @param[in] fref        Reference contact force
   * @param[in] nr          Dimension of residual vector
   */
  CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state, const FrameForce& fref, const std::size_t& nr);

  /**
   * @brief Initialize the contact force cost model
   *
   * For this case the default activation model is quadratic, i.e. `ActivationModelQuadTpl(nr)`, and `nr` and `nu` is
   * equals to 6 and `state->get_nv()`, respectively.
   *
   * @param[in] state       Multibody state
   * @param[in] fref        Reference contact force
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
   * @param[in] x     State vector
   * @param[in] u     Control input
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the contact force cost
   *
   * * The force derivatives are computed by DifferentialActionModelContactFwdDynamicsTpl and stored in
   * DataCollectorContactTpl.
   * @param[in] data  Contact force data
   * @param[in] x     State vector
   * @param[in] u     Control input
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the contact force cost data
   *
   * Each cost model has its own data that needs to be allocated. This function returns the allocated data for a
   * predefined cost.
   *
   * @param[in] data  shared data (it should be of type DataCollectorContactTpl)
   * @return the cost data.
   */
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the contact force
   */
  const FrameForce& get_fref() const;

  /**
   * @brief Modify the contact force
   */
  void set_fref(const FrameForce& fref);

 protected:
  /**
   * @brief Return the contact force
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Modify the contact force
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv);

  using Base::activation_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 protected:
  FrameForce fref_;  //!< Contact force
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
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  template <template <typename Scalar> class Model>
  CostDataContactForceTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), Arr_Ru(model->get_activation()->get_nr(), model->get_state()->get_nv()) {
    Arr_Ru.setZero();
    contact_type = Undefined;

    // Check that proper shared data has been passed
    DataCollectorContactTpl<Scalar>* d = dynamic_cast<DataCollectorContactTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorContact");
    }

    // Avoids data casting at runtime
    const FrameForce& fref = model->get_fref();
    std::string frame_name = model->get_state()->get_pinocchio()->frames[model->get_fref().frame].name;
    bool found_contact = false;
    for (typename ContactModelMultiple::ContactDataContainer::iterator it = d->contacts->contacts.begin();
         it != d->contacts->contacts.end(); ++it) {
      if (it->second->frame == fref.frame) {
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
