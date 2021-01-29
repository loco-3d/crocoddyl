///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_WRENCH_CONE_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_WRENCH_CONE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/wrench-cone.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Define a contact wrench cone residual function
 *
 * This residual function is defined as \f$\mathbf{r}=\mathbf{A}\boldsymbol{\lambda}\f$,
 * where \f$\mathbf{A}\f$ is the inequality matrix defined by the contact wrench cone, and \f$\boldsymbol{\lambda}\f$
 * is the current spatial forces. The current spatial forces \f$\boldsymbol{\lambda}\in\mathbb{R}^{nc}\f$ is computed
 * by `DifferentialActionModelContactFwdDynamicsTpl`, with `nc` as the dimension of the contact.
 *
 * Both residual and residual Jacobians are computed analytically, where the force vector \f$\boldsymbol{\lambda}\f$
 * and its Jacobians \f$\left(\frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{x}},
 * \frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{u}}\right)\f$ are computed by
 * `DifferentialActionModelContactFwdDynamicsTpl`. These values are stored in a shared data (i.e.
 * DataCollectorContactTpl). Note that this residual function cannot be used with other action models.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`,
 * `DifferentialActionModelContactFwdDynamicsTpl`, `DataCollectorContactTpl`
 */
template <typename _Scalar>
class ResidualModelContactWrenchConeTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataContactWrenchConeTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef WrenchConeTpl<Scalar> WrenchCone;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixX6s MatrixX6s;

  /**
   * @brief Initialize the contact wrench cone residual model
   *
   * @param[in] state  Multibody state
   * @param[in] id     Reference frame id
   * @param[in] fref   Reference contact wrench cone
   * @param[in] nu     Dimension of control vector
   */
  ResidualModelContactWrenchConeTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                    const WrenchCone& fref, const std::size_t nu);

  /**
   * @brief Initialize the contact wrench cone residual model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  Multibody state
   * @param[in] id     Reference frame id
   * @param[in] fref   Reference contact wrench cone
   */
  ResidualModelContactWrenchConeTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                    const WrenchCone& fref);
  virtual ~ResidualModelContactWrenchConeTpl();

  /**
   * @brief Compute the contact wrench cone residual
   *
   * The force vector is computed by DifferentialActionModelContactFwdDynamicsTpl and stored in
   * DataCollectorContactTpl.
   *
   * @param[in] data  Contact force data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the contact wrench cone residual
   *
   * The force derivatives are computed by DifferentialActionModelContactFwdDynamicsTpl and stored in
   * DataCollectorContactTpl.
   *
   * @param[in] data  Contact force data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the contact wrench cone residual data
   *
   * @param[in] data  shared data (it should be of type DataCollectorContactTpl)
   * @return the residual data.
   */
  virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Return the reference contact wrench cone
   */
  const WrenchCone& get_reference() const;

  /**
   * @brief Modify the reference frame id
   */
  void set_id(pinocchio::FrameIndex id);

  /**
   * @brief Modify the reference contact wrench cone
   */
  void set_reference(const WrenchCone& reference);

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  pinocchio::FrameIndex id_;  //!< Reference frame id
  WrenchCone fref_;           //!< Reference contact wrench cone
};

template <typename _Scalar>
struct ResidualDataContactWrenchConeTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef FrameWrenchConeTpl<Scalar> FrameWrenchCone;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ResidualDataContactWrenchConeTpl(Model<Scalar>* const model, DataCollectorAbstract* const data) : Base(model, data) {
    // Check that proper shared data has been passed
    DataCollectorContactTpl<Scalar>* d = dynamic_cast<DataCollectorContactTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorContact");
    }

    // Avoids data casting at runtime
    const pinocchio::FrameIndex id = model->get_id();
    const boost::shared_ptr<StateMultibody>& state = boost::static_pointer_cast<StateMultibody>(model->get_state());
    std::string frame_name = state->get_pinocchio()->frames[id].name;
    bool found_contact = false;
    for (typename ContactModelMultiple::ContactDataContainer::iterator it = d->contacts->contacts.begin();
         it != d->contacts->contacts.end(); ++it) {
      if (it->second->frame == id) {
        ContactData3DTpl<Scalar>* d3d = dynamic_cast<ContactData3DTpl<Scalar>*>(it->second.get());
        if (d3d != NULL) {
          found_contact = true;
          contact = it->second;
          throw_pretty("Domain error: there isn't defined at least a 6d contact for " + frame_name);
          break;
        }
        ContactData6DTpl<Scalar>* d6d = dynamic_cast<ContactData6DTpl<Scalar>*>(it->second.get());
        if (d6d != NULL) {
          found_contact = true;
          contact = it->second;
          break;
        }
        throw_pretty("Domain error: there isn't defined at least a 6d contact for " + frame_name);
        break;
      }
    }
    if (!found_contact) {
      throw_pretty("Domain error: there isn't defined contact data for " + frame_name);
    }
  }

  boost::shared_ptr<ContactDataAbstractTpl<Scalar> > contact;
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hxx"

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_WRENCH_CONE_HPP_
