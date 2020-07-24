///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTACT_IMPULSE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTACT_IMPULSE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

enum ImpulseType { Impulse3D, Impulse6D, Undefined };

/**
 * @brief Define a contact impulse cost function
 *
 * It builds a cost function that tracks a desired spatial impulse in the contact coordinates
 * \f${}^o\underline{\boldsymbol{\Lambda}}_c\in\mathbb{R}^{nc}\f$, i.e. the cost residual vector is described as:
 * \f{equation*}{ \mathbf{r} = {}^o\underline{\boldsymbol{\Lambda}}_c -
 * {}^o\underline{\boldsymbol{\Lambda}}_c^{reference},\f} where
 * \f${}^o\underline{\boldsymbol{\Lambda}}_c^{reference}\f$ is the reference spatial contact impulse in the frame
 * coordinate \f$c\f$, and \f$nc\f$ defines the dimension of constrained space \f$(nc < 6)\f$. The cost is computed,
 * from the residual vector \f$\mathbf{r}\in\mathbb{R}^{nc}\f$, through an user defined activation model. Additionally,
 * the reference impulse vector is defined using FrameForceTpl even for cases where \f$nc < 6\f$.
 *
 * The impulse vector \f${}^o\underline{\boldsymbol{\Lambda}}_c\f$ and its derivatives
 * \f$\left(\frac{\partial{}^o\underline{\boldsymbol{\Lambda}}_c}{\partial\mathbf{x}},
 * \frac{\partial{}^o\underline{\boldsymbol{\Lambda}}_c}{\partial\mathbf{u}}\right)\f$ are computed by
 * ActionModelImpulseFwdDynamicsTpl. These values are stored in a shared data (i.e.
 * DataCollectorImpulseTpl). Note that this cost function cannot be used with other action models.
 *
 * \sa ActionModelImpulseFwdDynamicsTpl, DataCollectorImpulseTpl, ActivationModelAbstractTpl
 */
template <typename _Scalar>
class CostModelContactImpulseTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataContactImpulseTpl<Scalar> Data;
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
   * @brief Initialize the contact impulse model
   *
   * Note that the `nr`, defined in the activation model, has to be lower / equals than 6.
   *
   * @param[in] state       Multibody state
   * @param[in] activation  Activation model
   * @param[in] fref        Reference spatial contact impulse in the contact coordinates
   */
  CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state,
                             boost::shared_ptr<ActivationModelAbstract> activation, const FrameForce& fref);

  /**
   * @brief Initialize the contact impulse cost model
   *
   * For this case the default activation model is quadratic, i.e. `ActivationModelQuadTpl(nr)`.
   * Note that the `nr`, defined in the activation model, has to be lower / equals than 6.
   *
   * @param[in] state       Multibody state
   * @param[in] fref        Reference spatial contact impulse in the contact coordinates
   * \f${}^o\underline{\boldsymbol{\Lambda}}_c^{reference}\f$
   * @param[in] nr          Dimension of residual vector
   */
  CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state, const FrameForce& fref, const std::size_t& nr);

  /**
   * @brief Initialize the contact impulse cost model
   *
   * For this case the default activation model is quadratic, i.e. `ActivationModelQuadTpl(nr)`, and `nr` is
   * equals to 6.
   *
   * @param[in] state       Multibody state
   * @param[in] fref        Reference spatial contact impulse in the contact coordinates
   * \f${}^o\underline{\boldsymbol{\Lambda}}_c^{reference}\f$
   */
  CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state, const FrameForce& fref);
  virtual ~CostModelContactImpulseTpl();

  /**
   * @brief Compute the contact impulse cost
   *
   * The impulse vector is computed by ActionModelImpulseFwdDynamicsTpl and stored in DataCollectorImpulseTpl.
   *
   * @param[in] data  Contact impulse data
   * @param[in] x     State vector \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the contact impulse cost
   *
   * The impulse derivatives are computed by ActionModelImpulseFwdDynamicsTpl and stored in DataCollectorImpulseTpl.
   *
   * @param[in] data  Contact impulse data
   * @param[in] x     State vector \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the contact impulse cost data
   *
   * Each cost model has its own data that needs to be allocated. This function returns the allocated data for a
   * predefined cost.
   *
   * @param[in] data  shared data (it should be of type DataCollectorImpulseTpl)
   * @return the cost data.
   */
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the reference spatial contact impulse in the contact coordinates
   * \f${}^o\underline{\boldsymbol{\Lambda}}_c^{reference}\f$
   */
  const FrameForce& get_fref() const;

  /**
   * @brief Modify the reference spatial contact impulse in the contact coordinates
   * \f${}^o\underline{\boldsymbol{\Lambda}}_c^{reference}\f$
   */
  void set_fref(const FrameForce& fref);

 protected:
  /**
   * @brief Return the reference spatial contact impulse in the contact coordinates
   * \f${}^o\underline{\boldsymbol{\Lambda}}_c^{reference}\f$
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Modify the reference spatial contact impulse in the contact coordinates
   * \f${}^o\underline{\boldsymbol{\Lambda}}_c^{reference}\f$
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv) const;

  using Base::activation_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 protected:
  FrameForce fref_;  //!< Reference spatial contact impulse in the contact coordinates
                     //!< \f${}^o\underline{\boldsymbol{\Lambda}}_c^{reference}\f$
};

template <typename _Scalar>
struct CostDataContactImpulseTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ImpulseModelMultipleTpl<Scalar> ImpulseModelMultiple;
  typedef FrameForceTpl<Scalar> FrameForce;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  template <template <typename Scalar> class Model>
  CostDataContactImpulseTpl(Model<Scalar>* const model, DataCollectorAbstract* const data) : Base(model, data) {
    impulse_type = Undefined;

    // Check that proper shared data has been passed
    DataCollectorImpulseTpl<Scalar>* d = dynamic_cast<DataCollectorImpulseTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorImpulse");
    }

    // Avoids data casting at runtime
    const FrameForce& fref = model->get_fref();
    std::string frame_name = model->get_state()->get_pinocchio()->frames[model->get_fref().frame].name;
    bool found_impulse = false;
    for (typename ImpulseModelMultiple::ImpulseDataContainer::iterator it = d->impulses->impulses.begin();
         it != d->impulses->impulses.end(); ++it) {
      if (it->second->frame == fref.frame) {
        ImpulseData3DTpl<Scalar>* d3d = dynamic_cast<ImpulseData3DTpl<Scalar>*>(it->second.get());
        if (d3d != NULL) {
          impulse_type = Impulse3D;
          if (model->get_activation()->get_nr() != 3) {
            throw_pretty("Domain error: nr isn't defined as 3 in the activation model for the 3d impulse in " +
                         frame_name);
          }
          found_impulse = true;
          impulse = it->second;
          break;
        }
        ImpulseData6DTpl<Scalar>* d6d = dynamic_cast<ImpulseData6DTpl<Scalar>*>(it->second.get());
        if (d6d != NULL) {
          impulse_type = Impulse6D;
          if (model->get_activation()->get_nr() != 6) {
            throw_pretty("Domain error: nr isn't defined as 6 in the activation model for the 3d impulse in " +
                         frame_name);
          }
          found_impulse = true;
          impulse = it->second;
          break;
        }
        throw_pretty("Domain error: there isn't defined at least a 3d impulse for " + frame_name);
        break;
      }
    }
    if (!found_impulse) {
      throw_pretty("Domain error: there isn't defined impulse data for " + frame_name);
    }
  }

  boost::shared_ptr<ImpulseDataAbstractTpl<Scalar> > impulse;
  ImpulseType impulse_type;
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
#include "crocoddyl/multibody/costs/contact-impulse.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTACT_IMPULSE_HPP_
