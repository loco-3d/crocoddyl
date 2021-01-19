///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_IMPULSE_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_IMPULSE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Define a contact impulse residual function
 *
 * This residual function is defined as \f$\mathbf{r}=\boldsymbol{\lambda}-\boldsymbol{\lambda}^*\f$,
 * where \f$\boldsymbol{\lambda}, \boldsymbol{\lambda}^*\f$ are the current and reference spatial impulses,
 * respectively. The current spatial impulses \f$\boldsymbol{\lambda}\in\mathbb{R}^{ni}\f$is computed by
 * `ActionModelImpulseFwdDynamicsTpl`, with `ni` as the dimension of the impulse.
 *
 * Both residual and its Jacobians are computed analytically, where th force vector \f$\boldsymbol{\lambda}\f$ and
 * its derivatives \f$\left(\frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{x}},
 * \frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{u}}\right)\f$ are computed by
 * `ActionModelImpulseFwdDynamicsTpl`. These values are stored in a shared data (i.e.
 * DataCollectorImpulseTpl). Note that this residual function cannot be used with other action models.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`, `ActionModelImpulseFwdDynamicsTpl`,
 * `DataCollectorImpulseTpl`, `ActivationModelAbstractTpl`
 */
template <typename _Scalar>
class ResidualModelContactImpulseTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataContactImpulseTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef pinocchio::ForceTpl<Scalar> Force;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the contact impulse residual model
   *
   * @param[in] state  Multibody state
   * @param[in] id     Reference frame id
   * @param[in] fref   Reference spatial contact impulse in the contact coordinates
   */
  ResidualModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                 const Force& fref);
  virtual ~ResidualModelContactImpulseTpl();

  /**
   * @brief Compute the contact impulse residual
   *
   * The impulse vector is computed by ActionModelImpulseFwdDynamicsTpl and stored in DataCollectorImpulseTpl.
   *
   * @param[in] data  Contact impulse data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the contact impulse residual
   *
   * The impulse derivatives are computed by ActionModelImpulseFwdDynamicsTpl and stored in DataCollectorImpulseTpl.
   *
   * @param[in] data  Contact impulse data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the contact impulse residual data
   *
   * @param[in] data  shared data (it should be of type DataCollectorImpulseTpl)
   * @return the residual data.
   */
  virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Return the reference spatial contact impulse in the contact coordinates
   */
  const Force& get_reference() const;

  /**
   * @brief Modify the reference frame id
   */
  void set_id(pinocchio::FrameIndex id);

  /**
   * @brief Modify the reference spatial contact impulse in the contact coordinates
   */
  void set_reference(const Force& reference);

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 protected:
  pinocchio::FrameIndex id_;  //!< Reference frame id
  Force fref_;                //!< Reference spatial contact impulse in the contact coordinates
};

template <typename _Scalar>
struct ResidualDataContactImpulseTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ImpulseModelMultipleTpl<Scalar> ImpulseModelMultiple;
  typedef FrameForceTpl<Scalar> FrameForce;
  typedef StateMultibodyTpl<Scalar> StateMultibody;

  template <template <typename Scalar> class Model>
  ResidualDataContactImpulseTpl(Model<Scalar>* const model, DataCollectorAbstract* const data) : Base(model, data) {
    impulse_type = ImpulseUndefined;

    // Check that proper shared data has been passed
    DataCollectorImpulseTpl<Scalar>* d = dynamic_cast<DataCollectorImpulseTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorImpulse");
    }

    // Avoids data casting at runtime
    const pinocchio::FrameIndex id = model->get_id();
    const boost::shared_ptr<StateMultibody>& state = boost::static_pointer_cast<StateMultibody>(model->get_state());
    std::string frame_name = state->get_pinocchio()->frames[id].name;
    bool found_impulse = false;
    for (typename ImpulseModelMultiple::ImpulseDataContainer::iterator it = d->impulses->impulses.begin();
         it != d->impulses->impulses.end(); ++it) {
      if (it->second->frame == id) {
        ImpulseData3DTpl<Scalar>* d3d = dynamic_cast<ImpulseData3DTpl<Scalar>*>(it->second.get());
        if (d3d != NULL) {
          impulse_type = Impulse3D;
          model->set_nr(3);
          r.resize(3);
          Rx.resize(3, model->get_state()->get_ndx());
          Ru.resize(3, model->get_nu());
          found_impulse = true;
          impulse = it->second;
          break;
        }
        ImpulseData6DTpl<Scalar>* d6d = dynamic_cast<ImpulseData6DTpl<Scalar>*>(it->second.get());
        if (d6d != NULL) {
          impulse_type = Impulse6D;
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
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/contact-impulse.hxx"

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_CONTACT_IMPULSE_HPP_
