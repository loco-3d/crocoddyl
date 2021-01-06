///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_IMPULSE_FRICTION_CONE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_IMPULSE_FRICTION_CONE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/multibody/friction-cone.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Impulse friction cone cost
 *
 * This cost function defines a residual vector as \f$\mathbf{r}=\mathbf{A}\boldsymbol{\lambda}\f$, where
 * \f$\mathbf{A}\in~\mathbb{R}^{nr\times nc}\f$ describes the linearized friction cone,
 * \f$\boldsymbol{\lambda}\in~\mathbb{R}^{ni}\f$ is the spatial contact impulse computed by
 * `ActionModelImpulseFwdDynamicsTpl`, and `nr`, `ni` are the number of cone facets and dimension of the
 * impulse, respectively.
 *
 * Both cost and residual derivatives are computed analytically, where th force vector \f$\boldsymbol{\lambda}\f$ and
 * its derivatives \f$\left(\frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{x}},
 * \frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{u}}\right)\f$ are computed by
 * `ActionModelImpulseFwdDynamicsTpl`. These values are stored in a shared data (i.e.
 * DataCollectorImpulseTpl). Note that this cost function cannot be used with other action models.
 * For the computation of the cost Hessian, we use the Gauss-Newton approximation, e.g.
 * \f$\mathbf{l_{xu}} = \mathbf{l_{x}}^T \mathbf{l_{u}} \f$.
 *
 * As described in CostModelAbstractTpl(), the cost value and its derivatives are calculated by `calc` and `calcDiff`,
 * respectively.
 *
 * \sa `CostModelAbstractTpl`, `ActionModelImpulseFwdDynamicsTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelImpulseFrictionConeTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataImpulseFrictionConeTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef FrameFrictionConeTpl<Scalar> FrameFrictionCone;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixX3s MatrixX3s;

  /**
   * @brief Initialize the impulse friction cone cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] fref        Friction cone
   */
  CostModelImpulseFrictionConeTpl(boost::shared_ptr<StateMultibody> state,
                                  boost::shared_ptr<ActivationModelAbstract> activation,
                                  const FrameFrictionCone& fref);

  /**
   * @brief Initialize the impulse friction cone cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   *
   * @param[in] state  State of the multibody system
   * @param[in] fref   Friction cone
   */
  CostModelImpulseFrictionConeTpl(boost::shared_ptr<StateMultibody> state, const FrameFrictionCone& fref);
  virtual ~CostModelImpulseFrictionConeTpl();

  /**
   * @brief Compute the impulse friction cone cost
   *
   * @param[in] data  Impulse friction cone cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the impulse friction cone cost
   *
   * @param[in] data  Impulse friction cone cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the impulse friction cone cost data
   */
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

 protected:
  /**
   * @brief Modify the impulse friction cone reference
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Return the impulse friction cone reference
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv) const;

  using Base::activation_;
  using Base::state_;

 private:
  FrameFrictionCone fref_;  //!< Friction cone
};

template <typename _Scalar>
struct CostDataImpulseFrictionConeTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ImpulseModelMultipleTpl<Scalar> ImpulseModelMultiple;
  typedef FrameFrictionConeTpl<Scalar> FrameFrictionCone;
  typedef StateMultibodyTpl<Scalar> StateMultibody;

  template <template <typename Scalar> class Model>
  CostDataImpulseFrictionConeTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), more_than_3_constraints(false) {
    // Check that proper shared data has been passed
    DataCollectorImpulseTpl<Scalar>* d = dynamic_cast<DataCollectorImpulseTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorImpulse");
    }

    // Avoids data casting at runtime
    FrameFrictionCone fref = model->template get_reference<FrameFrictionCone>();
    const boost::shared_ptr<StateMultibody>& state = boost::static_pointer_cast<StateMultibody>(model->get_state());
    std::string frame_name = state->get_pinocchio()->frames[fref.id].name;
    bool found_impulse = false;
    for (typename ImpulseModelMultiple::ImpulseDataContainer::iterator it = d->impulses->impulses.begin();
         it != d->impulses->impulses.end(); ++it) {
      if (it->second->frame == fref.id) {
        ImpulseData3DTpl<Scalar>* d3d = dynamic_cast<ImpulseData3DTpl<Scalar>*>(it->second.get());
        if (d3d != NULL) {
          found_impulse = true;
          impulse = it->second;
          break;
        }
        ImpulseData6DTpl<Scalar>* d6d = dynamic_cast<ImpulseData6DTpl<Scalar>*>(it->second.get());
        if (d6d != NULL) {
          more_than_3_constraints = true;
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
  bool more_than_3_constraints;
  using Base::activation;
  using Base::cost;
  using Base::Lx;
  using Base::Lxx;
  using Base::r;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/impulse-friction-cone.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_IMPULSE_FRICTION_CONE_HPP_
