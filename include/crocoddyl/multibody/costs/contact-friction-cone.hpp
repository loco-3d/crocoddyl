///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTACT_FRICTION_CONE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTACT_FRICTION_CONE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Contact friction cone cost
 *
 * This cost function defines a residual vector as \f$\mathbf{r}=\mathbf{A}\boldsymbol{\lambda}\f$, where
 * \f$\mathbf{A}\in~\mathbb{R}^{nr\times nc}\f$ describes the linearized friction cone,
 * \f$\boldsymbol{\lambda}\in~\mathbb{R}^{nc}\f$ is the spatial contact forces computed by
 * `DifferentialActionModelContactFwdDynamicsTpl`, and `nr`, `nc` are the number of cone facets and dimension of the
 * contact, respectively.
 *
 * Both cost and residual derivatives are computed analytically, where th force vector \f$\boldsymbol{\lambda}\f$ and
 * its derivatives \f$\left(\frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{x}},
 * \frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{u}}\right)\f$ are computed by
 * `DifferentialActionModelContactFwdDynamicsTpl`. These values are stored in a shared data (i.e.
 * DataCollectorContactTpl). Note that this cost function cannot be used with other action models.
 * For the computation of the cost Hessian, we use the Gauss-Newton approximation, e.g.
 * \f$\mathbf{l_{xu}} = \mathbf{l_{x}}^T \mathbf{l_{u}} \f$.
 *
 * As described in `CostModelResidualTpl()`, the cost value and its derivatives are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `CostModelResidualTpl`, `DifferentialActionModelContactFwdDynamicsTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelContactFrictionConeTpl : public CostModelResidualTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelResidualTpl<Scalar> Base;
  typedef CostDataContactFrictionConeTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ResidualModelContactFrictionConeTpl<Scalar> ResidualModelContactFrictionCone;
  typedef FrameFrictionConeTpl<Scalar> FrameFrictionCone;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the contact friction cone cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] fref        Friction cone
   * @param[in] nu          Dimension of the control vector
   */
  CostModelContactFrictionConeTpl(boost::shared_ptr<StateMultibody> state,
                                  boost::shared_ptr<ActivationModelAbstract> activation, const FrameFrictionCone& fref,
                                  const std::size_t nu);

  /**
   * @brief Initialize the contact friction cone cost model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] fref        Friction cone
   */
  CostModelContactFrictionConeTpl(boost::shared_ptr<StateMultibody> state,
                                  boost::shared_ptr<ActivationModelAbstract> activation,
                                  const FrameFrictionCone& fref);

  /**
   * @brief Initialize the contact friction cone cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   *
   * @param[in] state  State of the multibody system
   * @param[in] fref   Friction cone
   * @param[in] nu     Dimension of the control vector
   */
  CostModelContactFrictionConeTpl(boost::shared_ptr<StateMultibody> state, const FrameFrictionCone& fref,
                                  const std::size_t nu);

  /**
   * @brief Initialize the contact friction cone cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] fref   Friction cone
   */
  CostModelContactFrictionConeTpl(boost::shared_ptr<StateMultibody> state, const FrameFrictionCone& fref);
  virtual ~CostModelContactFrictionConeTpl();

 protected:
  /**
   * @brief Modify the contact friction cone reference
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Return the contact friction cone reference
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv);

  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;

 private:
  FrameFrictionCone fref_;  //!< Friction cone
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/contact-friction-cone.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTACT_FRICTION_CONE_HPP_
