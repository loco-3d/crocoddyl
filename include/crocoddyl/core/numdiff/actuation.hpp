///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, LAAS-CNRS
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_ACTUATION_HPP_
#define CROCODDYL_CORE_NUMDIFF_ACTUATION_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

/**
 * @brief This class computes the numerical differentiation of an actuation
 * model.
 *
 * It computes the Jacobian of the residual model via numerical differentiation,
 * i.e., \f$\frac{\partial\boldsymbol{\tau}}{\partial\mathbf{x}}\f$ and
 * \f$\frac{\partial\boldsymbol{\tau}}{\partial\mathbf{u}}\f$ which denote the
 * Jacobians of the actuation function
 * \f$\boldsymbol{\tau}(\mathbf{x},\mathbf{u})\f$.
 *
 * \sa `ActuationModelAbstractTpl()`, `calcDiff()`
 */
template <typename _Scalar>
class ActuationModelNumDiffTpl : public ActuationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActuationModelBase, ActuationModelNumDiffTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationDataNumDiffTpl<Scalar> Data;
  typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the numdiff residual model
   *
   * @param model  Actuation model that we want to apply the numerical
   * differentiation
   */
  explicit ActuationModelNumDiffTpl(std::shared_ptr<Base> model);

  /**
   * @brief Destroy the numdiff actuation model
   */
  virtual ~ActuationModelNumDiffTpl() = default;

  /**
   * @brief @copydoc Base::calc()
   */
  virtual void calc(const std::shared_ptr<ActuationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calc(const std::shared_ptr<ActuationDataAbstract>&
   * data, const Eigen::Ref<const VectorXs>& x)
   */
  virtual void calc(const std::shared_ptr<ActuationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief @copydoc Base::calcDiff()
   */
  virtual void calcDiff(const std::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calcDiff(const
   * std::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const
   * VectorXs>& x)
   */
  virtual void calcDiff(const std::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief @copydoc Base::commands()
   */
  virtual void commands(const std::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& tau) override;

  /**
   * @brief @copydoc Base::torqueTransform()
   */
  virtual void torqueTransform(
      const std::shared_ptr<ActuationDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::createData()
   */
  virtual std::shared_ptr<ActuationDataAbstract> createData() override;

  template <typename NewScalar>
  ActuationModelNumDiffTpl<NewScalar> cast() const;

  /**
   * @brief Return the original actuation model
   */
  const std::shared_ptr<Base>& get_model() const;

  /**
   * @brief Return the disturbance constant used by the numerical
   * differentiation routine
   */
  const Scalar get_disturbance() const;

  /**
   * @brief Modify the disturbance constant used by the numerical
   * differentiation routine
   */
  void set_disturbance(const Scalar disturbance);

 private:
  std::shared_ptr<Base> model_;  //!< Actuation model hat we want to apply the
                                 //!< numerical differentiation
  Scalar e_jac_;  //!< Constant used for computing disturbances in Jacobian
                  //!< calculation

 protected:
  using Base::nu_;
};

template <typename _Scalar>
struct ActuationDataNumDiffTpl : public ActuationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef ActuationDataAbstractTpl<Scalar> Base;

  /**
   * @brief Initialize the numdiff actuation data
   *
   * @tparam Model is the type of the `ActuationModelAbstractTpl`.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <template <typename Scalar> class Model>
  explicit ActuationDataNumDiffTpl(Model<Scalar>* const model)
      : Base(model),
        dx(model->get_model()->get_state()->get_ndx()),
        du(model->get_model()->get_nu()),
        xp(model->get_model()->get_state()->get_nx()) {
    dx.setZero();
    du.setZero();
    xp.setZero();
    const std::size_t ndx = model->get_model()->get_state()->get_ndx();
    const std::size_t nu = model->get_model()->get_nu();
    data_0 = model->get_model()->createData();
    for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(model->get_model()->createData());
    }
    for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(model->get_model()->createData());
    }
  }

  Scalar x_norm;  //!< Norm of the state vector
  Scalar
      xh_jac;  //!< Disturbance value used for computing \f$ \ell_\mathbf{x} \f$
  Scalar
      uh_jac;  //!< Disturbance value used for computing \f$ \ell_\mathbf{u} \f$
  VectorXs dx;  //!< State disturbance
  VectorXs du;  //!< Control disturbance
  VectorXs xp;  //!< The integrated state from the disturbance on one DoF "\f$
                //!< \int x dx_i \f$"
  std::shared_ptr<Base> data_0;  //!< The data that contains the final results
  std::vector<std::shared_ptr<Base> >
      data_x;  //!< The temporary data associated with the state variation
  std::vector<std::shared_ptr<Base> >
      data_u;  //!< The temporary data associated with the control variation

  using Base::dtau_du;
  using Base::dtau_dx;
  using Base::tau;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/actuation.hxx"

#endif  // CROCODDYL_CORE_NUMDIFF_ACTUATION_HPP_
