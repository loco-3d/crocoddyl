///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_COM_POSITION_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_COM_POSITION_HPP_

#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief CoM position residual
 *
 * This residual function defines the CoM tracking as
 * \f$\mathbf{r}=\mathbf{c}-\mathbf{c}^*\f$, where
 * \f$\mathbf{c},\mathbf{c}^*\in~\mathbb{R}^3\f$ are the current and reference
 * CoM position, respectively. Note that the dimension of the residual vector is
 * obtained from 3. Furthermore, the Jacobians of the residual function are
 * computed analytically.
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its
 * Jacobians are calculated by `calc` and `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelCoMPositionTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelCoMPositionTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataCoMPositionTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the CoM position residual model
   *
   * @param[in] state  State of the multibody system
   * @param[in] cref   Reference CoM position
   * @param[in] nu     Dimension of the control vector
   */
  ResidualModelCoMPositionTpl(std::shared_ptr<StateMultibody> state,
                              const Vector3s& cref, const std::size_t nu);

  /**
   * @brief Initialize the CoM position residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] cref   Reference CoM position
   */
  ResidualModelCoMPositionTpl(std::shared_ptr<StateMultibody> state,
                              const Vector3s& cref);
  virtual ~ResidualModelCoMPositionTpl() = default;

  /**
   * @brief Compute the CoM position residual
   *
   * @param[in] data  CoM position residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the derivatives of the CoM position residual
   *
   * @param[in] data  CoM position residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the com-position residual model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ResidualModelCoMPositionTpl<NewScalar> A residual model with the
   * new scalar type.
   */
  template <typename NewScalar>
  ResidualModelCoMPositionTpl<NewScalar> cast() const;

  /**
   * @brief Return the CoM position reference
   */
  const Vector3s& get_reference() const;

  /**
   * @brief Modify the CoM position reference
   */
  void set_reference(const Vector3s& cref);

  /**
   * @brief Print relevant information of the com-position residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::u_dependent_;
  using Base::v_dependent_;

 private:
  Vector3s cref_;  //!< Reference CoM position
};

template <typename _Scalar>
struct ResidualDataCoMPositionTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix3xs Matrix3xs;

  template <template <typename Scalar> class Model>
  ResidualDataCoMPositionTpl(Model<Scalar>* const model,
                             DataCollectorAbstract* const data)
      : Base(model, data) {
    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar>* d =
        dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty(
          "Invalid argument: the shared data should be derived from "
          "DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }
  virtual ~ResidualDataCoMPositionTpl() = default;

  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Pinocchio data
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/com-position.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ResidualModelCoMPositionTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ResidualDataCoMPositionTpl)

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_COM_POSITION_HPP_
