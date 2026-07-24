///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_PARAMS_INERTIAL_HPP_
#define CROCODDYL_MULTIBODY_PARAMS_INERTIAL_HPP_

#include <algorithm>
#include <iostream>
#include <limits>
#include <pinocchio/algorithm/regressor.hpp>
#include <pinocchio/spatial/inertia.hpp>

#include "crocoddyl/core/dynamics-base.hpp"
#include "crocoddyl/core/params-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/params/inertial-parametrization-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Parameter model for selected multibody inertias
 *
 * Each selected Pinocchio body contributes exactly ten consecutive dynamics
 * parameters in the order supplied to the constructor. Joint names and frame
 * names are accepted and resolved to unique non-universe parent joints. An
 * update changes only those inertias in the shared StateMultibody model and
 * stores the conversion Jacobians in caller-owned data. The torque regressor
 * maps Pinocchio's standard inertial regressor through these Jacobians. Every
 * coordinate is unbounded by default.
 *
 * The selected-body layout can change before manager registration. A real
 * layout change invalidates existing parameter, manager and dynamics data;
 * remove a registered item before changing its layout, re-add it afterwards,
 * and recreate all associated data.
 */
template <typename _Scalar>
class MultibodyInertialParamsTpl : public DynamicsParamsAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DynamicsParamsAbstractTpl<Scalar> Base;
  typedef ParamsDataAbstractTpl<Scalar> ParamsDataAbstract;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef DataCollectorMultibodyTpl<Scalar> DataCollectorMultibody;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef InertialParametrizationAbstractTpl<Scalar>
      InertialParametrizationAbstract;
  typedef MultibodyInertialParamsDataTpl<Scalar> MultibodyInertialParamsData;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  static const std::size_t kParametersPerBody = 10;

  CROCODDYL_DERIVED_CAST(ParamsModelBase, MultibodyInertialParamsTpl)

  /** @brief Identify every non-universe body in Pinocchio joint order. */
  MultibodyInertialParamsTpl(
      std::shared_ptr<StateMultibody> state,
      std::shared_ptr<InertialParametrizationAbstract> parametrization);
  /** @brief Identify the ordered set of joint or frame body names. */
  MultibodyInertialParamsTpl(
      std::shared_ptr<StateMultibody> state,
      std::shared_ptr<InertialParametrizationAbstract> parametrization,
      const std::vector<std::string>& body_names);
  virtual ~MultibodyInertialParamsTpl() = default;

  /** @brief Update selected inertias and their parameter Jacobians. */
  virtual void update(const std::shared_ptr<ParamsDataAbstract>& data,
                      const Eigen::Ref<const VectorXs>& p) override;

  /** @brief Compute \f$\partial\tau/\partial p\f$ at the supplied acceleration.
   */
  virtual void computeJointTorqueRegressor(
      const std::shared_ptr<DynamicsDataAbstract>& data,
      const std::shared_ptr<ParamsDataAbstract>& params,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override;

  /** @brief Allocate specialized data returned through the base pointer type.
   */
  virtual std::shared_ptr<ParamsDataAbstract> createData() override;

  /** @brief Check specialized type, dimensions and parametrization workspace.
   */
  virtual bool checkData(
      const std::shared_ptr<ParamsDataAbstract>& data) const override;

  /** @brief Return parameters representing the current selected inertias. */
  virtual VectorXs zero() const override;
  /** @brief Return a random smooth parameter vector. */
  virtual VectorXs rand() const override;

  /**
   * @brief Activate or deactivate a body in the parameter layout
   *
   * Joint and frame names resolve to their canonical non-universe parent
   * joint. Activation appends a new unbounded ten-parameter block;
   * deactivation removes its block while preserving all other ordering.
   * Repeated requests are no-ops. Missing names and the universe warn without
   * modifying the layout.
   *
   * @warning Call before manager registration, or remove the model from its
   * manager first and re-add it afterwards. Recreate all data after a real
   * layout change.
   */
  void changeBodyStatus(const std::string& body_name, const bool active);

  /** @brief Return the shared mutable inertial parametrization model. */
  const std::shared_ptr<InertialParametrizationAbstract>& get_parametrization()
      const;
  /** @brief Return selected Pinocchio joint ids in parameter-block order. */
  const std::vector<pinocchio::JointIndex>& get_joint_ids() const;
  /** @brief Return canonical joint names in parameter-block order. */
  std::vector<std::string> get_body_names() const;

  /** @brief Cast state, parametrization, body selection and bounds. */
  template <typename NewScalar>
  MultibodyInertialParamsTpl<NewScalar> cast() const;

  /** @brief Print dimensions, selection count and parametrization. */
  virtual void print(std::ostream& os) const override;

 protected:
  std::shared_ptr<MultibodyInertialParamsData> castData(
      const std::shared_ptr<ParamsDataAbstract>& data) const;

  static std::vector<std::string> createDefaultBodyNames(
      const std::shared_ptr<StateMultibody>& state);
  std::vector<pinocchio::JointIndex> resolveBodyIds(
      const std::vector<std::string>& body_names) const;

  std::shared_ptr<StateMultibody> state_multibody_;
  std::shared_ptr<InertialParametrizationAbstract> parametrization_;
  std::vector<pinocchio::JointIndex> joint_ids_;
  std::size_t layout_version_;

  friend struct MultibodyInertialParamsDataTpl<Scalar>;
};

/**
 * @brief Data for MultibodyInertialParamsTpl
 *
 * The inherited payload owns \f$p\f$ and the \f$nv\times np\f$ torque
 * regressor in the dynamics partition. This data additionally owns one
 * physical vector and conversion Jacobian per selected body and a single
 * reusable parametrization workspace. Torque-regressor calculations use the
 * non-owning DataCollectorMultibodyTpl attached to the dynamics data. Copies
 * own independent numerical storage and an independently allocated
 * parametrization workspace while retaining the shared parametrization model.
 */
template <typename _Scalar>
struct MultibodyInertialParamsDataTpl
    : public DynamicsParamsDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DynamicsParamsDataAbstractTpl<Scalar> Base;
  typedef InertialParametrizationAbstractTpl<Scalar>
      InertialParametrizationAbstract;
  typedef InertialParametrizationDataAbstractTpl<Scalar>
      InertialParametrizationDataAbstract;
  typedef Eigen::Matrix<Scalar, 10, 1> Vector10s;
  typedef Eigen::Matrix<Scalar, 10, 10> Matrix10s;
  typedef std::vector<Vector10s, Eigen::aligned_allocator<Vector10s> >
      Vector10Vector;
  typedef std::vector<Matrix10s, Eigen::aligned_allocator<Matrix10s> >
      Matrix10Vector;

  static_assert(Vector10s::RowsAtCompileTime == 10 &&
                    Vector10s::ColsAtCompileTime == 1,
                "Vector10s must preserve its fixed 10x1 layout");
  static_assert(Matrix10s::RowsAtCompileTime == 10 &&
                    Matrix10s::ColsAtCompileTime == 10,
                "Matrix10s must preserve its fixed 10x10 layout");

  /**
   * @brief Allocate data from a non-null compatible model
   *
   * The model supplies dimensions, layout generation and the retained shared
   * parametrization model used to allocate independent scratch on copies.
   *
   * @throw crocoddyl::Exception if `model` is null
   */
  template <template <typename Scalar> class Model>
  explicit MultibodyInertialParamsDataTpl(Model<Scalar>* const model)
      : Base(checkModel(model)->get_state(), checkModel(model)->get_np()),
        psi(model->get_joint_ids().size()),
        dpsi_dp(model->get_joint_ids().size()),
        parametrization(model->get_parametrization()->createData()),
        parametrization_model_(model->get_parametrization()),
        layout_version_(model->layout_version_) {
    setZero();
  }
  MultibodyInertialParamsDataTpl(const MultibodyInertialParamsDataTpl& other)
      : Base(other),
        psi(other.psi),
        dpsi_dp(other.dpsi_dp),
        parametrization(other.parametrization_model_->createData()),
        parametrization_model_(other.parametrization_model_),
        layout_version_(other.layout_version_) {}
  MultibodyInertialParamsDataTpl& operator=(
      const MultibodyInertialParamsDataTpl&) = delete;
  virtual ~MultibodyInertialParamsDataTpl() = default;

  /** @brief Reset inherited and inertial numerical storage without resizing. */
  virtual void setZero() override {
    Base::setZero();
    for (typename Vector10Vector::iterator it = psi.begin(); it != psi.end();
         ++it) {
      it->setZero();
    }
    for (typename Matrix10Vector::iterator it = dpsi_dp.begin();
         it != dpsi_dp.end(); ++it) {
      it->setZero();
    }
  }

  Vector10Vector psi;      //!< Ten-element physical vectors by body.
  Matrix10Vector dpsi_dp;  //!< Ten-by-ten conversion Jacobians by body.
  std::shared_ptr<InertialParametrizationDataAbstract>
      parametrization;  //!< Owned reusable parametrization workspace.

 private:
  template <template <typename Scalar> class Model>
  static Model<Scalar>* checkModel(Model<Scalar>* const model) {
    if (model == nullptr) {
      throw_pretty("Invalid argument: model is null");
    }
    return model;
  }

  std::shared_ptr<InertialParametrizationAbstract> parametrization_model_;
  std::size_t layout_version_;

  friend class MultibodyInertialParamsTpl<Scalar>;
};

}  // namespace crocoddyl

#include "crocoddyl/multibody/params/inertial.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::MultibodyInertialParamsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::MultibodyInertialParamsDataTpl)

#endif  // CROCODDYL_MULTIBODY_PARAMS_INERTIAL_HPP_
