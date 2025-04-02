///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_STATES_MULTIBODY_HPP_
#define CROCODDYL_MULTIBODY_STATES_MULTIBODY_HPP_

#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

/**
 * @brief State multibody representation
 *
 * A multibody state is described by the configuration point and its tangential
 * velocity, or in other words, by the generalized position and velocity
 * coordinates of a rigid-body system. For this state, we describe its
 * operators: difference, integrates, transport and their derivatives for any
 * Pinocchio model.
 *
 * For more details about these operators, please read the documentation of the
 * `StateAbstractTpl` class.
 *
 * \sa `diff()`, `integrate()`, `Jdiff()`, `Jintegrate()` and
 * `JintegrateTransport()`
 */
template <typename _Scalar>
class StateMultibodyTpl : public StateAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(StateBase, StateMultibodyTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateAbstractTpl<Scalar> Base;
  typedef pinocchio::ModelTpl<Scalar> PinocchioModel;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the multibody state
   *
   * @param[in] model  Pinocchio model
   */
  explicit StateMultibodyTpl(std::shared_ptr<PinocchioModel> model);
  StateMultibodyTpl();
  virtual ~StateMultibodyTpl();

  /**
   * @brief Generate a zero state.
   *
   * Note that the zero configuration is computed using `pinocchio::neutral`.
   */
  virtual VectorXs zero() const override;

  /**
   * @brief Generate a random state
   *
   * Note that the random configuration is computed using `pinocchio::random`
   * which satisfies the manifold definition (e.g., the quaterion definition)
   */
  virtual VectorXs rand() const override;

  virtual void diff(const Eigen::Ref<const VectorXs>& x0,
                    const Eigen::Ref<const VectorXs>& x1,
                    Eigen::Ref<VectorXs> dxout) const override;
  virtual void integrate(const Eigen::Ref<const VectorXs>& x,
                         const Eigen::Ref<const VectorXs>& dx,
                         Eigen::Ref<VectorXs> xout) const override;
  virtual void Jdiff(const Eigen::Ref<const VectorXs>&,
                     const Eigen::Ref<const VectorXs>&,
                     Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                     const Jcomponent firstsecond = both) const override;

  virtual void Jintegrate(const Eigen::Ref<const VectorXs>& x,
                          const Eigen::Ref<const VectorXs>& dx,
                          Eigen::Ref<MatrixXs> Jfirst,
                          Eigen::Ref<MatrixXs> Jsecond,
                          const Jcomponent firstsecond = both,
                          const AssignmentOp = setto) const override;
  virtual void JintegrateTransport(const Eigen::Ref<const VectorXs>& x,
                                   const Eigen::Ref<const VectorXs>& dx,
                                   Eigen::Ref<MatrixXs> Jin,
                                   const Jcomponent firstsecond) const override;

  /**
   * @brief Return the Pinocchio model (i.e., model of the rigid body system)
   */
  const std::shared_ptr<PinocchioModel>& get_pinocchio() const;

  template <typename NewScalar>
  StateMultibodyTpl<NewScalar> cast() const;

  /**
   * @brief Print relevant information of the state multibody
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::has_limits_;
  using Base::lb_;
  using Base::ndx_;
  using Base::nq_;
  using Base::nv_;
  using Base::nx_;
  using Base::ub_;

 private:
  std::shared_ptr<PinocchioModel> pinocchio_;  //!< Pinocchio model
  VectorXs x0_;                                //!< Zero state
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/states/multibody.hxx"

extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    crocoddyl::StateMultibodyTpl<double>;
extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    crocoddyl::StateMultibodyTpl<float>;

#endif  // CROCODDYL_MULTIBODY_STATES_MULTIBODY_HPP_
