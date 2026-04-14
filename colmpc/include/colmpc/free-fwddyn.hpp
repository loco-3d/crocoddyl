#ifndef COLMPC_ACTIONS_FREE_FWDDYN_HPP_
#define COLMPC_ACTIONS_FREE_FWDDYN_HPP_

#include <stdexcept>

#ifdef PINOCCHIO_WITH_CPPAD_SUPPORT  // TODO(cmastalli): Removed after merging
                                     // Pinocchio v.2.4.8
#include <pinocchio/codegen/cppadcg.hpp>
#endif

#include "colmpc/data/geometry-data.hpp"
#include "colmpc/multibody.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"

namespace colmpc {

template <typename _Scalar>
class DifferentialActionModelFreeFwdDynamicsTpl
    : public crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef crocoddyl::DifferentialActionModelFreeFwdDynamicsTpl<Scalar> Base;
  typedef DifferentialActionDataFreeFwdDynamicsTpl<Scalar> Data;
  typedef crocoddyl::DifferentialActionDataAbstractTpl<Scalar>
      DifferentialActionDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef crocoddyl::ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef crocoddyl::ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef crocoddyl::MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  DifferentialActionModelFreeFwdDynamicsTpl(
      std::shared_ptr<StateMultibody> state,
      std::shared_ptr<ActuationModelAbstract> actuation,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints = nullptr);
  virtual ~DifferentialActionModelFreeFwdDynamicsTpl();

  /**
   * @brief Compute the system acceleration, and cost value
   *
   * It computes the system acceleration using the free forward-dynamics.
   *
   * @param[in] data  Free forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc Base::calc(const
   * std::shared_ptr<DifferentialActionDataAbstract>& data, const
   * Eigen::Ref<const VectorXs>& x)
   */
  virtual void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the free forward-dynamics data
   *
   * @return free forward-dynamics data
   */
  virtual std::shared_ptr<DifferentialActionDataAbstract> createData();

  /**
   * @brief Check that the given data belongs to the free forward-dynamics data
   */
  virtual bool checkData(
      const std::shared_ptr<DifferentialActionDataAbstract>& data);

  /**
   * @brief Return the Pinocchio model
   */
  inline pinocchio::GeometryModel& get_geometry() const { return geometry_; }

  /**
   * @brief Print relevant information of the free forward-dynamics model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::g_lb_;   //!< Lower bound of the inequality constraints
  using Base::g_ub_;   //!< Upper bound of the inequality constraints
  using Base::nu_;     //!< Control dimension
  using Base::state_;  //!< Model of the state

 private:
  pinocchio::GeometryModel& geometry_;  //!< Pinocchio model
};

template <typename _Scalar>
struct DifferentialActionDataFreeFwdDynamicsTpl
    : crocoddyl::DifferentialActionDataFreeFwdDynamicsTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef crocoddyl::DifferentialActionDataFreeFwdDynamicsTpl<Scalar> Base;

  explicit DifferentialActionDataFreeFwdDynamicsTpl(
      DifferentialActionModelFreeFwdDynamicsTpl<Scalar>* const model);

  pinocchio::GeometryData geometry;
};

extern template class DifferentialActionModelFreeFwdDynamicsTpl<double>;
extern template class DifferentialActionDataFreeFwdDynamicsTpl<double>;

}  // namespace colmpc

#endif  // COLMPC_ACTIONS_FREE_FWDDYN_HPP_
