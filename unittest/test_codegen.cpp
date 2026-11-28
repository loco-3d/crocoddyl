///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <cmath>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/container/aligned-vector.hpp>
#include <pinocchio/multibody/sample-models.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <type_traits>

#include "crocoddyl/core/actions/lqr.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/activations/weighted-quadratic-barrier.hpp"
#include "crocoddyl/core/codegen/action.hpp"
#include "crocoddyl/core/codegen/observer.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/integrator/discretized.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/integrator/rk.hpp"
#include "crocoddyl/core/integrator/time.hpp"
#include "crocoddyl/core/numdiff/observer.hpp"
#include "crocoddyl/core/observer/euler.hpp"
#include "crocoddyl/core/optctrl/observation.hpp"
#include "crocoddyl/core/params/integrator-timeopt.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/residuals/parameters.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/actuations/joint-friction.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/dynamics/constrained-forward.hpp"
#include "crocoddyl/multibody/dynamics/impulse-forward.hpp"
#include "crocoddyl/multibody/implicit-constraints/contact.hpp"
#include "crocoddyl/multibody/implicit-constraints/multiple-implicit-constraints.hpp"
#include "crocoddyl/multibody/params/actuation.hpp"
#include "crocoddyl/multibody/params/inertial.hpp"
#include "crocoddyl/multibody/params/log-cholesky.hpp"
#include "crocoddyl/multibody/residuals/centroidal-momentum.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/frame-rotation.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/residuals/frame-velocity.hpp"
#include "crocoddyl/multibody/residuals/potential-energy.hpp"
#include "crocoddyl/multibody/residuals/power.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "factory/solver.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace crocoddyl::unittest;

template <typename MatrixLike>
void check_codegen_matrix_approx(const MatrixLike& lhs, const MatrixLike& rhs,
                                 const typename MatrixLike::Scalar tol,
                                 const std::string& name) {
  BOOST_REQUIRE_EQUAL(lhs.rows(), rhs.rows());
  BOOST_REQUIRE_EQUAL(lhs.cols(), rhs.cols());
  if (lhs.size() == 0) {
    BOOST_TEST_MESSAGE(name << " empty block");
    return;
  }
  const typename MatrixLike::Scalar max_err = (lhs - rhs).cwiseAbs().maxCoeff();
  BOOST_TEST_MESSAGE(name << " max abs err = " << max_err);
  BOOST_CHECK(lhs.isApprox(rhs, tol));
}

template <typename MatrixLike>
void check_codegen_matrix_abs_approx(const MatrixLike& lhs,
                                     const MatrixLike& rhs,
                                     const typename MatrixLike::Scalar tol,
                                     const std::string& name) {
  BOOST_REQUIRE_EQUAL(lhs.rows(), rhs.rows());
  BOOST_REQUIRE_EQUAL(lhs.cols(), rhs.cols());
  if (lhs.size() == 0) {
    BOOST_TEST_MESSAGE(name << " empty block");
    return;
  }
  const typename MatrixLike::Scalar max_err = (lhs - rhs).cwiseAbs().maxCoeff();
  BOOST_TEST_MESSAGE(name << " max abs err = " << max_err);
  BOOST_CHECK_MESSAGE(max_err <= tol,
                      name << " max abs err " << max_err << " exceeds " << tol);
}

template <typename MatrixLike>
void check_codegen_matrix_finite(const MatrixLike& value,
                                 const std::string& name) {
  BOOST_TEST_MESSAGE(name << " finite check");
  BOOST_CHECK(value.allFinite());
}

template <typename ModelTpl, typename Scalar>
typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs assemble_running_cost_hessian(
    const std::shared_ptr<ModelTpl>& model,
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>& data) {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  const std::size_t ndx = model->get_state()->get_ndx();
  const std::size_t nu = model->get_nu();
  const std::size_t np = model->get_np();
  MatrixXs H = MatrixXs::Zero(ndx + nu + np, ndx + nu + np);
  H.block(0, 0, ndx, ndx) = data->Lxx;
  H.block(0, ndx, ndx, nu) = data->Lxu;
  H.block(ndx, 0, nu, ndx) = data->Lxu.transpose();
  H.block(ndx, ndx, nu, nu) = data->Luu;
  H.block(ndx + nu, ndx + nu, np, np) = data->Lpp;
  H.block(ndx + nu, 0, np, ndx) = data->Lpx;
  H.block(0, ndx + nu, ndx, np) = data->Lpx.transpose();
  H.block(ndx + nu, ndx, np, nu) = data->Lpu;
  H.block(ndx, ndx + nu, nu, np) = data->Lpu.transpose();
  return H;
}

template <typename ModelTpl, typename Scalar>
typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs
assemble_terminal_cost_hessian(
    const std::shared_ptr<ModelTpl>& model,
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>& data) {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  const std::size_t ndx = model->get_state()->get_ndx();
  const std::size_t np = model->get_np();
  MatrixXs H = MatrixXs::Zero(ndx + np, ndx + np);
  H.block(0, 0, ndx, ndx) = data->Lxx;
  H.block(ndx, ndx, np, np) = data->Lpp;
  H.block(ndx, 0, np, ndx) = data->Lpx;
  H.block(0, ndx, ndx, np) = data->Lpx.transpose();
  return H;
}

template <typename ModelTpl, typename Scalar>
Scalar evaluate_running_cost_tangent(
    const std::shared_ptr<ModelTpl>& model,
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>& data,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        x,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        u,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        p,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        z) {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  const std::size_t ndx = model->get_state()->get_ndx();
  const std::size_t nu = model->get_nu();
  const std::size_t np = model->get_np();
  VectorXs x_eval(model->get_state()->get_nx());
  VectorXs u_eval(nu);
  model->get_state()->integrate(x, z.head(ndx), x_eval);
  u_eval = u + z.segment(ndx, nu);
  if (np != 0u) {
    const VectorXs p_eval = p + z.tail(np);
    model->update_p(data, p_eval);
  }
  model->calc(data, x_eval, u_eval);
  return data->cost;
}

template <typename ModelTpl, typename Scalar>
Scalar evaluate_terminal_cost_tangent(
    const std::shared_ptr<ModelTpl>& model,
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>& data,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        x,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        p,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        z) {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  const std::size_t ndx = model->get_state()->get_ndx();
  const std::size_t np = model->get_np();
  VectorXs x_eval(model->get_state()->get_nx());
  model->get_state()->integrate(x, z.head(ndx), x_eval);
  if (np != 0u) {
    const VectorXs p_eval = p + z.tail(np);
    model->update_p(data, p_eval);
  }
  model->calc(data, x_eval);
  return data->cost;
}

template <typename ModelTpl, typename Scalar>
typename crocoddyl::MathBaseTpl<Scalar>::VectorXs
finite_difference_running_cost_gradient(
    const std::shared_ptr<ModelTpl>& model,
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>& data,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        x,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        u,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        p,
    const Scalar h) {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  const std::size_t n =
      model->get_state()->get_ndx() + model->get_nu() + model->get_np();
  VectorXs grad = VectorXs::Zero(n);
  VectorXs z = VectorXs::Zero(n);
  for (std::size_t i = 0; i < n; ++i) {
    z.setZero();
    z[i] = h;
    const Scalar fp = evaluate_running_cost_tangent(model, data, x, u, p, z);
    z[i] = -h;
    const Scalar fm = evaluate_running_cost_tangent(model, data, x, u, p, z);
    grad[i] = (fp - fm) / (Scalar(2.) * h);
  }
  if (model->get_np() != 0u) {
    model->update_p(data, p);
  }
  return grad;
}

template <typename ModelTpl, typename Scalar>
typename crocoddyl::MathBaseTpl<Scalar>::VectorXs
finite_difference_terminal_cost_gradient(
    const std::shared_ptr<ModelTpl>& model,
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>& data,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        x,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        p,
    const Scalar h) {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  const std::size_t n = model->get_state()->get_ndx() + model->get_np();
  VectorXs grad = VectorXs::Zero(n);
  VectorXs z = VectorXs::Zero(n);
  for (std::size_t i = 0; i < n; ++i) {
    z.setZero();
    z[i] = h;
    const Scalar fp = evaluate_terminal_cost_tangent(model, data, x, p, z);
    z[i] = -h;
    const Scalar fm = evaluate_terminal_cost_tangent(model, data, x, p, z);
    grad[i] = (fp - fm) / (Scalar(2.) * h);
  }
  if (model->get_np() != 0u) {
    model->update_p(data, p);
  }
  return grad;
}

template <typename ModelTpl, typename Scalar>
typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs
finite_difference_running_cost_hessian(
    const std::shared_ptr<ModelTpl>& model,
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>& data,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        x,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        u,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        p,
    const Scalar h) {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  const std::size_t n =
      model->get_state()->get_ndx() + model->get_nu() + model->get_np();
  MatrixXs H = MatrixXs::Zero(n, n);
  VectorXs zi = VectorXs::Zero(n);
  VectorXs zj = VectorXs::Zero(n);
  const Scalar f0 = evaluate_running_cost_tangent(model, data, x, u, p, zi);
  for (std::size_t i = 0; i < n; ++i) {
    zi.setZero();
    zi[i] = h;
    const Scalar fp = evaluate_running_cost_tangent(model, data, x, u, p, zi);
    zi[i] = -h;
    const Scalar fm = evaluate_running_cost_tangent(model, data, x, u, p, zi);
    H(i, i) = (fp - Scalar(2.) * f0 + fm) / (h * h);
    for (std::size_t j = i + 1; j < n; ++j) {
      zi.setZero();
      zj.setZero();
      zi[i] = h;
      zj[j] = h;
      VectorXs zsum = zi + zj;
      const Scalar fpp =
          evaluate_running_cost_tangent(model, data, x, u, p, zsum);
      zj[j] = -h;
      zsum = zi + zj;
      const Scalar fpm =
          evaluate_running_cost_tangent(model, data, x, u, p, zsum);
      zi[i] = -h;
      zj[j] = h;
      zsum = zi + zj;
      const Scalar fmp =
          evaluate_running_cost_tangent(model, data, x, u, p, zsum);
      zj[j] = -h;
      zsum = zi + zj;
      const Scalar fmm =
          evaluate_running_cost_tangent(model, data, x, u, p, zsum);
      H(i, j) = (fpp - fpm - fmp + fmm) / (Scalar(4.) * h * h);
      H(j, i) = H(i, j);
    }
  }
  if (model->get_np() != 0u) {
    model->update_p(data, p);
  }
  return H;
}

template <typename ModelTpl, typename Scalar>
typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs
finite_difference_terminal_cost_hessian(
    const std::shared_ptr<ModelTpl>& model,
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>& data,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        x,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        p,
    const Scalar h) {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  const std::size_t n = model->get_state()->get_ndx() + model->get_np();
  MatrixXs H = MatrixXs::Zero(n, n);
  VectorXs zi = VectorXs::Zero(n);
  VectorXs zj = VectorXs::Zero(n);
  const Scalar f0 = evaluate_terminal_cost_tangent(model, data, x, p, zi);
  for (std::size_t i = 0; i < n; ++i) {
    zi.setZero();
    zi[i] = h;
    const Scalar fp = evaluate_terminal_cost_tangent(model, data, x, p, zi);
    zi[i] = -h;
    const Scalar fm = evaluate_terminal_cost_tangent(model, data, x, p, zi);
    H(i, i) = (fp - Scalar(2.) * f0 + fm) / (h * h);
    for (std::size_t j = i + 1; j < n; ++j) {
      zi.setZero();
      zj.setZero();
      zi[i] = h;
      zj[j] = h;
      VectorXs zsum = zi + zj;
      const Scalar fpp =
          evaluate_terminal_cost_tangent(model, data, x, p, zsum);
      zj[j] = -h;
      zsum = zi + zj;
      const Scalar fpm =
          evaluate_terminal_cost_tangent(model, data, x, p, zsum);
      zi[i] = -h;
      zj[j] = h;
      zsum = zi + zj;
      const Scalar fmp =
          evaluate_terminal_cost_tangent(model, data, x, p, zsum);
      zj[j] = -h;
      zsum = zi + zj;
      const Scalar fmm =
          evaluate_terminal_cost_tangent(model, data, x, p, zsum);
      H(i, j) = (fpp - fpm - fmp + fmm) / (Scalar(4.) * h * h);
      H(j, i) = H(i, j);
    }
  }
  if (model->get_np() != 0u) {
    model->update_p(data, p);
  }
  return H;
}

/// \brief Changing the environment variables in a autodiff model. This function
/// needs to be passed to the ActionModelCodeGen in order to make the calc and
/// calcdiff be dependent on
///        some parameter of the action model (like a cost reference). Inside
///        the function definition, set the env_vector where you want it to be
///        defined inside ad_model.
/// \param[in,out] ad_model    the ActionModelCodeGen that needs to be recorded
/// \param[in]     env_vector  the environment vector which would be set in
/// ad_model.
template <typename Scalar>
void change_env(
    std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>> ad_model,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        env_vector) {
  typedef typename crocoddyl::ResidualModelFrameTranslationTpl<Scalar>
      ResidualModelFrameTranslation;
  typedef typename crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar>
      DynamicsModelConstrainedForward;

  crocoddyl::IntegratedActionModelEulerTpl<Scalar>* m =
      static_cast<crocoddyl::IntegratedActionModelEulerTpl<Scalar>*>(
          ad_model.get());
  DynamicsModelConstrainedForward* md =
      static_cast<DynamicsModelConstrainedForward*>(m->get_dynamics().get());
  std::shared_ptr<ResidualModelFrameTranslation> residual =
      std::static_pointer_cast<ResidualModelFrameTranslation>(
          m->get_costs()
              ->get_costs()
              .find("gripperTrans")
              ->second->cost->get_residual());
  residual->set_id(md->get_pinocchio().getFrameId("gripper_left_joint"));
  residual->set_reference(env_vector);
}

template <typename Scalar>
const std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>>
build_arm_action_model() {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector3s Vector3s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Matrix3s Matrix3s;
  typedef typename crocoddyl::CostModelAbstractTpl<Scalar> CostModelAbstract;
  typedef typename crocoddyl::CostModelResidualTpl<Scalar> CostModelResidual;
  typedef typename crocoddyl::ResidualModelFramePlacementTpl<Scalar>
      ResidualModelFramePlacement;
  typedef typename crocoddyl::ResidualModelFrameTranslationTpl<Scalar>
      ResidualModelFrameTranslation;
  typedef typename crocoddyl::ResidualModelFrameRotationTpl<Scalar>
      ResidualModelFrameRotation;
  typedef typename crocoddyl::ResidualModelFrameVelocityTpl<Scalar>
      ResidualModelFrameVelocity;
  typedef typename crocoddyl::ResidualModelStateTpl<Scalar> ResidualModelState;
  typedef typename crocoddyl::ResidualModelControlTpl<Scalar>
      ResidualModelControl;
  typedef typename crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef typename crocoddyl::ActionModelAbstractTpl<Scalar>
      ActionModelAbstract;
  typedef typename crocoddyl::ActuationModelMultibodyTpl<Scalar>
      ActuationModelMultibody;
  typedef typename crocoddyl::ConstraintModelManagerTpl<Scalar>
      ConstraintModelManager;
  typedef typename crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar>
      DynamicsModelConstrainedForward;
  typedef typename crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar>
      ImplicitConstraintModelMultiple;
  typedef typename crocoddyl::IntegratedActionModelEulerTpl<Scalar>
      IntegratedActionModelEuler;
  typedef typename crocoddyl::IntegratorTimeTpl<Scalar> IntegratorTime;

  typedef typename crocoddyl::ActivationBoundsTpl<Scalar> ActivationBounds;
  typedef typename crocoddyl::ActivationModelQuadraticBarrierTpl<Scalar>
      ActivationModelQuadraticBarrier;
  typedef typename crocoddyl::ActivationModelWeightedQuadraticBarrierTpl<Scalar>
      ActivationModelWeightedQuadraticBarrier;

  // because urdf is not supported with all scalar types.
  pinocchio::ModelTpl<double> modeld;
  pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR
                              "/talos_data/robots/talos_left_arm.urdf",
                              modeld);
  pinocchio::srdf::loadReferenceConfigurations(
      modeld, EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf",
      false);

  pinocchio::ModelTpl<Scalar> model_full(modeld.cast<Scalar>()), model;
  std::vector<pinocchio::JointIndex> locked_joints{5, 6, 7};
  pinocchio::buildReducedModel(model_full, locked_joints,
                               VectorXs::Zero(model_full.nq), model);

  std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar>> state =
      std::make_shared<crocoddyl::StateMultibodyTpl<Scalar>>(
          std::make_shared<pinocchio::ModelTpl<Scalar>>(model));

  std::shared_ptr<CostModelAbstract> goalTrackingCost =
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelFramePlacement>(
                     state, model.getFrameId("gripper_left_joint"),
                     pinocchio::SE3Tpl<Scalar>(
                         Matrix3s::Identity(),
                         Vector3s(Scalar(0), Scalar(0), Scalar(.4)))));
  std::shared_ptr<CostModelAbstract> goalTranslationCost =
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelFrameTranslation>(
                     state, model.getFrameId("gripper_left_joint"),
                     Vector3s(Scalar(0), Scalar(0), Scalar(.4))));
  std::shared_ptr<CostModelAbstract> goalRotationCost =
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelFrameRotation>(
                     state, model.getFrameId("gripper_left_joint"),
                     Matrix3s::Identity()));
  std::shared_ptr<CostModelAbstract> goalVelocityCost =
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelFrameVelocity>(
                     state, model.getFrameId("gripper_left_joint"),
                     pinocchio::MotionTpl<Scalar>(
                         Vector3s(Scalar(0), Scalar(0), Scalar(.4)),
                         Vector3s(Scalar(0), Scalar(0), Scalar(.4))),
                     pinocchio::ReferenceFrame::LOCAL));
  std::shared_ptr<CostModelAbstract> xRegCost =
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelState>(state));
  std::shared_ptr<CostModelAbstract> uRegCost =
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelControl>(state));

  // Create a cost model per the running and terminal action model.
  std::shared_ptr<CostModelSum> runningCostModel =
      std::make_shared<CostModelSum>(state);

  VectorXs lowlim = (model.lowerPositionLimit);
  VectorXs uplim = (model.upperPositionLimit);
  VectorXs xlb(model.nq + model.nv), xub(model.nq + model.nv);
  xlb << lowlim, -VectorXs::Ones(model.nv);
  xub << uplim, VectorXs::Ones(model.nv);

  // xlb.tail(model.nv) *= Scalar(-1) * std::numeric_limits<Scalar>::max();
  // xub.tail(model.nv) *= std::numeric_limits<Scalar>::max();

  VectorXs xweights(model.nv + model.nv);
  xweights.head(model.nv).fill(Scalar(10.));
  xweights.tail(model.nv).fill(Scalar(100.));

  ActivationBounds bounds(xlb, xub);
  std::shared_ptr<ActivationModelQuadraticBarrier> activation_bounded =
      std::make_shared<ActivationModelQuadraticBarrier>(bounds);
  std::shared_ptr<ActivationModelWeightedQuadraticBarrier>
      weighted_activation_bounded =
          std::make_shared<ActivationModelWeightedQuadraticBarrier>(bounds,
                                                                    xweights);

  std::shared_ptr<CostModelAbstract> jointLimitCost =
      std::make_shared<CostModelResidual>(
          state, activation_bounded,
          std::make_shared<ResidualModelState>(state));

  std::shared_ptr<CostModelAbstract> jointLimitCost2 =
      std::make_shared<CostModelResidual>(
          state, weighted_activation_bounded,
          std::make_shared<ResidualModelState>(state));

  // Then let's added the running and terminal cost functions
  runningCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));
  runningCostModel->addCost("gripperTrans", goalTranslationCost, Scalar(1));
  runningCostModel->addCost("gripperRot", goalRotationCost, Scalar(1));
  runningCostModel->addCost("gripperVel", goalVelocityCost, Scalar(1));
  runningCostModel->addCost("jointLim", jointLimitCost, Scalar(1e3));
  runningCostModel->addCost("jointLim2", jointLimitCost2, Scalar(1e3));
  runningCostModel->addCost("xReg", xRegCost, Scalar(1e-4));
  runningCostModel->addCost("uReg", uRegCost, Scalar(1e-4));

  // We define an actuation model
  std::shared_ptr<ActuationModelMultibody> actuation =
      std::make_shared<ActuationModelMultibody>(state);

  std::shared_ptr<ImplicitConstraintModelMultiple> constraints_dynamics =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  std::shared_ptr<DynamicsModelConstrainedForward> runningDynamics =
      std::make_shared<DynamicsModelConstrainedForward>(state, actuation,
                                                        constraints_dynamics);
  std::shared_ptr<ActionModelAbstract> runningModel =
      std::make_shared<IntegratedActionModelEuler>(
          runningDynamics, runningCostModel,
          std::shared_ptr<ConstraintModelManager>(),
          std::shared_ptr<
              crocoddyl::ControlParametrizationModelAbstractTpl<Scalar>>(),
          std::make_shared<IntegratorTime>(Scalar(1e-3)));
  return runningModel;
}

template <typename Scalar>
const std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>>
build_bipedal_action_model() {
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector2s Vector2s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector3s Vector3s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector6s Vector6s;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Matrix3s Matrix3s;
  typedef typename crocoddyl::CostModelAbstractTpl<Scalar> CostModelAbstract;
  typedef typename crocoddyl::CostModelResidualTpl<Scalar> CostModelResidual;
  typedef typename crocoddyl::ResidualModelFramePlacementTpl<Scalar>
      ResidualModelFramePlacement;
  typedef typename crocoddyl::ResidualModelStateTpl<Scalar> ResidualModelState;
  typedef typename crocoddyl::ResidualModelControlTpl<Scalar>
      ResidualModelControl;
  typedef typename crocoddyl::ResidualModelCoMPositionTpl<Scalar>
      ResidualModelCoMPosition;
  typedef typename crocoddyl::ResidualModelContactForceTpl<Scalar>
      ResidualModelContactForce;
  typedef typename crocoddyl::ResidualModelCentroidalMomentumTpl<Scalar>
      ResidualModelCentroidalMomentum;
  typedef typename crocoddyl::ContactModelTpl<Scalar> ContactModel;
  typedef typename crocoddyl::ConstraintModelManagerTpl<Scalar>
      ConstraintModelManager;
  typedef typename crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef typename crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar>
      DynamicsModelConstrainedForward;
  typedef typename crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar>
      ImplicitConstraintModelMultiple;
  typedef typename crocoddyl::ActionModelAbstractTpl<Scalar>
      ActionModelAbstract;
  typedef typename crocoddyl::ActuationModelFloatingBaseTpl<Scalar>
      ActuationModelFloatingBase;
  typedef typename crocoddyl::IntegratedActionModelEulerTpl<Scalar>
      IntegratedActionModelEuler;
  typedef typename crocoddyl::IntegratorTimeTpl<Scalar> IntegratorTime;

  const std::string RF = "leg_right_6_joint";
  const std::string LF = "leg_left_6_joint";

  pinocchio::ModelTpl<double> modeld;
  pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR
                              "/talos_data/robots/talos_reduced.urdf",
                              pinocchio::JointModelFreeFlyer(), modeld);
  modeld.lowerPositionLimit.head<7>().array() = -1;
  modeld.upperPositionLimit.head<7>().array() = 1.;
  pinocchio::srdf::loadReferenceConfigurations(
      modeld, EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/srdf/talos.srdf",
      false);

  pinocchio::ModelTpl<Scalar> model(modeld.cast<Scalar>());
  std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar>> state =
      std::make_shared<crocoddyl::StateMultibodyTpl<Scalar>>(
          std::make_shared<pinocchio::ModelTpl<Scalar>>(model));

  std::shared_ptr<ActuationModelFloatingBase> actuation =
      std::make_shared<ActuationModelFloatingBase>(state);

  std::shared_ptr<CostModelAbstract> goalTrackingCost =
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelFramePlacement>(
                     state, model.getFrameId("arm_right_7_joint"),
                     pinocchio::SE3Tpl<Scalar>(
                         Matrix3s::Identity(),
                         Vector3s(Scalar(.0), Scalar(.0), Scalar(.4))),
                     actuation->get_nu()));
  std::shared_ptr<CostModelAbstract> centroidalCost =
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelCentroidalMomentum>(
                     state, Vector6s::Zero(), actuation->get_nu()));
  std::shared_ptr<CostModelAbstract> comCost =
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelCoMPosition>(
                     state, Vector3s::Zero(), actuation->get_nu()));
  std::shared_ptr<CostModelAbstract> contactForceCost =
      std::make_shared<CostModelResidual>(
          state,
          std::make_shared<ResidualModelContactForce>(
              state, model.getFrameId(RF), pinocchio::ForceTpl<Scalar>::Zero(),
              6, actuation->get_nu()));
  std::shared_ptr<CostModelAbstract> xRegCost =
      std::make_shared<CostModelResidual>(
          state,
          std::make_shared<ResidualModelState>(state, actuation->get_nu()));
  std::shared_ptr<CostModelAbstract> uRegCost =
      std::make_shared<CostModelResidual>(
          state,
          std::make_shared<ResidualModelControl>(state, actuation->get_nu()));

  // Create a cost model per the running and terminal action model.
  std::shared_ptr<CostModelSum> runningCostModel =
      std::make_shared<CostModelSum>(state, actuation->get_nu());

  // Then let's added the running and terminal cost functions
  runningCostModel->addCost("gripperPose", goalTrackingCost, Scalar(1));
  runningCostModel->addCost("xReg", xRegCost, Scalar(1e-4));
  runningCostModel->addCost("uReg", uRegCost, Scalar(1e-4));
  runningCostModel->addCost("contactforce", contactForceCost, Scalar(1e-4));
  runningCostModel->addCost("comcost", comCost, Scalar(1e-4));
  runningCostModel->addCost("centroidal", centroidalCost, Scalar(1e-4));

  std::shared_ptr<ImplicitConstraintModelMultiple> contact_models =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  typename ContactModel::MaskArray mask6d = {
      {true, true, true, true, true, true}};
  std::shared_ptr<ContactModel> support_contact_model6D =
      std::make_shared<ContactModel>(
          state, model.getFrameId(RF), pinocchio::SE3Tpl<Scalar>::Identity(),
          pinocchio::LOCAL_WORLD_ALIGNED, actuation->get_nu(),
          Vector2s(Scalar(0.), Scalar(50.)), mask6d);
  contact_models->addConstraint(
      model.frames[model.getFrameId(RF)].name + "_contact",
      support_contact_model6D);

  typename ContactModel::MaskArray mask3d = {
      {true, true, true, false, false, false}};
  std::shared_ptr<ContactModel> support_contact_model3D =
      std::make_shared<ContactModel>(
          state, model.getFrameId(LF), pinocchio::SE3Tpl<Scalar>::Identity(),
          pinocchio::LOCAL_WORLD_ALIGNED, actuation->get_nu(),
          Vector2s(Scalar(0.), Scalar(50.)), mask3d);
  contact_models->addConstraint(
      model.frames[model.getFrameId(LF)].name + "_contact",
      support_contact_model3D);

  std::shared_ptr<DynamicsModelConstrainedForward> runningDynamics =
      std::make_shared<DynamicsModelConstrainedForward>(state, actuation,
                                                        contact_models);
  std::shared_ptr<ActionModelAbstract> runningModel =
      std::make_shared<IntegratedActionModelEuler>(
          runningDynamics, runningCostModel,
          std::shared_ptr<ConstraintModelManager>(),
          std::shared_ptr<
              crocoddyl::ControlParametrizationModelAbstractTpl<Scalar>>(),
          std::make_shared<IntegratorTime>(Scalar(1e-3)));
  return runningModel;
}

template <typename Scalar>
const std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>>
build_freeflyer_manifold_action_model() {
  typedef crocoddyl::ActuationModelMultibodyTpl<Scalar> ActuationModelMultibody;
  typedef crocoddyl::CostModelResidualTpl<Scalar> CostModelResidual;
  typedef crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar>
      DynamicsModelConstrainedForward;
  typedef crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar>
      ImplicitConstraintModelMultiple;
  typedef crocoddyl::IntegratedActionModelEulerTpl<Scalar>
      IntegratedActionModelEuler;
  typedef crocoddyl::IntegratorTimeTpl<Scalar> IntegratorTime;
  typedef crocoddyl::ResidualModelControlTpl<Scalar> ResidualModelControl;
  typedef crocoddyl::ResidualModelStateTpl<Scalar> ResidualModelState;

  pinocchio::ModelTpl<double> modeld;
  const pinocchio::JointIndex root_id =
      modeld.addJoint(0, pinocchio::JointModelFreeFlyer(),
                      pinocchio::SE3::Identity(), "root_joint");
  modeld.appendBodyToJoint(root_id, pinocchio::Inertia::FromBox(1., .2, .3, .4),
                           pinocchio::SE3::Identity());
  const pinocchio::JointIndex joint_id = modeld.addJoint(
      root_id, pinocchio::JointModelRY(), pinocchio::SE3::Identity(), "joint1");
  modeld.appendBodyToJoint(joint_id,
                           pinocchio::Inertia::FromBox(.7, .1, .2, .3),
                           pinocchio::SE3::Identity());
  modeld.lowerPositionLimit.setConstant(-1.);
  modeld.upperPositionLimit.setConstant(1.);

  pinocchio::ModelTpl<Scalar> model(modeld.cast<Scalar>());
  std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar>> state =
      std::make_shared<crocoddyl::StateMultibodyTpl<Scalar>>(
          std::make_shared<pinocchio::ModelTpl<Scalar>>(model));
  std::shared_ptr<ActuationModelMultibody> actuation =
      std::make_shared<ActuationModelMultibody>(state);
  std::shared_ptr<ImplicitConstraintModelMultiple> dyn_constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  std::shared_ptr<DynamicsModelConstrainedForward> dynamics =
      std::make_shared<DynamicsModelConstrainedForward>(state, actuation,
                                                        dyn_constraints);
  std::shared_ptr<CostModelSum> costs =
      std::make_shared<CostModelSum>(state, actuation->get_nu());
  costs->addCost("xReg",
                 std::make_shared<CostModelResidual>(
                     state, std::make_shared<ResidualModelState>(
                                state, actuation->get_nu())),
                 Scalar(1.));
  costs->addCost("uReg",
                 std::make_shared<CostModelResidual>(
                     state, std::make_shared<ResidualModelControl>(
                                state, actuation->get_nu())),
                 Scalar(1e-2));
  return std::make_shared<IntegratedActionModelEuler>(
      dynamics, costs,
      std::shared_ptr<crocoddyl::ConstraintModelManagerTpl<Scalar>>(),
      std::shared_ptr<
          crocoddyl::ControlParametrizationModelAbstractTpl<Scalar>>(),
      std::make_shared<IntegratorTime>(Scalar(1e-3)));
}

pinocchio::ModelTpl<double> build_codegen_freeflyer_model() {
  pinocchio::ModelTpl<double> model;
  const pinocchio::JointIndex root_id =
      model.addJoint(0, pinocchio::JointModelFreeFlyer(),
                     pinocchio::SE3::Identity(), "root_joint");
  model.appendBodyToJoint(root_id, pinocchio::Inertia::FromBox(1., .2, .3, .4),
                          pinocchio::SE3::Identity());
  const pinocchio::JointIndex joint_id = model.addJoint(
      root_id, pinocchio::JointModelRY(), pinocchio::SE3::Identity(), "joint1");
  model.appendBodyToJoint(joint_id, pinocchio::Inertia::FromBox(.7, .1, .2, .3),
                          pinocchio::SE3::Identity());
  model.lowerPositionLimit.setConstant(-1.);
  model.upperPositionLimit.setConstant(1.);
  return model;
}

template <typename Scalar>
std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar>>
build_codegen_state_from_model(const pinocchio::ModelTpl<double>& modeld) {
  return std::make_shared<crocoddyl::StateMultibodyTpl<Scalar>>(
      std::make_shared<pinocchio::ModelTpl<Scalar>>(
          modeld.template cast<Scalar>()));
}

template <typename Scalar>
std::shared_ptr<crocoddyl::ParameterManagerTpl<Scalar>>
build_codegen_inertial_params(
    const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar>>& state) {
  typedef crocoddyl::LogCholeskyParametrizationTpl<Scalar>
      LogCholeskyParametrization;
  typedef crocoddyl::MultibodyInertialParamsTpl<Scalar> MultibodyInertialParams;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const std::shared_ptr<LogCholeskyParametrization> parametrization =
      std::make_shared<LogCholeskyParametrization>();
  const std::vector<std::string> body_names(
      state->get_pinocchio()->names.begin() + 1,
      state->get_pinocchio()->names.begin() + 2);
  VectorXs p_seed(10);
  p_seed << Scalar(0.2), Scalar(-0.1), Scalar(0.15), Scalar(-0.2), Scalar(0.1),
      Scalar(-0.25), Scalar(0.3), Scalar(0.05), Scalar(-0.08), Scalar(0.12);
  const std::shared_ptr<
      typename LogCholeskyParametrization::InertialParametrizationDataAbstract>
      data = parametrization->createData();
  VectorXs psi(10);
  parametrization->fromParametrization(data, psi, p_seed);
  state->get_pinocchio()
      ->inertias[state->get_pinocchio()->getJointId(body_names[0])] =
      pinocchio::InertiaTpl<Scalar>::FromDynamicParameters(psi);

  const std::shared_ptr<MultibodyInertialParams> inertia =
      std::make_shared<MultibodyInertialParams>(state, parametrization,
                                                body_names);
  const std::shared_ptr<crocoddyl::ParameterManagerTpl<Scalar>> params =
      std::make_shared<crocoddyl::ParameterManagerTpl<Scalar>>(state);
  params->addParam("inertia", inertia);
  return params;
}

template <typename Scalar>
std::shared_ptr<crocoddyl::ActuationModelMultibodyTpl<Scalar>>
build_codegen_friction_actuation(
    const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar>>& state) {
  typedef crocoddyl::ActuationModelMultibodyTpl<Scalar> ActuationModel;
  typedef crocoddyl::JointDynamicsModelAbstractTpl<Scalar> JointModel;
  typedef crocoddyl::JointDynamicsModelFrictionTpl<Scalar> FrictionModel;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const std::shared_ptr<pinocchio::ModelTpl<Scalar>>& pin_model =
      state->get_pinocchio();
  pinocchio::JointIndex joint_id = 1;
  for (; joint_id < static_cast<pinocchio::JointIndex>(pin_model->njoints);
       ++joint_id) {
    if (pin_model->joints[joint_id].nv() == 1) {
      break;
    }
  }
  BOOST_REQUIRE_LT(joint_id,
                   static_cast<pinocchio::JointIndex>(pin_model->njoints));
  VectorXs mu(3);
  using std::log;
  mu << log(Scalar(0.15)), log(Scalar(3.)), log(Scalar(0.2));
  std::vector<std::shared_ptr<JointModel>> joints;
  joints.push_back(std::make_shared<FrictionModel>(
      joint_id, static_cast<std::size_t>(pin_model->joints[joint_id].nq()), mu,
      crocoddyl::JointFrictionType::CoulombViscous));
  return std::make_shared<ActuationModel>(state, joints);
}

template <typename Scalar>
struct CodegenParameterizedActionCaseTpl {
  typedef crocoddyl::ActionModelAbstractTpl<Scalar> ActionModelAbstract;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  std::string name;
  std::shared_ptr<ActionModelAbstract> model;
  std::shared_ptr<ParameterManager> params;
  VectorXs p;
};

template <typename Scalar>
void build_codegen_parameter_terms(
    const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar>>& state,
    const std::size_t nu,
    const std::shared_ptr<crocoddyl::ParameterManagerTpl<Scalar>>& params,
    std::shared_ptr<crocoddyl::CostModelSumTpl<Scalar>>& costs,
    std::shared_ptr<crocoddyl::ConstraintModelManagerTpl<Scalar>>&
        constraints) {
  typedef crocoddyl::ConstraintModelResidualTpl<Scalar> ConstraintModelResidual;
  typedef crocoddyl::CostModelResidualTpl<Scalar> CostModelResidual;
  typedef crocoddyl::ResidualModelParametersTpl<Scalar> ResidualModelParameters;
  typedef crocoddyl::ResidualModelStateTpl<Scalar> ResidualModelState;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const std::shared_ptr<ResidualModelParameters> parameter_residual =
      std::make_shared<ResidualModelParameters>(state, params->zero(), nu);
  const std::shared_ptr<ResidualModelState> state_residual =
      std::make_shared<ResidualModelState>(state, state->zero(), nu);
  costs = std::make_shared<crocoddyl::CostModelSumTpl<Scalar>>(
      state, nu, params->get_np());
  costs->addCost("parameters",
                 std::make_shared<CostModelResidual>(state, parameter_residual),
                 Scalar(0.7));
  constraints = std::make_shared<crocoddyl::ConstraintModelManagerTpl<Scalar>>(
      state, nu, params->get_np());
  constraints->addConstraint("a_inactive",
                             std::make_shared<ConstraintModelResidual>(
                                 state, parameter_residual, true),
                             false);
  constraints->addConstraint(
      "b_inequality",
      std::make_shared<ConstraintModelResidual>(
          state, parameter_residual, -VectorXs::Ones(params->get_np()),
          VectorXs::Ones(params->get_np()), true));
  constraints->addConstraint("c_equality",
                             std::make_shared<ConstraintModelResidual>(
                                 state, parameter_residual, true));
  constraints->addConstraint(
      "d_terminal_inequality",
      std::make_shared<ConstraintModelResidual>(
          state, state_residual, -VectorXs::Ones(state->get_ndx()),
          VectorXs::Ones(state->get_ndx()), true));
  constraints->addConstraint(
      "e_terminal_equality",
      std::make_shared<ConstraintModelResidual>(state, state_residual, true));
}

template <typename Scalar>
CodegenParameterizedActionCaseTpl<Scalar>
build_codegen_parameterized_integrator(
    const pinocchio::ModelTpl<double>& modeld, const std::string& name,
    const crocoddyl::RKType rk_type) {
  typedef crocoddyl::ActuationModelMultibodyTpl<Scalar> ActuationModel;
  typedef crocoddyl::ActuationMultibodyParamsTpl<Scalar> ActuationParams;
  typedef crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar> DynamicsModel;
  typedef crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar>
      ImplicitConstraintModelMultiple;
  typedef crocoddyl::IntegratedActionModelEulerTpl<Scalar> Euler;
  typedef crocoddyl::IntegratedActionModelRKTpl<Scalar> RK;
  typedef crocoddyl::IntegratorTimeTpl<Scalar> IntegratorTime;
  typedef crocoddyl::IntegratorTimeoptParamsTpl<Scalar> TimeParams;
  typedef crocoddyl::LQRParamsTpl<Scalar> LQRParams;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  CodegenParameterizedActionCaseTpl<Scalar> result;
  result.name = name;
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar>> state =
      build_codegen_state_from_model<Scalar>(modeld);
  const std::shared_ptr<ActuationModel> actuation =
      build_codegen_friction_actuation(state);
  const std::shared_ptr<ImplicitConstraintModelMultiple> implicit =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  const std::shared_ptr<DynamicsModel> dynamics =
      std::make_shared<DynamicsModel>(state, actuation, implicit);
  const std::shared_ptr<IntegratorTime> integrator_time =
      std::make_shared<IntegratorTime>(Scalar(0.012), true);
  result.params = build_codegen_inertial_params(state);
  result.params->addParam("actuation",
                          std::make_shared<ActuationParams>(actuation));
  result.params->addParam("a_inactive", std::make_shared<LQRParams>(state, 1),
                          false);
  result.params->addParam("b_time",
                          std::make_shared<TimeParams>(state, integrator_time));
  std::shared_ptr<crocoddyl::CostModelSumTpl<Scalar>> costs;
  std::shared_ptr<crocoddyl::ConstraintModelManagerTpl<Scalar>> constraints;
  build_codegen_parameter_terms(state, actuation->get_nu(), result.params,
                                costs, constraints);
  if (name == "euler") {
    result.model = std::make_shared<Euler>(dynamics, costs, constraints,
                                           nullptr, integrator_time);
  } else {
    result.model = std::make_shared<RK>(dynamics, costs, constraints, nullptr,
                                        integrator_time, rk_type);
  }
  result.p =
      VectorXs::LinSpaced(result.params->get_np(), Scalar(-0.03), Scalar(0.04));
  using std::log;
  result.p[0] = log(Scalar(0.017));
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> data =
      result.model->createData(result.params->createData());
  result.model->set_params(data, result.params);
  result.model->update_p(data, result.p);
  return result;
}

template <typename Scalar>
CodegenParameterizedActionCaseTpl<Scalar>
build_codegen_parameterized_discretized(
    const pinocchio::ModelTpl<double>& modeld) {
  typedef crocoddyl::ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef crocoddyl::DiscretizedActionModelTpl<Scalar> Discretized;
  typedef crocoddyl::DynamicsModelImpulseForwardTpl<Scalar> ImpulseDynamics;
  typedef crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar>
      ImplicitConstraintModelMultiple;
  typedef crocoddyl::ContactModelTpl<Scalar> ContactModel;
  typedef crocoddyl::LQRParamsTpl<Scalar> LQRParams;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  CodegenParameterizedActionCaseTpl<Scalar> result;
  result.name = "discretized";
  pinocchio::ModelTpl<double> impulse_modeld = modeld;
  const pinocchio::JointIndex contact_joint =
      impulse_modeld.getJointId("joint1");
  impulse_modeld.addFrame(pinocchio::Frame("contact", contact_joint,
                                           pinocchio::SE3::Identity(),
                                           pinocchio::OP_FRAME));
  const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar>> state =
      build_codegen_state_from_model<Scalar>(impulse_modeld);
  const std::shared_ptr<ImplicitConstraintModelMultiple> implicit =
      std::make_shared<ImplicitConstraintModelMultiple>(state, 0);
  typename ContactModel::MaskArray mask = {
      {true, true, true, false, false, false}};
  const pinocchio::FrameIndex frame_id = static_cast<pinocchio::FrameIndex>(
      state->get_pinocchio()->frames.size() - 1);
  const typename ContactModel::Vector2s gains = ContactModel::Vector2s::Zero();
  implicit->addConstraint(
      "contact", std::make_shared<ContactModel>(
                     state, frame_id, pinocchio::SE3Tpl<Scalar>::Identity(),
                     pinocchio::LOCAL_WORLD_ALIGNED, 0, gains, mask));
  const std::shared_ptr<ImpulseDynamics> dynamics =
      std::make_shared<ImpulseDynamics>(state, implicit);
  result.params = build_codegen_inertial_params(state);
  result.params->addParam("a_inactive", std::make_shared<LQRParams>(state, 1),
                          false);
  std::shared_ptr<CostModelSum> costs;
  std::shared_ptr<ConstraintModelManager> constraints;
  build_codegen_parameter_terms(state, 0, result.params, costs, constraints);
  result.model = std::make_shared<Discretized>(dynamics, costs, constraints);
  result.p = result.params->zero();
  result.p += VectorXs::LinSpaced(result.p.size(), Scalar(-0.02), Scalar(0.03));
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> data =
      result.model->createData(result.params->createData());
  result.model->set_params(data, result.params);
  result.model->update_p(data, result.p);
  return result;
}

template <typename Scalar>
std::shared_ptr<crocoddyl::ObserverModelAbstractTpl<Scalar>>
build_codegen_observer_model(
    const pinocchio::ModelTpl<double>& modeld,
    std::shared_ptr<crocoddyl::ParameterManagerTpl<Scalar>>& params_out) {
  typedef crocoddyl::ActuationModelMultibodyTpl<Scalar> ActuationModelMultibody;
  typedef crocoddyl::ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef crocoddyl::ConstraintModelResidualTpl<Scalar> ConstraintModelResidual;
  typedef crocoddyl::CostModelResidualTpl<Scalar> CostModelResidual;
  typedef crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar>
      DynamicsModelConstrainedForward;
  typedef crocoddyl::ImplicitConstraintModelMultipleTpl<Scalar>
      ImplicitConstraintModelMultiple;
  typedef crocoddyl::IntegratedObserverModelEulerTpl<Scalar>
      IntegratedObserverModelEuler;
  typedef crocoddyl::ResidualModelControlTpl<Scalar> ResidualModelControl;
  typedef crocoddyl::ResidualModelParametersTpl<Scalar> ResidualModelParameters;
  typedef crocoddyl::ResidualModelPowerTpl<Scalar> ResidualModelPower;
  typedef crocoddyl::ResidualModelStateTpl<Scalar> ResidualModelState;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const std::shared_ptr<crocoddyl::StateMultibodyTpl<Scalar>> state =
      build_codegen_state_from_model<Scalar>(modeld);
  const std::shared_ptr<ActuationModelMultibody> actuation =
      std::make_shared<ActuationModelMultibody>(state);
  const std::shared_ptr<ImplicitConstraintModelMultiple> dyn_constraints =
      std::make_shared<ImplicitConstraintModelMultiple>(state,
                                                        actuation->get_nu());
  const std::shared_ptr<DynamicsModelConstrainedForward> dynamics =
      std::make_shared<DynamicsModelConstrainedForward>(state, actuation,
                                                        dyn_constraints);
  params_out = build_codegen_inertial_params(state);

  const std::size_t observer_nu = state->get_ndx() + dynamics->get_nu();
  const std::shared_ptr<CostModelSum> costs =
      std::make_shared<CostModelSum>(state, observer_nu, params_out->get_np());
  costs->addCost("xReg",
                 std::make_shared<CostModelResidual>(
                     state, std::make_shared<ResidualModelState>(
                                state, state->zero(), observer_nu)),
                 Scalar(1.));
  costs->addCost(
      "uReg",
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelControl>(state, observer_nu)),
      Scalar(1.));
  costs->addCost("pReg",
                 std::make_shared<CostModelResidual>(
                     state, std::make_shared<ResidualModelParameters>(
                                state, params_out->zero(), observer_nu)),
                 Scalar(1.));
  costs->addCost("powerObs",
                 std::make_shared<CostModelResidual>(
                     state, std::make_shared<ResidualModelPower>(
                                state, observer_nu, params_out->get_np(),
                                Scalar(0.2), "inertia")),
                 Scalar(1.));

  const std::shared_ptr<ConstraintModelManager> constraints =
      std::make_shared<ConstraintModelManager>(state, observer_nu,
                                               params_out->get_np());
  const std::shared_ptr<ResidualModelParameters> p_residual =
      std::make_shared<ResidualModelParameters>(state, params_out->zero(),
                                                observer_nu);
  constraints->addConstraint(
      "parameter_equality",
      std::make_shared<ConstraintModelResidual>(state, p_residual));
  constraints->addConstraint(
      "parameter_inequality",
      std::make_shared<ConstraintModelResidual>(
          state, p_residual, -VectorXs::Ones(params_out->get_np()),
          VectorXs::Ones(params_out->get_np())));
  const std::shared_ptr<ResidualModelState> state_constraint =
      std::make_shared<ResidualModelState>(state, state->zero(), observer_nu);
  constraints->addConstraint(
      "state_inequality",
      std::make_shared<ConstraintModelResidual>(
          state, state_constraint, -VectorXs::Ones(state->get_ndx()),
          VectorXs::Ones(state->get_ndx())));
  constraints->addConstraint(
      "state_equality",
      std::make_shared<ConstraintModelResidual>(state, state_constraint));

  const std::shared_ptr<IntegratedObserverModelEuler> observer =
      std::make_shared<IntegratedObserverModelEuler>(dynamics, costs,
                                                     constraints);
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> setup_data =
      observer->createData(params_out->createData());
  observer->set_params(setup_data, params_out);
  return observer;
}

template <typename Scalar>
void set_codegen_observer_state_environment(
    const std::shared_ptr<crocoddyl::ObserverModelAbstractTpl<Scalar>>& model,
    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>&
        env,
    const std::string& state_observation_cost, const std::string& weight_cost) {
  typedef crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef crocoddyl::IntegratedObserverModelAbstractTpl<Scalar>
      IntegratedObserverModelAbstract;
  typedef crocoddyl::ResidualModelPowerTpl<Scalar> ResidualModelPower;
  typedef crocoddyl::ResidualModelStateTpl<Scalar> ResidualModelState;

  const std::shared_ptr<IntegratedObserverModelAbstract> integrated =
      std::dynamic_pointer_cast<IntegratedObserverModelAbstract>(model);
  BOOST_REQUIRE(integrated != nullptr);

  const std::shared_ptr<CostModelSum>& costs = integrated->get_costs();
  typename CostModelSum::CostModelContainer::const_iterator state_it =
      costs->get_costs().find(state_observation_cost);
  BOOST_REQUIRE(state_it != costs->get_costs().end());

  const std::shared_ptr<ResidualModelState> residual =
      std::dynamic_pointer_cast<ResidualModelState>(
          state_it->second->cost->get_residual());
  BOOST_REQUIRE(residual != nullptr);

  const std::size_t nx = residual->get_state()->get_nx();
  const std::size_t expected = nx + (weight_cost.empty() ? 0u : 1u);
  const std::size_t env_size = static_cast<std::size_t>(env.size());
  BOOST_REQUIRE(env_size == expected || env_size == expected + 1u);
  residual->set_reference(env.head(nx));
  if (!weight_cost.empty()) {
    typename CostModelSum::CostModelContainer::const_iterator weight_it =
        costs->get_costs().find(weight_cost);
    BOOST_REQUIRE(weight_it != costs->get_costs().end());
    weight_it->second->weight = env[nx];
  }
  if (env_size == expected + 1u) {
    typename CostModelSum::CostModelContainer::const_iterator power_it =
        costs->get_costs().find("powerObs");
    BOOST_REQUIRE(power_it != costs->get_costs().end());
    const std::shared_ptr<ResidualModelPower> residual_power =
        std::dynamic_pointer_cast<ResidualModelPower>(
            power_it->second->cost->get_residual());
    BOOST_REQUIRE(residual_power != nullptr);
    residual_power->set_reference(env[expected]);
  }
}

template <typename Scalar>
void compare_codegen_observer_data(
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>&
        codegen_data,
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>&
        direct_data,
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>&
        numdiff_data,
    const Scalar tol, const std::string& prefix,
    const bool check_cost_gradients = true) {
  const std::shared_ptr<crocoddyl::ObserverDataAbstractTpl<Scalar>>
      codegen_observer_data =
          std::dynamic_pointer_cast<crocoddyl::ObserverDataAbstractTpl<Scalar>>(
              codegen_data);
  const std::shared_ptr<crocoddyl::ObserverDataAbstractTpl<Scalar>>
      direct_observer_data =
          std::dynamic_pointer_cast<crocoddyl::ObserverDataAbstractTpl<Scalar>>(
              direct_data);
  BOOST_REQUIRE(codegen_observer_data != nullptr);
  BOOST_REQUIRE(direct_observer_data != nullptr);

  const std::shared_ptr<crocoddyl::ObserverDataAbstractTpl<Scalar>>
      numdiff_observer_data =
          std::dynamic_pointer_cast<crocoddyl::ObserverDataAbstractTpl<Scalar>>(
              numdiff_data);
  BOOST_REQUIRE(numdiff_observer_data != nullptr);

  check_codegen_matrix_approx(codegen_data->xnext, direct_data->xnext, tol,
                              prefix + " xnext");
  const Scalar cost_tol =
      std::is_same<Scalar, float>::value ? Scalar(5e-5) : Scalar(1e-9);
  BOOST_CHECK_LE(std::abs(codegen_data->cost - direct_data->cost), cost_tol);
  check_codegen_matrix_approx(codegen_data->g, direct_data->g, tol,
                              prefix + " g");
  check_codegen_matrix_approx(codegen_data->h, direct_data->h, tol,
                              prefix + " h");
  check_codegen_matrix_approx(codegen_data->Fx, direct_data->Fx, tol,
                              prefix + " Fx");
  check_codegen_matrix_approx(codegen_data->Fu, direct_data->Fu, tol,
                              prefix + " Fu");
  check_codegen_matrix_approx(codegen_data->Fp, direct_data->Fp, tol,
                              prefix + " Fp");
  if (check_cost_gradients) {
    check_codegen_matrix_abs_approx(codegen_data->Lx, numdiff_data->Lx, tol,
                                    prefix + " Lx");
    check_codegen_matrix_abs_approx(codegen_data->Lu, numdiff_data->Lu, tol,
                                    prefix + " Lu");
    check_codegen_matrix_abs_approx(codegen_data->Lp, numdiff_data->Lp, tol,
                                    prefix + " Lp");
  }
  check_codegen_matrix_finite(codegen_data->Lxx, prefix + " Lxx");
  check_codegen_matrix_finite(codegen_data->Lxu, prefix + " Lxu");
  check_codegen_matrix_finite(codegen_data->Luu, prefix + " Luu");
  check_codegen_matrix_finite(codegen_data->Lpp, prefix + " Lpp");
  check_codegen_matrix_finite(codegen_data->Lpx, prefix + " Lpx");
  check_codegen_matrix_finite(codegen_data->Lpu, prefix + " Lpu");
  check_codegen_matrix_approx(codegen_data->Gx, direct_data->Gx, tol,
                              prefix + " Gx");
  check_codegen_matrix_approx(codegen_data->Gu, direct_data->Gu, tol,
                              prefix + " Gu");
  check_codegen_matrix_approx(codegen_data->Gp, direct_data->Gp, tol,
                              prefix + " Gp");
  check_codegen_matrix_approx(codegen_data->Hx, direct_data->Hx, tol,
                              prefix + " Hx");
  check_codegen_matrix_approx(codegen_data->Hu, direct_data->Hu, tol,
                              prefix + " Hu");
  check_codegen_matrix_approx(codegen_data->Hp, direct_data->Hp, tol,
                              prefix + " Hp");
  check_codegen_matrix_approx(codegen_observer_data->dissipative_E,
                              direct_observer_data->dissipative_E, tol,
                              prefix + " dissipative_E");
  check_codegen_matrix_approx(codegen_observer_data->Ex,
                              direct_observer_data->Ex, tol, prefix + " Ex");
  check_codegen_matrix_approx(codegen_observer_data->Eu,
                              direct_observer_data->Eu, tol, prefix + " Eu");
  check_codegen_matrix_approx(codegen_observer_data->Ep,
                              direct_observer_data->Ep, tol, prefix + " Ep");
  check_codegen_matrix_approx(codegen_observer_data->dissipative_E,
                              numdiff_observer_data->dissipative_E, tol,
                              prefix + " numdiff dissipative_E");
  check_codegen_matrix_approx(codegen_observer_data->Ex,
                              numdiff_observer_data->Ex, tol,
                              prefix + " numdiff Ex");
  check_codegen_matrix_approx(codegen_observer_data->Eu,
                              numdiff_observer_data->Eu, tol,
                              prefix + " numdiff Eu");
  check_codegen_matrix_approx(codegen_observer_data->Ep,
                              numdiff_observer_data->Ep, tol,
                              prefix + " numdiff Ep");
}

template <typename Scalar>
void test_codegen_observer_parameterized_power_impl(const std::string& name) {
  typedef crocoddyl::ObserverModelAbstractTpl<Scalar> ObserverModelAbstract;
  typedef crocoddyl::ObserverModelCodeGenTpl<Scalar> ObserverModelCodeGen;
  typedef crocoddyl::ObserverDataCodeGenTpl<Scalar> ObserverDataCodeGen;
  typedef crocoddyl::ObserverModelNumDiffTpl<Scalar> ObserverModelNumDiff;
  typedef crocoddyl::IntegratedObserverModelAbstractTpl<Scalar>
      IntegratedObserverModelAbstract;
  typedef crocoddyl::ObservationProblemTpl<Scalar> ObservationProblem;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef crocoddyl::ParameterPhaseModelTpl<Scalar> ParameterPhaseModel;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const Scalar tol =
      std::is_same<Scalar, float>::value ? Scalar(6e-3) : Scalar(5e-7);
  pinocchio::ModelTpl<double> modeld;
  pinocchio::buildModels::manipulator(modeld);
  std::shared_ptr<ParameterManager> direct_params;
  const std::shared_ptr<ObserverModelAbstract> direct_model =
      build_codegen_observer_model<Scalar>(modeld, direct_params);
  const std::shared_ptr<IntegratedObserverModelAbstract> integrated_model =
      std::dynamic_pointer_cast<IntegratedObserverModelAbstract>(direct_model);
  BOOST_REQUIRE(integrated_model != nullptr);
  integrated_model->get_costs()->removeCost("powerObs");

  const std::shared_ptr<ObserverModelCodeGen> codegen_model =
      std::make_shared<ObserverModelCodeGen>(
          direct_model, "pddp_" + name + "_observer_normal_autodiff_codegen",
          true);
  ObserverModelNumDiff numdiff_model(direct_model, direct_params);
  BOOST_CHECK_EQUAL(codegen_model->get_np(), direct_params->get_np());
  BOOST_CHECK_EQUAL(codegen_model->get_ng(), direct_model->get_ng());
  BOOST_CHECK_EQUAL(codegen_model->get_nh(), direct_model->get_nh());
  BOOST_CHECK_EQUAL(codegen_model->get_ng_T(), direct_model->get_ng_T());
  BOOST_CHECK_EQUAL(codegen_model->get_nh_T(), direct_model->get_nh_T());

  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> direct_data =
      direct_model->createData(direct_params->createData());
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> codegen_data =
      codegen_model->createData();
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> numdiff_data =
      numdiff_model.createData(direct_params->createData());
  direct_model->set_params(direct_data, direct_params);
  codegen_model->set_params(codegen_data, direct_params);

  VectorXs p = direct_params->zero();
  p += VectorXs::LinSpaced(p.size(), Scalar(-0.03), Scalar(0.04));
  direct_model->update_p(direct_data, p);
  codegen_model->update_p(codegen_data, p);
  numdiff_model.update_p(numdiff_data, p);

  const VectorXs tau =
      VectorXs::LinSpaced(codegen_model->get_ntau(), Scalar(-0.2), Scalar(0.3));
  direct_model->update_tau(tau);
  codegen_model->update_tau(tau);
  numdiff_model.update_tau(tau);

  const VectorXs dx = VectorXs::LinSpaced(direct_model->get_state()->get_ndx(),
                                          Scalar(-0.08), Scalar(0.11));
  VectorXs x(direct_model->get_state()->get_nx());
  direct_model->get_state()->integrate(direct_model->get_state()->zero(), dx,
                                       x);
  const VectorXs w =
      VectorXs::LinSpaced(direct_model->get_nu(), Scalar(-0.01), Scalar(0.015));
  direct_model->calc(direct_data, x, w);
  direct_model->calcDiff(direct_data, x, w);
  numdiff_model.calc(numdiff_data, x, w);
  numdiff_model.calcDiff(numdiff_data, x, w);
  codegen_model->calc(codegen_data, x, w);
  codegen_model->calcDiff(codegen_data, x, w);
  compare_codegen_observer_data(codegen_data, direct_data, numdiff_data, tol,
                                name);
  BOOST_CHECK_GT(codegen_data->Fp.norm(), Scalar(0));
  const MatrixXs running_H_codegen =
      assemble_running_cost_hessian(codegen_model, codegen_data);
  const MatrixXs running_H_direct =
      assemble_running_cost_hessian(direct_model, direct_data);
  check_codegen_matrix_abs_approx(running_H_codegen, running_H_direct, tol,
                                  name + " running GN Hessian");

  const VectorXs x_terminal = direct_model->get_state()->zero();
  direct_model->calc(direct_data, x_terminal);
  direct_model->calcDiff(direct_data, x_terminal);
  numdiff_model.calc(numdiff_data, x_terminal);
  numdiff_model.calcDiff(numdiff_data, x_terminal);
  codegen_model->calc(codegen_data, x_terminal);
  codegen_model->calcDiff(codegen_data, x_terminal);
  compare_codegen_observer_data(codegen_data, direct_data, numdiff_data, tol,
                                name + " terminal");
  const MatrixXs terminal_H_codegen =
      assemble_terminal_cost_hessian(codegen_model, codegen_data);
  const MatrixXs terminal_H_direct =
      assemble_terminal_cost_hessian(direct_model, direct_data);
  check_codegen_matrix_abs_approx(terminal_H_codegen, terminal_H_direct, tol,
                                  name + " terminal GN Hessian");

  const VectorXs tau_updated = VectorXs::LinSpaced(codegen_model->get_ntau(),
                                                   Scalar(0.35), Scalar(-0.25));
  codegen_model->update_tau(tau_updated);
  codegen_model->calc(codegen_data, x, w);
  const std::shared_ptr<ObserverDataCodeGen> typed_codegen_data =
      std::dynamic_pointer_cast<ObserverDataCodeGen>(codegen_data);
  BOOST_REQUIRE(typed_codegen_data != nullptr);
  const VectorXs running_tau_input =
      typed_codegen_data->X.tail(codegen_model->get_ntau());
  check_codegen_matrix_approx(running_tau_input, tau_updated, tol,
                              name + " running tau input");
  codegen_model->calc(codegen_data, x);
  const VectorXs terminal_tau_input =
      typed_codegen_data->X_T.tail(codegen_model->get_ntau());
  check_codegen_matrix_approx(terminal_tau_input, tau_updated, tol,
                              name + " terminal tau input");

  const std::vector<std::shared_ptr<ObserverModelAbstract>> running_models{
      codegen_model};
  const std::vector<VectorXs> tau_meas{tau};
  const std::shared_ptr<ParameterPhaseModel> params_model =
      std::make_shared<ParameterPhaseModel>(direct_params);
  ObservationProblem problem(x, tau_meas, running_models, codegen_model,
                             params_model);
  problem.update_p(p);
  VectorXs xnext(direct_model->get_state()->get_nx());
  direct_model->get_state()->integrate(x, Scalar(0.25) * dx, xnext);
  const std::vector<VectorXs> xs{x, xnext};
  const std::vector<VectorXs> ws{w};
  BOOST_CHECK_NO_THROW(problem.calc(xs, ws));
  BOOST_CHECK_NO_THROW(problem.calcDiff(xs, ws));
  const std::shared_ptr<ObserverDataCodeGen> problem_running_data =
      std::dynamic_pointer_cast<ObserverDataCodeGen>(
          problem.get_runningDatas()[0]);
  BOOST_REQUIRE(problem_running_data != nullptr);
  const VectorXs problem_parameter_input = problem_running_data->X.segment(
      codegen_model->get_state()->get_nx() + codegen_model->get_nu(),
      codegen_model->get_np());
  check_codegen_matrix_approx(problem_parameter_input, p, tol,
                              name + " observation problem parameter input");
}

void test_codegen_4DoFArm() {
  typedef double Scalar;
  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::Vector3s Vector3s;
  const Scalar tol_x = Scalar(1e-9);
  const Scalar tol_diff = Scalar(1e-7);
  typedef typename crocoddyl::ResidualModelFrameTranslationTpl<Scalar>
      ResidualModelFrameTranslation;

  std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>> runningModelD =
      build_arm_action_model<Scalar>();

  // The definition of the ActionModelCodeGen takes the size of the environment
  // variable, and the function setting the environment variable as arguments.
  std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>> runningModelCG =
      std::make_shared<crocoddyl::ActionModelCodeGenTpl<Scalar>>(
          runningModelD, "pyrene_arm_running", false, 3, change_env<ADScalar>,
          crocoddyl::defaultCompilerType(), "-O0");

  // Check that code-generated action model is the same as original.
  /**************************************************************************/
  std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> runningDataCG =
      runningModelCG->createData();
  std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> runningDataD =
      runningModelD->createData();

  // Change cost reference
  // ********************************************************/
  const Vector3s new_ref(Vector3s::Random());
  crocoddyl::ActionModelCodeGenTpl<Scalar>* rmcg =
      static_cast<crocoddyl::ActionModelCodeGenTpl<Scalar>*>(
          runningModelCG.get());
  rmcg->update_p(runningDataCG, new_ref);
  crocoddyl::IntegratedActionModelEulerTpl<Scalar>* m =
      static_cast<crocoddyl::IntegratedActionModelEulerTpl<Scalar>*>(
          runningModelD.get());
  crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar>* md =
      static_cast<crocoddyl::DynamicsModelConstrainedForwardTpl<Scalar>*>(
          m->get_dynamics().get());

  std::shared_ptr<ResidualModelFrameTranslation> residual =
      std::static_pointer_cast<ResidualModelFrameTranslation>(
          m->get_costs()
              ->get_costs()
              .find("gripperTrans")
              ->second->cost->get_residual());
  residual->set_id(md->get_pinocchio().getFrameId("gripper_left_joint"));
  residual->set_reference(new_ref);
  /*************************************************************/

  VectorXs x_rand = runningModelCG->get_state()->zero();
  x_rand.tail(runningModelCG->get_state()->get_nv()).setConstant(Scalar(0.1));
  VectorXs u_rand = VectorXs::Random(runningModelCG->get_nu());
  runningModelD->calc(runningDataD, x_rand, u_rand);
  runningModelD->calcDiff(runningDataD, x_rand, u_rand);
  runningModelCG->calc(runningDataCG, x_rand, u_rand);
  runningModelCG->calcDiff(runningDataCG, x_rand, u_rand);

  check_codegen_matrix_approx(runningDataCG->xnext, runningDataD->xnext, tol_x,
                              "arm xnext");
  BOOST_CHECK_CLOSE(runningDataCG->cost, runningDataD->cost, Scalar(1e-10));
  check_codegen_matrix_approx(runningDataCG->Lx, runningDataD->Lx, tol_diff,
                              "arm Lx");
  check_codegen_matrix_approx(runningDataCG->Lu, runningDataD->Lu, tol_diff,
                              "arm Lu");
  check_codegen_matrix_approx(runningDataCG->Lxx, runningDataD->Lxx, tol_diff,
                              "arm Lxx");
  check_codegen_matrix_approx(runningDataCG->Lxu, runningDataD->Lxu, tol_diff,
                              "arm Lxu");
  check_codegen_matrix_approx(runningDataCG->Luu, runningDataD->Luu, tol_diff,
                              "arm Luu");
  check_codegen_matrix_approx(runningDataCG->Fx, runningDataD->Fx, tol_diff,
                              "arm Fx");
  check_codegen_matrix_approx(runningDataCG->Fu, runningDataD->Fu, tol_diff,
                              "arm Fu");
}

void test_codegen_bipedal() {
  typedef double Scalar;
  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  const Scalar tol_x = Scalar(1e-9);
  const Scalar tol_diff = Scalar(1e-6);
  std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>> runningModelD =
      build_bipedal_action_model<Scalar>();
  std::shared_ptr<crocoddyl::ActionModelAbstractTpl<ADScalar>> runningModelAD =
      build_bipedal_action_model<ADScalar>();

  const typename crocoddyl::ActionModelCodeGenTpl<Scalar>::ParamsEnvironment
      empty_update =
          [](std::shared_ptr<crocoddyl::ActionModelAbstractTpl<ADScalar>>,
             const Eigen::Ref<
                 const typename crocoddyl::MathBaseTpl<ADScalar>::VectorXs>&) {
          };
  std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>> runningModelCG =
      std::make_shared<crocoddyl::ActionModelCodeGenTpl<Scalar>>(
          runningModelAD, "pyrene_biped", false, 0, empty_update,
          crocoddyl::defaultCompilerType(), "-O0");

  // Check that code-generated action model is the same as original.
  /**************************************************************************/
  std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> runningDataCG =
      runningModelCG->createData();
  std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> runningDataD =
      runningModelD->createData();
  VectorXs x_rand = runningModelCG->get_state()->rand();
  VectorXs u_rand = VectorXs::Random(runningModelCG->get_nu());
  runningModelD->calc(runningDataD, x_rand, u_rand);
  runningModelD->calcDiff(runningDataD, x_rand, u_rand);
  runningModelCG->calc(runningDataCG, x_rand, u_rand);
  runningModelCG->calcDiff(runningDataCG, x_rand, u_rand);

  check_codegen_matrix_approx(runningDataCG->xnext, runningDataD->xnext, tol_x,
                              "biped xnext");
  BOOST_CHECK_CLOSE(runningDataCG->cost, runningDataD->cost, Scalar(1e-10));
  check_codegen_matrix_approx(runningDataCG->Lx, runningDataD->Lx, tol_diff,
                              "biped Lx");
  check_codegen_matrix_approx(runningDataCG->Lu, runningDataD->Lu, tol_diff,
                              "biped Lu");
  check_codegen_matrix_approx(runningDataCG->Lxx, runningDataD->Lxx, tol_diff,
                              "biped Lxx");
  check_codegen_matrix_approx(runningDataCG->Lxu, runningDataD->Lxu, tol_diff,
                              "biped Lxu");
  check_codegen_matrix_approx(runningDataCG->Luu, runningDataD->Luu, tol_diff,
                              "biped Luu");
  check_codegen_matrix_approx(runningDataCG->Fx, runningDataD->Fx, tol_diff,
                              "biped Fx");
  check_codegen_matrix_approx(runningDataCG->Fu, runningDataD->Fu, tol_diff,
                              "biped Fu");
}

void test_codegen_freeflyer_autodiff_manifold() {
  typedef double Scalar;
  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  const Scalar tol_x = Scalar(1e-9);
  const Scalar tol_diff = Scalar(1e-6);
  std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>> runningModelD =
      build_freeflyer_manifold_action_model<Scalar>();
  std::shared_ptr<crocoddyl::ActionModelAbstractTpl<ADScalar>> runningModelAD =
      build_freeflyer_manifold_action_model<ADScalar>();

  BOOST_REQUIRE_NE(runningModelD->get_state()->get_nx(),
                   runningModelD->get_state()->get_ndx());

  const typename crocoddyl::ActionModelCodeGenTpl<Scalar>::ParamsEnvironment
      empty_update =
          [](std::shared_ptr<crocoddyl::ActionModelAbstractTpl<ADScalar>>,
             const Eigen::Ref<
                 const typename crocoddyl::MathBaseTpl<ADScalar>::VectorXs>&) {
          };
  std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar>> runningModelCG =
      std::make_shared<crocoddyl::ActionModelCodeGenTpl<Scalar>>(
          runningModelAD, "pddp_freeflyer_autodiff", true, 0, empty_update,
          crocoddyl::defaultCompilerType(), "-O0");

  std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> runningDataCG =
      runningModelCG->createData();
  std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> runningDataD =
      runningModelD->createData();
  const VectorXs x_rand = runningModelCG->get_state()->rand();
  const VectorXs u_rand = VectorXs::Random(runningModelCG->get_nu());
  runningModelD->calc(runningDataD, x_rand, u_rand);
  runningModelD->calcDiff(runningDataD, x_rand, u_rand);
  runningModelCG->calc(runningDataCG, x_rand, u_rand);
  runningModelCG->calcDiff(runningDataCG, x_rand, u_rand);

  check_codegen_matrix_approx(runningDataCG->xnext, runningDataD->xnext, tol_x,
                              "freeflyer-autodiff xnext");
  BOOST_CHECK_CLOSE(runningDataCG->cost, runningDataD->cost, Scalar(1e-10));
  check_codegen_matrix_approx(runningDataCG->Lx, runningDataD->Lx, tol_diff,
                              "freeflyer-autodiff Lx");
  check_codegen_matrix_approx(runningDataCG->Lu, runningDataD->Lu, tol_diff,
                              "freeflyer-autodiff Lu");
  check_codegen_matrix_approx(runningDataCG->Fx, runningDataD->Fx, tol_diff,
                              "freeflyer-autodiff Fx");
  check_codegen_matrix_approx(runningDataCG->Fu, runningDataD->Fu, tol_diff,
                              "freeflyer-autodiff Fu");
  check_codegen_matrix_finite(runningDataCG->Lxx, "freeflyer-autodiff Lxx");
  check_codegen_matrix_finite(runningDataCG->Lxu, "freeflyer-autodiff Lxu");
  check_codegen_matrix_finite(runningDataCG->Luu, "freeflyer-autodiff Luu");
  const MatrixXs running_H_codegen =
      assemble_running_cost_hessian(runningModelCG, runningDataCG);
  const MatrixXs running_H_direct =
      assemble_running_cost_hessian(runningModelD, runningDataD);
  check_codegen_matrix_abs_approx(running_H_codegen, running_H_direct, tol_diff,
                                  "freeflyer-autodiff running GN Hessian");

  runningModelD->calc(runningDataD, x_rand);
  runningModelD->calcDiff(runningDataD, x_rand);
  runningModelCG->calc(runningDataCG, x_rand);
  runningModelCG->calcDiff(runningDataCG, x_rand);
  const MatrixXs terminal_Fx_identity =
      MatrixXs::Identity(runningModelD->get_state()->get_ndx(),
                         runningModelD->get_state()->get_ndx());
  check_codegen_matrix_approx(runningDataCG->Lx, runningDataD->Lx, tol_diff,
                              "freeflyer-autodiff terminal Lx");
  check_codegen_matrix_approx(runningDataCG->Fx, terminal_Fx_identity, tol_diff,
                              "freeflyer-autodiff terminal Fx");
  check_codegen_matrix_finite(runningDataCG->Lxx,
                              "freeflyer-autodiff terminal Lxx");
  const MatrixXs terminal_H_codegen =
      assemble_terminal_cost_hessian(runningModelCG, runningDataCG);
  const MatrixXs terminal_H_direct =
      assemble_terminal_cost_hessian(runningModelD, runningDataD);
  check_codegen_matrix_abs_approx(terminal_H_codegen, terminal_H_direct,
                                  tol_diff,
                                  "freeflyer-autodiff terminal GN Hessian");
}

template <typename Scalar>
void test_codegen_parameterized_lqr_impl(const std::string& suffix) {
  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef crocoddyl::ActionModelLQRTpl<Scalar> ActionModelLQR;
  typedef crocoddyl::ActionModelLQRTpl<ADScalar> ADActionModelLQR;
  typedef crocoddyl::ActionModelCodeGenTpl<Scalar> ActionModelCodeGen;
  typedef crocoddyl::LQRParamsTpl<Scalar> LQRParams;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const std::size_t nx = 4;
  const std::size_t nu = 3;
  const std::size_t np = 2;
  const std::size_t ng = 2;
  const std::size_t nh = 1;
  const Scalar tol =
      std::is_same<Scalar, float>::value ? Scalar(2e-4) : Scalar(1e-10);

  const std::shared_ptr<ActionModelLQR> direct_model =
      std::make_shared<ActionModelLQR>(nx, nu, np, ng, nh, false);
  const std::shared_ptr<ADActionModelLQR> ad_model =
      std::make_shared<ADActionModelLQR>(nx, nu, np, ng, nh, false);
  const typename ActionModelCodeGen::ParamsEnvironment empty_update =
      [](std::shared_ptr<crocoddyl::ActionModelAbstractTpl<ADScalar>>,
         const Eigen::Ref<
             const typename crocoddyl::MathBaseTpl<ADScalar>::VectorXs>&) {};
  const std::shared_ptr<ActionModelCodeGen> codegen_model =
      std::make_shared<ActionModelCodeGen>(
          ad_model, "pddp_param_lqr_codegen_" + suffix, false, 0, empty_update,
          crocoddyl::defaultCompilerType(), "-O0");

  BOOST_CHECK_EQUAL(codegen_model->get_np(), np);

  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> direct_data =
      direct_model->createData();
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> codegen_data =
      codegen_model->createData();
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>
      codegen_terminal_data = codegen_model->createData();
  const std::shared_ptr<ParameterManager> params =
      std::make_shared<ParameterManager>(direct_model->get_state());
  params->addParam("lqr",
                   std::make_shared<LQRParams>(direct_model->get_state(), np));
  params->addParam("inactive",
                   std::make_shared<LQRParams>(direct_model->get_state(), np),
                   false);
  direct_model->set_params(direct_data, params);
  codegen_model->set_params(codegen_data, params);
  codegen_model->set_params(codegen_terminal_data, params);

  const VectorXs p = params->rand();
  direct_model->update_p(direct_data, p);
  codegen_model->update_p(codegen_data, p);
  codegen_model->update_p(codegen_terminal_data, p);

  const VectorXs x = direct_model->get_state()->rand();
  const VectorXs u = VectorXs::Random(nu);
  direct_model->calc(direct_data, x, u);
  direct_model->calcDiff(direct_data, x, u);
  codegen_model->calc(codegen_data, x, u);
  codegen_model->calcDiff(codegen_data, x, u);

  check_codegen_matrix_approx(codegen_data->xnext, direct_data->xnext, tol,
                              "param-lqr xnext");
  const Scalar cost_tolerance =
      std::is_same<Scalar, float>::value ? Scalar(1e-3) : Scalar(1e-12);
  BOOST_CHECK_CLOSE(codegen_data->cost, direct_data->cost, cost_tolerance);
  check_codegen_matrix_approx(codegen_data->g, direct_data->g, tol,
                              "param-lqr g");
  check_codegen_matrix_approx(codegen_data->h, direct_data->h, tol,
                              "param-lqr h");
  check_codegen_matrix_approx(codegen_data->Fx, direct_data->Fx, tol,
                              "param-lqr Fx");
  check_codegen_matrix_approx(codegen_data->Fu, direct_data->Fu, tol,
                              "param-lqr Fu");
  check_codegen_matrix_approx(codegen_data->Fp, direct_data->Fp, tol,
                              "param-lqr Fp");
  check_codegen_matrix_approx(codegen_data->Lx, direct_data->Lx, tol,
                              "param-lqr Lx");
  check_codegen_matrix_approx(codegen_data->Lu, direct_data->Lu, tol,
                              "param-lqr Lu");
  check_codegen_matrix_approx(codegen_data->Lp, direct_data->Lp, tol,
                              "param-lqr Lp");
  check_codegen_matrix_approx(codegen_data->Lxx, direct_data->Lxx, tol,
                              "param-lqr Lxx");
  check_codegen_matrix_approx(codegen_data->Lxu, direct_data->Lxu, tol,
                              "param-lqr Lxu");
  check_codegen_matrix_approx(codegen_data->Luu, direct_data->Luu, tol,
                              "param-lqr Luu");
  check_codegen_matrix_approx(codegen_data->Lpp, direct_data->Lpp, tol,
                              "param-lqr Lpp");
  check_codegen_matrix_approx(codegen_data->Lpx, direct_data->Lpx, tol,
                              "param-lqr Lpx");
  check_codegen_matrix_approx(codegen_data->Lpu, direct_data->Lpu, tol,
                              "param-lqr Lpu");
  check_codegen_matrix_approx(codegen_data->Gx, direct_data->Gx, tol,
                              "param-lqr Gx");
  check_codegen_matrix_approx(codegen_data->Gu, direct_data->Gu, tol,
                              "param-lqr Gu");
  check_codegen_matrix_approx(codegen_data->Gp, direct_data->Gp, tol,
                              "param-lqr Gp");
  check_codegen_matrix_approx(codegen_data->Hx, direct_data->Hx, tol,
                              "param-lqr Hx");
  check_codegen_matrix_approx(codegen_data->Hu, direct_data->Hu, tol,
                              "param-lqr Hu");
  check_codegen_matrix_approx(codegen_data->Hp, direct_data->Hp, tol,
                              "param-lqr Hp");

  codegen_terminal_data->Fp.setConstant(Scalar(21));
  codegen_terminal_data->Fu.setConstant(Scalar(22));
  codegen_terminal_data->Lu.setConstant(Scalar(23));
  codegen_terminal_data->Lxu.setConstant(Scalar(24));
  codegen_terminal_data->Luu.setConstant(Scalar(25));
  codegen_terminal_data->Lpu.setConstant(Scalar(26));
  codegen_terminal_data->Gu.setConstant(Scalar(27));
  codegen_terminal_data->Hu.setConstant(Scalar(28));

  direct_model->calc(direct_data, x);
  direct_model->calcDiff(direct_data, x);
  codegen_model->calc(codegen_terminal_data, x);
  codegen_model->calcDiff(codegen_terminal_data, x);
  BOOST_CHECK(codegen_terminal_data->Fp.isConstant(Scalar(21)));
  BOOST_CHECK(codegen_terminal_data->Fu.isConstant(Scalar(22)));
  BOOST_CHECK(codegen_terminal_data->Lu.isConstant(Scalar(23)));
  BOOST_CHECK(codegen_terminal_data->Lxu.isConstant(Scalar(24)));
  BOOST_CHECK(codegen_terminal_data->Luu.isConstant(Scalar(25)));
  BOOST_CHECK(codegen_terminal_data->Lpu.isConstant(Scalar(26)));
  BOOST_CHECK(codegen_terminal_data->Gu.isConstant(Scalar(27)));
  BOOST_CHECK(codegen_terminal_data->Hu.isConstant(Scalar(28)));
  check_codegen_matrix_approx(codegen_terminal_data->g, direct_data->g, tol,
                              "param-lqr terminal g");
  check_codegen_matrix_approx(codegen_terminal_data->h, direct_data->h, tol,
                              "param-lqr terminal h");
  check_codegen_matrix_approx(codegen_terminal_data->Lp, direct_data->Lp, tol,
                              "param-lqr terminal Lp");
  check_codegen_matrix_approx(codegen_terminal_data->Lpp, direct_data->Lpp, tol,
                              "param-lqr terminal Lpp");
  check_codegen_matrix_approx(codegen_terminal_data->Lpx, direct_data->Lpx, tol,
                              "param-lqr terminal Lpx");
  check_codegen_matrix_approx(codegen_terminal_data->Gp, direct_data->Gp, tol,
                              "param-lqr terminal Gp");
  check_codegen_matrix_approx(codegen_terminal_data->Hp, direct_data->Hp, tol,
                              "param-lqr terminal Hp");

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      codegen_model->update_p(codegen_data, p);
      codegen_model->update_p(codegen_terminal_data, p);
      codegen_model->calc(codegen_data, x, u);
      codegen_model->calcDiff(codegen_data, x, u);
      codegen_model->calc(codegen_terminal_data, x);
      codegen_model->calcDiff(codegen_terminal_data, x);
    }
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);

  const std::shared_ptr<ADActionModelLQR> ad_model_autodiff =
      std::make_shared<ADActionModelLQR>(nx, nu, np, ng, nh, false);
  const std::shared_ptr<ActionModelCodeGen> codegen_autodiff_model =
      std::make_shared<ActionModelCodeGen>(
          ad_model_autodiff, "pddp_param_lqr_codegen_autodiff_" + suffix, true,
          0, empty_update, crocoddyl::defaultCompilerType(), "-O0");
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>
      codegen_autodiff_data = codegen_autodiff_model->createData();
  codegen_autodiff_model->set_params(codegen_autodiff_data, params);
  codegen_autodiff_model->update_p(codegen_autodiff_data, p);

  direct_model->calc(direct_data, x, u);
  direct_model->calcDiff(direct_data, x, u);
  codegen_autodiff_model->calc(codegen_autodiff_data, x, u);
  codegen_autodiff_model->calcDiff(codegen_autodiff_data, x, u);
  check_codegen_matrix_approx(codegen_autodiff_data->Fp, direct_data->Fp, tol,
                              "param-lqr-autodiff Fp");
  check_codegen_matrix_approx(codegen_autodiff_data->Lp, direct_data->Lp, tol,
                              "param-lqr-autodiff Lp");
  check_codegen_matrix_approx(codegen_autodiff_data->Lpp, direct_data->Lpp, tol,
                              "param-lqr-autodiff Lpp");
  check_codegen_matrix_approx(codegen_autodiff_data->Lpx, direct_data->Lpx, tol,
                              "param-lqr-autodiff Lpx");
  check_codegen_matrix_approx(codegen_autodiff_data->Lpu, direct_data->Lpu, tol,
                              "param-lqr-autodiff Lpu");
  check_codegen_matrix_approx(codegen_autodiff_data->Gp, direct_data->Gp, tol,
                              "param-lqr-autodiff Gp");
  check_codegen_matrix_approx(codegen_autodiff_data->Hp, direct_data->Hp, tol,
                              "param-lqr-autodiff Hp");
  const typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs running_H_autodiff =
      assemble_running_cost_hessian(codegen_autodiff_model,
                                    codegen_autodiff_data);
  const typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs running_H_direct =
      assemble_running_cost_hessian(direct_model, direct_data);
  check_codegen_matrix_abs_approx(running_H_autodiff, running_H_direct, tol,
                                  "param-lqr-autodiff running GN Hessian");

  direct_model->calc(direct_data, x);
  direct_model->calcDiff(direct_data, x);
  codegen_autodiff_model->calc(codegen_autodiff_data, x);
  codegen_autodiff_model->calcDiff(codegen_autodiff_data, x);
  check_codegen_matrix_approx(codegen_autodiff_data->Lp, direct_data->Lp, tol,
                              "param-lqr-autodiff terminal Lp");
  check_codegen_matrix_approx(codegen_autodiff_data->Lpp, direct_data->Lpp, tol,
                              "param-lqr-autodiff terminal Lpp");
  check_codegen_matrix_approx(codegen_autodiff_data->Lpx, direct_data->Lpx, tol,
                              "param-lqr-autodiff terminal Lpx");
  check_codegen_matrix_approx(codegen_autodiff_data->Gp, direct_data->Gp, tol,
                              "param-lqr-autodiff terminal Gp");
  check_codegen_matrix_approx(codegen_autodiff_data->Hp, direct_data->Hp, tol,
                              "param-lqr-autodiff terminal Hp");
  const typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs terminal_H_autodiff =
      assemble_terminal_cost_hessian(codegen_autodiff_model,
                                     codegen_autodiff_data);
  const typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs terminal_H_direct =
      assemble_terminal_cost_hessian(direct_model, direct_data);
  check_codegen_matrix_abs_approx(terminal_H_autodiff, terminal_H_direct, tol,
                                  "param-lqr-autodiff terminal GN Hessian");
}

void test_codegen_parameterized_lqr() {
  test_codegen_parameterized_lqr_impl<double>("float64");
}

void test_codegen_parameterized_lqr_float32() {
  test_codegen_parameterized_lqr_impl<float>("float32");
}

template <typename Scalar>
void compare_codegen_parameterized_action_data(
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>& codegen,
    const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>& direct,
    const Scalar tol, const std::string& prefix, const bool terminal) {
  check_codegen_matrix_abs_approx(codegen->xnext, direct->xnext, tol,
                                  prefix + " xnext");
  BOOST_CHECK_SMALL(codegen->cost - direct->cost, tol);
  check_codegen_matrix_abs_approx(codegen->g, direct->g, tol, prefix + " g");
  check_codegen_matrix_abs_approx(codegen->h, direct->h, tol, prefix + " h");
  BOOST_CHECK_MESSAGE(codegen->Fx.allFinite(),
                      prefix << " generated Fx is not finite");
  BOOST_CHECK_MESSAGE(direct->Fx.allFinite(),
                      prefix << " direct Fx is not finite");
  check_codegen_matrix_abs_approx(codegen->Fx, direct->Fx, tol, prefix + " Fx");
  check_codegen_matrix_abs_approx(codegen->Fp, direct->Fp, tol, prefix + " Fp");
  check_codegen_matrix_abs_approx(codegen->Lx, direct->Lx, tol, prefix + " Lx");
  check_codegen_matrix_abs_approx(codegen->Lp, direct->Lp, tol, prefix + " Lp");
  check_codegen_matrix_abs_approx(codegen->Lxx, direct->Lxx, tol,
                                  prefix + " Lxx");
  check_codegen_matrix_abs_approx(codegen->Lpp, direct->Lpp, tol,
                                  prefix + " Lpp");
  check_codegen_matrix_abs_approx(codegen->Lpx, direct->Lpx, tol,
                                  prefix + " Lpx");
  check_codegen_matrix_abs_approx(codegen->Gx, direct->Gx, tol, prefix + " Gx");
  check_codegen_matrix_abs_approx(codegen->Gp, direct->Gp, tol, prefix + " Gp");
  check_codegen_matrix_abs_approx(codegen->Hx, direct->Hx, tol, prefix + " Hx");
  check_codegen_matrix_abs_approx(codegen->Hp, direct->Hp, tol, prefix + " Hp");
  if (!terminal) {
    check_codegen_matrix_abs_approx(codegen->Fu, direct->Fu, tol,
                                    prefix + " Fu");
    check_codegen_matrix_abs_approx(codegen->Lu, direct->Lu, tol,
                                    prefix + " Lu");
    check_codegen_matrix_abs_approx(codegen->Lxu, direct->Lxu, tol,
                                    prefix + " Lxu");
    check_codegen_matrix_abs_approx(codegen->Luu, direct->Luu, tol,
                                    prefix + " Luu");
    check_codegen_matrix_abs_approx(codegen->Lpu, direct->Lpu, tol,
                                    prefix + " Lpu");
    check_codegen_matrix_abs_approx(codegen->Gu, direct->Gu, tol,
                                    prefix + " Gu");
    check_codegen_matrix_abs_approx(codegen->Hu, direct->Hu, tol,
                                    prefix + " Hu");
  }
}

template <typename Scalar>
void test_codegen_parameterized_integrators_impl(const std::string& suffix) {
  typedef crocoddyl::ActionModelCodeGenTpl<Scalar> ActionModelCodeGen;
  typedef crocoddyl::ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const pinocchio::ModelTpl<double> modeld = build_codegen_freeflyer_model();
  std::vector<CodegenParameterizedActionCaseTpl<Scalar>> cases;
  cases.push_back(build_codegen_parameterized_integrator<Scalar>(
      modeld, "euler", crocoddyl::two));
  cases.push_back(build_codegen_parameterized_integrator<Scalar>(
      modeld, "rk2", crocoddyl::two));
  cases.push_back(build_codegen_parameterized_integrator<Scalar>(
      modeld, "rk3", crocoddyl::three));
  cases.push_back(build_codegen_parameterized_integrator<Scalar>(
      modeld, "rk4", crocoddyl::four));
  cases.push_back(build_codegen_parameterized_discretized<Scalar>(modeld));

  const Scalar tol =
      std::is_same<Scalar, float>::value ? Scalar(6e-3) : Scalar(1e-4);
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const CodegenParameterizedActionCaseTpl<Scalar>& item = cases[i];
    BOOST_REQUIRE(!item.params->getParamStatus("a_inactive"));
    BOOST_REQUIRE_EQUAL(item.model->get_np(), item.params->get_np());

    const std::string prefix = item.name + "-" + suffix;
    const std::shared_ptr<ActionModelCodeGen> codegen =
        std::make_shared<ActionModelCodeGen>(
            item.model, "pddp_parameterized_autodiff_" + prefix, true);
    BOOST_REQUIRE_EQUAL(codegen->get_np(), item.params->get_np());

    const std::shared_ptr<ActionDataAbstract> direct = item.model->createData();
    const std::shared_ptr<ActionDataAbstract> generated = codegen->createData();
    const std::shared_ptr<ActionDataAbstract> direct_terminal =
        item.model->createData();
    const std::shared_ptr<ActionDataAbstract> generated_terminal =
        codegen->createData();
    codegen->set_params(generated, item.params);
    codegen->set_params(generated_terminal, item.params);
    item.model->update_p(direct, item.p);
    item.model->update_p(direct_terminal, item.p);
    codegen->update_p(generated, item.p);
    codegen->update_p(generated_terminal, item.p);

    const VectorXs dx = VectorXs::LinSpaced(item.model->get_state()->get_ndx(),
                                            Scalar(-0.06), Scalar(0.08));
    VectorXs x(item.model->get_state()->get_nx());
    item.model->get_state()->integrate(item.model->get_state()->zero(), dx, x);
    const VectorXs u =
        VectorXs::LinSpaced(item.model->get_nu(), Scalar(-0.08), Scalar(0.09));
    item.model->calc(direct, x, u);
    item.model->calcDiff(direct, x, u);
    codegen->calc(generated, x, u);
    codegen->calcDiff(generated, x, u);
    compare_codegen_parameterized_action_data(generated, direct, tol, prefix,
                                              false);
    BOOST_CHECK_GT(generated->Lp.norm(), Scalar(0));
    BOOST_CHECK_GT(generated->Gp.norm(), Scalar(0));
    BOOST_CHECK_GT(generated->Hp.norm(), Scalar(0));
    BOOST_CHECK_GT(generated->Fp.norm(), Scalar(0));

    item.model->calc(direct_terminal, x);
    item.model->calcDiff(direct_terminal, x);
    codegen->calc(generated_terminal, x);
    codegen->calcDiff(generated_terminal, x);
    compare_codegen_parameterized_action_data(
        generated_terminal, direct_terminal, tol, prefix + " terminal", true);
    BOOST_CHECK_GT(generated_terminal->Lp.norm(), Scalar(0));
    BOOST_CHECK_GT(generated_terminal->Gp.rows(), 0);
    BOOST_CHECK_GT(generated_terminal->Hp.rows(), 0);
    BOOST_CHECK(!item.params->getParamStatus("a_inactive"));

    const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
    Eigen::internal::set_is_malloc_allowed(false);
    try {
      for (std::size_t k = 0; k < 100; ++k) {
        codegen->update_p(generated, item.p);
        codegen->calc(generated, x, u);
        codegen->calcDiff(generated, x, u);
        codegen->update_p(generated_terminal, item.p);
        codegen->calc(generated_terminal, x);
        codegen->calcDiff(generated_terminal, x);
      }
    } catch (...) {
      Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
      throw;
    }
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
  }
}

void test_codegen_parameterized_integrators() {
  test_codegen_parameterized_integrators_impl<double>("float64");
}

void test_codegen_parameterized_integrators_float32() {
  test_codegen_parameterized_integrators_impl<float>("float32");
}

void test_codegen_observer_euler_parameterized_power() {
  test_codegen_observer_parameterized_power_impl<double>("euler");
}

void test_codegen_observer_euler_parameterized_power_float32() {
  test_codegen_observer_parameterized_power_impl<float>("euler_float32");
}

template <typename Scalar>
void test_codegen_observer_state_environment_autodiff_impl(
    const std::string& suffix) {
  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef crocoddyl::ObserverModelAbstractTpl<Scalar> ObserverModelAbstract;
  typedef crocoddyl::ObserverModelAbstractTpl<ADScalar> ADObserverModelAbstract;
  typedef crocoddyl::ObserverModelCodeGenTpl<Scalar> ObserverModelCodeGen;
  typedef crocoddyl::ObserverModelNumDiffTpl<Scalar> ObserverModelNumDiff;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const Scalar tol =
      std::is_same<Scalar, float>::value ? Scalar(5e-3) : Scalar(1e-7);
  const Scalar fd_step =
      std::is_same<Scalar, float>::value ? Scalar(3e-3) : Scalar(1e-5);
  const pinocchio::ModelTpl<double> modeld = build_codegen_freeflyer_model();
  std::shared_ptr<ParameterManager> direct_params;
  std::shared_ptr<crocoddyl::ParameterManagerTpl<ADScalar>> ad_params;
  const std::shared_ptr<ObserverModelAbstract> direct_model =
      build_codegen_observer_model<Scalar>(modeld, direct_params);
  const std::shared_ptr<ADObserverModelAbstract> ad_model =
      build_codegen_observer_model<ADScalar>(modeld, ad_params);
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<ADScalar>> ad_data =
      ad_model->createData(ad_params->createData());
  ad_model->set_params(ad_data, ad_params);

  BOOST_REQUIRE_NE(direct_model->get_state()->get_nx(),
                   direct_model->get_state()->get_ndx());

  const std::size_t nx = direct_model->get_state()->get_nx();
  const std::size_t nenv = nx + 1u;
  const std::shared_ptr<ObserverModelCodeGen> codegen_model =
      std::make_shared<ObserverModelCodeGen>(
          ad_model, "pddp_observer_state_env_autodiff_" + suffix, true, 0, nenv,
          "xReg", "pReg", crocoddyl::defaultCompilerType(), "-O0");
  ObserverModelNumDiff numdiff_model(direct_model, direct_params);

  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> direct_data =
      direct_model->createData(direct_params->createData());
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> codegen_data =
      codegen_model->createData();
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>>
      codegen_terminal_data = codegen_model->createData();
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> numdiff_data =
      numdiff_model.createData(direct_params->createData());
  direct_model->set_params(direct_data, direct_params);
  codegen_model->set_params(codegen_data, direct_params);
  codegen_model->set_params(codegen_terminal_data, direct_params);

  VectorXs p = direct_params->zero();
  p += VectorXs::LinSpaced(p.size(), Scalar(-0.03), Scalar(0.04));
  direct_model->update_p(direct_data, p);
  codegen_model->update_p(codegen_data, p);
  codegen_model->update_p(codegen_terminal_data, p);
  numdiff_model.update_p(numdiff_data, p);

  VectorXs env(nenv);
  const VectorXs dx_observed = VectorXs::LinSpaced(
      direct_model->get_state()->get_ndx(), Scalar(0.06), Scalar(-0.05));
  direct_model->get_state()->integrate(direct_model->get_state()->zero(),
                                       dx_observed, env.head(nx));
  env[nx] = Scalar(0.25);
  set_codegen_observer_state_environment(direct_model, env, "xReg", "pReg");
  codegen_model->update_env(codegen_data, env);
  codegen_model->update_env(codegen_terminal_data, env);

  const VectorXs tau =
      VectorXs::LinSpaced(codegen_model->get_ntau(), Scalar(-0.2), Scalar(0.3));
  direct_model->update_tau(tau);
  codegen_model->update_tau(tau);
  numdiff_model.update_tau(tau);

  const VectorXs dx = VectorXs::LinSpaced(direct_model->get_state()->get_ndx(),
                                          Scalar(-0.08), Scalar(0.11));
  VectorXs x(direct_model->get_state()->get_nx());
  direct_model->get_state()->integrate(direct_model->get_state()->zero(), dx,
                                       x);
  const VectorXs w =
      VectorXs::LinSpaced(direct_model->get_nu(), Scalar(-0.01), Scalar(0.015));
  direct_model->calc(direct_data, x, w);
  direct_model->calcDiff(direct_data, x, w);
  numdiff_model.calc(numdiff_data, x, w);
  numdiff_model.calcDiff(numdiff_data, x, w);
  codegen_model->calc(codegen_data, x, w);
  codegen_model->calcDiff(codegen_data, x, w);
  compare_codegen_observer_data(codegen_data, direct_data, numdiff_data, tol,
                                "observer-state-env", false);
  const VectorXs running_grad_fd = finite_difference_running_cost_gradient(
      direct_model, direct_data, x, w, p, fd_step);
  const std::size_t ndx = direct_model->get_state()->get_ndx();
  const std::size_t nu = direct_model->get_nu();
  check_codegen_matrix_abs_approx(codegen_data->Lx,
                                  VectorXs(running_grad_fd.head(ndx)), tol,
                                  "observer-state-env Lx");
  check_codegen_matrix_abs_approx(codegen_data->Lu,
                                  VectorXs(running_grad_fd.segment(ndx, nu)),
                                  tol, "observer-state-env Lu");
  check_codegen_matrix_abs_approx(
      codegen_data->Lp, VectorXs(running_grad_fd.tail(direct_model->get_np())),
      tol, "observer-state-env Lp");

  codegen_terminal_data->Fu.setConstant(Scalar(31));
  codegen_terminal_data->Lu.setConstant(Scalar(32));
  codegen_terminal_data->Lxu.setConstant(Scalar(33));
  codegen_terminal_data->Luu.setConstant(Scalar(34));
  codegen_terminal_data->Lpu.setConstant(Scalar(35));
  codegen_terminal_data->Gu.setConstant(Scalar(36));
  codegen_terminal_data->Hu.setConstant(Scalar(37));
  direct_data->Fu.setConstant(Scalar(31));
  direct_data->Lu.setConstant(Scalar(32));
  direct_data->Lxu.setConstant(Scalar(33));
  direct_data->Luu.setConstant(Scalar(34));
  direct_data->Lpu.setConstant(Scalar(35));
  direct_data->Gu.setConstant(Scalar(36));
  direct_data->Hu.setConstant(Scalar(37));
  direct_model->calc(direct_data, x);
  direct_model->calcDiff(direct_data, x);
  numdiff_model.calc(numdiff_data, x);
  numdiff_model.calcDiff(numdiff_data, x);
  codegen_model->calc(codegen_terminal_data, x);
  codegen_model->calcDiff(codegen_terminal_data, x);
  BOOST_CHECK(codegen_terminal_data->Fu.isConstant(Scalar(31)));
  BOOST_CHECK(codegen_terminal_data->Lu.isConstant(Scalar(32)));
  BOOST_CHECK(codegen_terminal_data->Lxu.isConstant(Scalar(33)));
  BOOST_CHECK(codegen_terminal_data->Luu.isConstant(Scalar(34)));
  BOOST_CHECK(codegen_terminal_data->Lpu.isConstant(Scalar(35)));
  BOOST_CHECK(codegen_terminal_data->Gu.isConstant(Scalar(36)));
  BOOST_CHECK(codegen_terminal_data->Hu.isConstant(Scalar(37)));
  compare_codegen_observer_data(codegen_terminal_data, direct_data,
                                numdiff_data, tol,
                                "observer-state-env terminal", false);
  const VectorXs terminal_grad_fd = finite_difference_terminal_cost_gradient(
      direct_model, direct_data, x, p, fd_step);
  check_codegen_matrix_abs_approx(codegen_terminal_data->Lx,
                                  VectorXs(terminal_grad_fd.head(ndx)), tol,
                                  "observer-state-env terminal Lx");
  check_codegen_matrix_abs_approx(
      codegen_terminal_data->Lp,
      VectorXs(terminal_grad_fd.tail(direct_model->get_np())), tol,
      "observer-state-env terminal Lp");

  const bool malloc_was_allowed = Eigen::internal::is_malloc_allowed();
  Eigen::internal::set_is_malloc_allowed(false);
  try {
    for (std::size_t i = 0; i < 100; ++i) {
      codegen_model->update_p(codegen_data, p);
      codegen_model->update_p(codegen_terminal_data, p);
      codegen_model->update_env(codegen_data, env);
      codegen_model->update_env(codegen_terminal_data, env);
      codegen_model->update_tau(tau);
      codegen_model->calc(codegen_data, x, w);
      codegen_model->calcDiff(codegen_data, x, w);
      codegen_model->calc(codegen_terminal_data, x);
      codegen_model->calcDiff(codegen_terminal_data, x);
    }
  } catch (...) {
    Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
    throw;
  }
  Eigen::internal::set_is_malloc_allowed(malloc_was_allowed);
}

void test_codegen_observer_state_environment_autodiff() {
  test_codegen_observer_state_environment_autodiff_impl<double>("float64");
}

void test_codegen_observer_state_environment_autodiff_float32() {
  test_codegen_observer_state_environment_autodiff_impl<float>("float32");
}

void test_codegen_observer_state_environment_analytic_lpp() {
  typedef double Scalar;
  typedef crocoddyl::ObserverModelAbstractTpl<Scalar> ObserverModelAbstract;
  typedef crocoddyl::ObserverModelCodeGenTpl<Scalar> ObserverModelCodeGen;
  typedef crocoddyl::IntegratedObserverModelAbstractTpl<Scalar>
      IntegratedObserverModelAbstract;
  typedef crocoddyl::ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::MatrixXs MatrixXs;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const Scalar tol = Scalar(1e-9);
  const pinocchio::ModelTpl<double> modeld = build_codegen_freeflyer_model();
  std::shared_ptr<ParameterManager> direct_params;
  const std::shared_ptr<ObserverModelAbstract> direct_model =
      build_codegen_observer_model<Scalar>(modeld, direct_params);
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> setup_data =
      direct_model->createData(direct_params->createData());
  direct_model->set_params(setup_data, direct_params);

  const std::size_t nx = direct_model->get_state()->get_nx();
  const std::size_t nenv = nx + 1u;
  const std::shared_ptr<ObserverModelCodeGen> codegen_model =
      std::make_shared<ObserverModelCodeGen>(
          direct_model, "pddp_observer_state_env_analytic_lpp", false, 0, nenv,
          "xReg", "pReg", crocoddyl::defaultCompilerType(), "-O0");

  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> direct_data =
      direct_model->createData(direct_params->createData());
  const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar>> codegen_data =
      codegen_model->createData();
  direct_model->set_params(direct_data, direct_params);
  codegen_model->set_params(codegen_data, direct_params);

  const VectorXs p = direct_params->rand();
  const VectorXs tau = VectorXs::Random(codegen_model->get_ntau());
  direct_model->update_tau(tau);
  codegen_model->update_tau(tau);
  direct_model->update_p(direct_data, p);
  codegen_model->update_p(codegen_data, p);

  const std::size_t np = direct_model->get_np();
  const std::size_t ndx = direct_model->get_state()->get_ndx();
  const std::size_t nu = direct_model->get_nu();
  const Scalar p_reg_weight = Scalar(0.25);
  const std::shared_ptr<IntegratedObserverModelAbstract> integrated =
      std::dynamic_pointer_cast<IntegratedObserverModelAbstract>(direct_model);
  BOOST_REQUIRE(integrated != nullptr);
  const Scalar expected_scale = integrated->get_dt() * p_reg_weight;
  const VectorXs expected_Lp_delta =
      expected_scale * (p - direct_params->zero());
  const MatrixXs expected_Lpp_delta =
      expected_scale * MatrixXs::Identity(np, np);
  const MatrixXs expected_Lpx_delta = MatrixXs::Zero(np, ndx);
  const MatrixXs expected_Lpu_delta = MatrixXs::Zero(np, nu);

  VectorXs direct_Lp_zero(np), direct_Lp_weight(np), codegen_Lp_zero(np),
      codegen_Lp_weight(np);
  MatrixXs direct_Lpp_zero(np, np), direct_Lpp_weight(np, np),
      codegen_Lpp_zero(np, np), codegen_Lpp_weight(np, np);
  MatrixXs direct_Lpx_zero(np, ndx), direct_Lpx_weight(np, ndx),
      codegen_Lpx_zero(np, ndx), codegen_Lpx_weight(np, ndx);
  MatrixXs direct_Lpu_zero(np, nu), direct_Lpu_weight(np, nu),
      codegen_Lpu_zero(np, nu), codegen_Lpu_weight(np, nu);

  const VectorXs xref = direct_model->get_state()->rand();
  const VectorXs x = direct_model->get_state()->rand();
  const VectorXs w = Scalar(1e-2) * VectorXs::Random(direct_model->get_nu());

  const auto evaluate = [&](const Scalar weight, VectorXs& direct_Lp,
                            MatrixXs& direct_Lpp, MatrixXs& direct_Lpx,
                            MatrixXs& direct_Lpu, VectorXs& codegen_Lp,
                            MatrixXs& codegen_Lpp, MatrixXs& codegen_Lpx,
                            MatrixXs& codegen_Lpu) {
    VectorXs env(nenv);
    env.head(nx) = xref;
    env[nx] = weight;
    set_codegen_observer_state_environment(direct_model, env, "xReg", "pReg");
    codegen_model->update_env(codegen_data, env);

    direct_model->calc(direct_data, x, w);
    direct_model->calcDiff(direct_data, x, w);
    codegen_model->calc(codegen_data, x, w);
    codegen_model->calcDiff(codegen_data, x, w);

    direct_Lp = direct_data->Lp;
    direct_Lpp = direct_data->Lpp;
    direct_Lpx = direct_data->Lpx;
    direct_Lpu = direct_data->Lpu;
    codegen_Lp = codegen_data->Lp;
    codegen_Lpp = codegen_data->Lpp;
    codegen_Lpx = codegen_data->Lpx;
    codegen_Lpu = codegen_data->Lpu;
  };

  evaluate(Scalar(0.), direct_Lp_zero, direct_Lpp_zero, direct_Lpx_zero,
           direct_Lpu_zero, codegen_Lp_zero, codegen_Lpp_zero, codegen_Lpx_zero,
           codegen_Lpu_zero);
  evaluate(p_reg_weight, direct_Lp_weight, direct_Lpp_weight, direct_Lpx_weight,
           direct_Lpu_weight, codegen_Lp_weight, codegen_Lpp_weight,
           codegen_Lpx_weight, codegen_Lpu_weight);

  const VectorXs direct_Lp_delta = direct_Lp_weight - direct_Lp_zero;
  const VectorXs codegen_Lp_delta = codegen_Lp_weight - codegen_Lp_zero;
  const MatrixXs direct_Lpp_delta = direct_Lpp_weight - direct_Lpp_zero;
  const MatrixXs codegen_Lpp_delta = codegen_Lpp_weight - codegen_Lpp_zero;
  const MatrixXs direct_Lpx_delta = direct_Lpx_weight - direct_Lpx_zero;
  const MatrixXs codegen_Lpx_delta = codegen_Lpx_weight - codegen_Lpx_zero;
  const MatrixXs direct_Lpu_delta = direct_Lpu_weight - direct_Lpu_zero;
  const MatrixXs codegen_Lpu_delta = codegen_Lpu_weight - codegen_Lpu_zero;

  (void)expected_Lp_delta;
  check_codegen_matrix_abs_approx(codegen_Lp_delta, direct_Lp_delta, tol,
                                  "codegen/direct pReg env Lp delta");
  check_codegen_matrix_abs_approx(direct_Lpp_delta, expected_Lpp_delta, tol,
                                  "direct pReg env Lpp delta");
  check_codegen_matrix_abs_approx(codegen_Lpp_delta, expected_Lpp_delta, tol,
                                  "codegen pReg env Lpp delta");
  check_codegen_matrix_abs_approx(codegen_Lpp_delta, direct_Lpp_delta, tol,
                                  "codegen/direct pReg env Lpp delta");
  check_codegen_matrix_abs_approx(direct_Lpx_delta, expected_Lpx_delta, tol,
                                  "direct pReg env Lpx delta");
  check_codegen_matrix_abs_approx(codegen_Lpx_delta, expected_Lpx_delta, tol,
                                  "codegen pReg env Lpx delta");
  check_codegen_matrix_abs_approx(codegen_Lpx_delta, direct_Lpx_delta, tol,
                                  "codegen/direct pReg env Lpx delta");
  check_codegen_matrix_abs_approx(direct_Lpu_delta, expected_Lpu_delta, tol,
                                  "direct pReg env Lpu delta");
  check_codegen_matrix_abs_approx(codegen_Lpu_delta, expected_Lpu_delta, tol,
                                  "codegen pReg env Lpu delta");
  check_codegen_matrix_abs_approx(codegen_Lpu_delta, direct_Lpu_delta, tol,
                                  "codegen/direct pReg env Lpu delta");
  BOOST_CHECK_GE(codegen_Lpp_delta.diagonal().minCoeff() + tol, expected_scale);
}

void test_codegen_cost_sum_cast_preserves_np() {
  typedef double Scalar;
  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef crocoddyl::CostModelResidualTpl<Scalar> CostModelResidual;
  typedef crocoddyl::CostModelSumTpl<Scalar> CostModelSum;
  typedef crocoddyl::CostModelSumTpl<ADScalar> ADCostModelSum;
  typedef crocoddyl::DataCollectorParamsTpl<ADScalar> ADDataCollectorParams;
  typedef crocoddyl::ParamsDataAbstractTpl<ADScalar> ADParamsDataAbstract;
  typedef crocoddyl::ResidualModelParametersTpl<Scalar> ResidualModelParameters;
  typedef crocoddyl::StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename crocoddyl::MathBaseTpl<Scalar>::VectorXs VectorXs;

  const pinocchio::ModelTpl<double> modeld = build_codegen_freeflyer_model();
  const std::shared_ptr<StateMultibody> state =
      build_codegen_state_from_model<Scalar>(modeld);
  const std::size_t nu = state->get_nv();
  const std::size_t np = 7u;
  CostModelSum costs(state, nu, np);
  costs.addCost(
      "pReg",
      std::make_shared<CostModelResidual>(
          state, std::make_shared<ResidualModelParameters>(
                     state, VectorXs::Zero(static_cast<Eigen::Index>(np)), nu)),
      Scalar(1.));

  ADCostModelSum casted = costs.template cast<ADScalar>();
  BOOST_CHECK_EQUAL(casted.get_np(), np);
  const std::shared_ptr<ADParamsDataAbstract> params_data =
      std::make_shared<ADParamsDataAbstract>(np, 0u);
  ADDataCollectorParams collector(params_data);
  const std::shared_ptr<crocoddyl::CostDataSumTpl<ADScalar>> data =
      casted.createData(&collector);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Lp.size()), np);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Lpp.rows()), np);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Lpp.cols()), np);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Lpx.rows()), np);
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(data->Lpu.rows()), np);
}

bool init_function() {
  const std::string test_name = "test_codegen";
  test_suite* ts = BOOST_TEST_SUITE(test_name);
  ts->add(BOOST_TEST_CASE(&test_codegen_cost_sum_cast_preserves_np));
  ts->add(BOOST_TEST_CASE(&test_codegen_4DoFArm));
  ts->add(BOOST_TEST_CASE(&test_codegen_bipedal));
  ts->add(BOOST_TEST_CASE(&test_codegen_freeflyer_autodiff_manifold));
  ts->add(BOOST_TEST_CASE(&test_codegen_parameterized_lqr));
  ts->add(BOOST_TEST_CASE(&test_codegen_parameterized_lqr_float32));
  ts->add(BOOST_TEST_CASE(&test_codegen_parameterized_integrators));
  ts->add(BOOST_TEST_CASE(&test_codegen_parameterized_integrators_float32));
  ts->add(BOOST_TEST_CASE(&test_codegen_observer_euler_parameterized_power));
  ts->add(BOOST_TEST_CASE(
      &test_codegen_observer_euler_parameterized_power_float32));
  ts->add(BOOST_TEST_CASE(&test_codegen_observer_state_environment_autodiff));
  ts->add(BOOST_TEST_CASE(
      &test_codegen_observer_state_environment_autodiff_float32));
  ts->add(
      BOOST_TEST_CASE(&test_codegen_observer_state_environment_analytic_lpp));
  framework::master_test_suite().add(ts);

  return true;
}

int main(int argc, char* argv[]) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
