///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_DDP_DEPRECATED_HPP_
#define CROCODDYL_CORE_SOLVERS_DDP_DEPRECATED_HPP_

#include "crocoddyl/core/solvers/fddp.hpp"

namespace crocoddyl {

/**
 * @brief Differential Dynamic Programming (DDP) solver
 *
 * The DDP solver computes an optimal trajectory and control commands by
 * iterates running `backwardPass()` and `forwardPass()`. The backward-pass
 * updates locally the quadratic approximation of the problem and computes
 * descent direction. If the warm-start is feasible, then it computes the gaps
 * \f$\mathbf{f}_s\f$ and run a modified Riccati sweep: \f{eqnarray*}
 *   \mathbf{Q}_{\mathbf{x}_k} &=& \mathbf{l}_{\mathbf{x}_k} +
 * \mathbf{f}^\top_{\mathbf{x}_k} (V_{\mathbf{x}_{k+1}} +
 * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
 *   \mathbf{Q}_{\mathbf{u}_k} &=& \mathbf{l}_{\mathbf{u}_k} +
 * \mathbf{f}^\top_{\mathbf{u}_k} (V_{\mathbf{x}_{k+1}} +
 * V_{\mathbf{xx}_{k+1}}\mathbf{\bar{f}}_{k+1}),\\
 *   \mathbf{Q}_{\mathbf{xx}_k} &=& \mathbf{l}_{\mathbf{xx}_k} +
 * \mathbf{f}^\top_{\mathbf{x}_k} V_{\mathbf{xx}_{k+1}}
 * \mathbf{f}_{\mathbf{x}_k},\\
 *   \mathbf{Q}_{\mathbf{xu}_k} &=& \mathbf{l}_{\mathbf{xu}_k} +
 * \mathbf{f}^\top_{\mathbf{x}_k} V_{\mathbf{xx}_{k+1}}
 * \mathbf{f}_{\mathbf{u}_k},\\
 *   \mathbf{Q}_{\mathbf{uu}_k} &=& \mathbf{l}_{\mathbf{uu}_k} +
 * \mathbf{f}^\top_{\mathbf{u}_k} V_{\mathbf{xx}_{k+1}}
 * \mathbf{f}_{\mathbf{u}_k}.
 * \f}
 * Then, the forward-pass rollouts this new policy by integrating the system
 * dynamics along a tuple of optimized control commands \f$\mathbf{u}^*_s\f$,
 * i.e. \f{eqnarray}
 *   \mathbf{\hat{x}}_0 &=& \mathbf{\tilde{x}}_0,\\
 *   \mathbf{\hat{u}}_k &=& \mathbf{u}_k + \alpha\mathbf{k}_k +
 * \mathbf{K}_k(\mathbf{\hat{x}}_k-\mathbf{x}_k),\\ \mathbf{\hat{x}}_{k+1} &=&
 * \mathbf{f}_k(\mathbf{\hat{x}}_k,\mathbf{\hat{u}}_k). \f}
 *
 * \sa SolverAbstract(), `backwardPass()` and `forwardPass()`
 */
class SolverDDP : public SolverFDDPTpl<double> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the DDP solver
   *
   * @param[in] problem  shooting problem
   */
  DEPRECATED(
      "Do not use SolverDDP. Instead, you should use SolverFDDP(problem, "
      "DynamicsSolverType::SingleShoot)",
      explicit SolverDDP(std::shared_ptr<ShootingProblemTpl<double>>
                             problem) : SolverFDDPTpl<double>(problem){};)
  virtual ~SolverDDP() = default;

  virtual void calcDiff() { return SolverFDDPTpl<double>::calcDir(); }
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_DDP_DEPRECATED_HPP_
