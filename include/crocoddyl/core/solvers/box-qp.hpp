///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
#define CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_

#include "crocoddyl/core/utils/exception.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <vector>

namespace crocoddyl {

struct BoxQPSolution {
  BoxQPSolution() {}
  BoxQPSolution(const Eigen::MatrixXd& Hff_inv, const Eigen::VectorXd& x, const std::vector<size_t>& free_idx,
                const std::vector<size_t>& clamped_idx)
      : Hff_inv(Hff_inv), x(x), free_idx(free_idx), clamped_idx(clamped_idx) {}

  Eigen::MatrixXd Hff_inv;
  Eigen::MatrixXd x;
  std::vector<size_t> free_idx;
  std::vector<size_t> clamped_idx;
};

// Based on Yuval Tassa's BoxQP
// Cf. https://www.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization
class BoxQP {
 public:
  BoxQP(const std::size_t nx, std::size_t maxiter = 100, const double th_acceptstep = 0.1, const double th_grad = 1e-9,
        const double reg = 1e-9);
  ~BoxQP();

  const BoxQPSolution& solve(const Eigen::MatrixXd& H, const Eigen::VectorXd& q, const Eigen::VectorXd& lb,
                             const Eigen::VectorXd& ub, const Eigen::VectorXd& xinit);
  const BoxQPSolution& get_solution() const;
  const std::size_t& get_nx() const;
  const std::size_t& get_maxiter() const;
  const double& get_th_acceptstep() const;
  const double& get_th_grad() const;
  const double& get_reg() const;
  const std::vector<double>& get_alphas() const;
  void set_nx(const std::size_t& nx);
  void set_maxiter(const std::size_t& maxiter);
  void set_th_acceptstep(const double& th_acceptstep);
  void set_th_grad(const double& th_grad);
  void set_reg(const double& reg);
  void set_alphas(const std::vector<double>& alphas);

 private:
  std::size_t nx_;
  BoxQPSolution solution_;
  std::size_t maxiter_;
  double th_acceptstep_;
  double th_grad_;
  double reg_;

  std::size_t nf_;
  std::size_t nc_;
  std::vector<double> alphas_;
  Eigen::VectorXd x_;
  Eigen::VectorXd xnew_;
  Eigen::VectorXd g_;
  Eigen::VectorXd dx_;

  Eigen::VectorXd qf_;
  Eigen::VectorXd xf_;
  Eigen::VectorXd xc_;
  Eigen::VectorXd dxf_;
  Eigen::MatrixXd Hff_;
  Eigen::MatrixXd Hff_inv_;
  Eigen::MatrixXd Hfc_;
  Eigen::LLT<Eigen::MatrixXd> Hff_inv_llt_;
  std::vector<size_t> clamped_idx_;
  std::vector<size_t> free_idx_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------------- */
#include "crocoddyl/core/solvers/box-qp.hxx"

#endif  // CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
