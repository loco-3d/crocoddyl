#ifdef CROCODDYL_WITH_IPOPT
#ifndef __CROCODDYL_CORE_SOLVERS_IPOPT_IPOPT_IFACE_HPP__
#define __CROCODDYL_CORE_SOLVERS_IPOPT_IPOPT_IFACE_HPP__

#include <coin-or/IpTNLP.hpp>

#include "crocoddyl/core/optctrl/shooting.hpp"

class IpoptInterface : public Ipopt::TNLP {
 public:
  IpoptInterface(const boost::shared_ptr<crocoddyl::ShootingProblem> &problem);

  virtual ~IpoptInterface();

  virtual bool get_nlp_info(Ipopt::Index &n, Ipopt::Index &m, Ipopt::Index &nnz_jac_g, Ipopt::Index &nnz_h_lag,
                            IndexStyleEnum &index_style);

  virtual bool get_bounds_info(Ipopt::Index n, Ipopt::Number *x_l, Ipopt::Number *x_u, Ipopt::Index m,
                               Ipopt::Number *g_l, Ipopt::Number *g_u);

  virtual bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number *x, bool init_z, Ipopt::Number *z_L,
                                  Ipopt::Number *z_U, Ipopt::Index m, bool init_lambda, Ipopt::Number *lambda);

  virtual bool eval_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number &obj_value);

  virtual bool eval_grad_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number *grad_f);

  virtual bool eval_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Index m, Ipopt::Number *g);

  virtual bool eval_jac_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Index m, Ipopt::Index nele_jac,
                          Ipopt::Index *iRow, Ipopt::Index *jCol, Ipopt::Number *values);

  virtual bool eval_h(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number obj_factor, Ipopt::Index m,
                      const Ipopt::Number *lambda, bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index *iRow,
                      Ipopt::Index *jCol, Ipopt::Number *values);

  virtual void finalize_solution(Ipopt::SolverReturn status, Ipopt::Index n, const Ipopt::Number *x,
                                 const Ipopt::Number *z_L, const Ipopt::Number *z_U, Ipopt::Index m,
                                 const Ipopt::Number *g, const Ipopt::Number *lambda, Ipopt::Number obj_value,
                                 const Ipopt::IpoptData *ip_data, Ipopt::IpoptCalculatedQuantities *ip_cq);

  bool intermediate_callback(Ipopt::AlgorithmMode mode, Ipopt::Index iter, Ipopt::Number obj_value,
                             Ipopt::Number inf_pr, Ipopt::Number inf_du, Ipopt::Number mu, Ipopt::Number d_norm,
                             Ipopt::Number regularization_size, Ipopt::Number alpha_du, Ipopt::Number alpha_pr,
                             Ipopt::Index ls_trials, const Ipopt::IpoptData *ip_data,
                             Ipopt::IpoptCalculatedQuantities *ip_cq);

  const std::size_t &get_nvar() const;

  const std::vector<Eigen::VectorXd> &get_xs() const;
  const std::vector<Eigen::VectorXd> &get_us() const;
  const boost::shared_ptr<crocoddyl::ShootingProblem> &get_problem() const;

  void set_xs(const std::vector<Eigen::VectorXd> &xs);
  void set_us(const std::vector<Eigen::VectorXd> &us);

 private:
  boost::shared_ptr<crocoddyl::ShootingProblem> problem_;
  boost::shared_ptr<crocoddyl::StateAbstract> state_;

  std::vector<Eigen::VectorXd> xs_;
  std::vector<Eigen::VectorXd> us_;

  std::size_t nx_;
  std::size_t ndx_;
  std::size_t nu_;
  std::size_t T_;

  std::size_t nconst_;
  std::size_t nvar_;

  struct Data {
    Eigen::VectorXd x;
    Eigen::VectorXd xnext;
    Eigen::VectorXd dx;
    Eigen::VectorXd dxnext;
    Eigen::VectorXd x_diff;
    Eigen::VectorXd control;

    Eigen::MatrixXd Jsum_x;
    Eigen::MatrixXd Jsum_dx;

    Eigen::MatrixXd Jsum_xnext;
    Eigen::MatrixXd Jsum_dxnext;

    Eigen::MatrixXd Jdiff_xnext;
    Eigen::MatrixXd Jdiff_x;

    Eigen::MatrixXd Jg_dx;
    Eigen::MatrixXd Jg_dxnext;
    Eigen::MatrixXd Jg_u;

    Eigen::VectorXd Ldx;
    Eigen::MatrixXd Ldxdx;
    Eigen::MatrixXd Ldxu;
  } data_;

  IpoptInterface(const IpoptInterface &);

  IpoptInterface &operator=(const IpoptInterface &);
};

#endif
#endif
