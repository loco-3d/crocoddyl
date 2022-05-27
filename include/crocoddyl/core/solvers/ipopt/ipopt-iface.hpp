#ifdef CROCODDYL_WITH_IPOPT
#ifndef __CROCODDYL_CORE_SOLVERS_IPOPT_IPOPT_IFACE_HPP__
#define __CROCODDYL_CORE_SOLVERS_IPOPT_IPOPT_IFACE_HPP__

#include <coin-or/IpTNLP.hpp>

#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/mathbase.hpp"

namespace crocoddyl {

struct IpoptInterfaceData;

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

  boost::shared_ptr<IpoptInterfaceData> createData(const std::size_t &nx, const std::size_t &ndx, const std::size_t &nu);

  const std::size_t &get_nvar() const;

  const std::vector<Eigen::VectorXd> &get_xs() const;
  const std::vector<Eigen::VectorXd> &get_us() const;
  const boost::shared_ptr<crocoddyl::ShootingProblem> &get_problem() const;

  void set_xs(const std::vector<Eigen::VectorXd> &xs);
  void set_us(const std::vector<Eigen::VectorXd> &us);

  void set_consider_control_bounds(const bool &consider_bounds);

 private:
  boost::shared_ptr<crocoddyl::ShootingProblem> problem_;

  std::vector<Eigen::VectorXd> xs_;  // before solve: store initial; after solve: store solution
  std::vector<Eigen::VectorXd> us_;  // before solve: store initial; after solve: store solution

  std::size_t T_;

  std::size_t nconst_;
  std::size_t nvar_;

  std::vector<boost::shared_ptr<IpoptInterfaceData>> datas_;

  bool consider_control_bounds_;

  IpoptInterface(const IpoptInterface &);

  IpoptInterface &operator=(const IpoptInterface &);
};

struct IpoptInterfaceData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IpoptInterfaceData(const std::size_t &nx, const std::size_t &ndx, const std::size_t &nu)
      : x(nx),
        xnext(nx),
        dx(ndx),
        dxnext(ndx),
        x_diff(ndx),
        u(nu),
        Jsum_x(ndx, ndx),
        Jsum_dx(ndx, ndx),
        Jsum_xnext(ndx, ndx),
        Jsum_dxnext(ndx, ndx),
        Jdiff_xnext(ndx, ndx),
        Jdiff_x(ndx, ndx),
        Jg_dx(ndx, ndx),
        Jg_dxnext(ndx, ndx),
        Jg_u(ndx, ndx),
        Jg_ic(ndx, ndx),
        Ldx(ndx),
        Ldxdx(ndx, ndx),
        Ldxu(ndx, nu) {
    x.setZero();
    xnext.setZero();
    dx.setZero();
    dxnext.setZero();
    x_diff.setZero();

    u.setZero();

    Jsum_x.setZero();
    Jsum_dx.setZero();
    Jsum_xnext.setZero();
    Jsum_dxnext.setZero();

    Jdiff_xnext.setZero();
    Jdiff_x.setZero();

    Jg_dx.setZero();
    Jg_dxnext.setZero();
    Jg_u.setZero();
    Jg_ic.setZero();

    Ldx.setZero();
    Ldxdx.setZero();
    Ldxu.setZero();
  }

  Eigen::VectorXd x;
  Eigen::VectorXd xnext; // Mighnt not be using that
  Eigen::VectorXd dx;
  Eigen::VectorXd dxnext; // Imight not be using that
  Eigen::VectorXd x_diff;
  Eigen::VectorXd u;

  Eigen::MatrixXd Jsum_x;
  Eigen::MatrixXd Jsum_dx;

  Eigen::MatrixXd Jsum_xnext; // might not be using that
  Eigen::MatrixXd Jsum_dxnext; // might not be using that

  Eigen::MatrixXd Jdiff_xnext; // might not be using that (if removed from initial condition)
  Eigen::MatrixXd Jdiff_x;

  Eigen::MatrixXd Jg_dx;
  Eigen::MatrixXd Jg_dxnext; // might not be using that
  Eigen::MatrixXd Jg_u;
  Eigen::MatrixXd Jg_ic;

  Eigen::VectorXd Ldx;
  Eigen::MatrixXd Ldxdx;
  Eigen::MatrixXd Ldxu;
};

}  // namespace crocoddyl

#endif
#endif
