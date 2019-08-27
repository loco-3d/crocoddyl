///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
#define CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_

#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/multibody/costs/cost-sum.hpp"
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace crocoddyl {

struct DifferentialActionDataAbstract;  // forward declaration

class DifferentialActionModelAbstract {
 public:
  DifferentialActionModelAbstract(StateAbstract& state, unsigned int const& nu, unsigned int const& nr = 1);
  virtual ~DifferentialActionModelAbstract();

  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                        const bool& recalc = true) = 0;
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x);
  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x);

  unsigned int const& get_nu() const;
  unsigned int const& get_nr() const;
  StateAbstract& get_state() const;

 protected:
  unsigned int nu_;
  unsigned int nr_;
  StateAbstract& state_;
  Eigen::VectorXd unone_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::VectorXd& x,
                 const Eigen::VectorXd& u = Eigen::VectorXd()) {
    if (u.size() == 0) {
      calc(data, x);
    } else {
      calc(data, x, u);
    }
  }

  void calcDiff_wrap(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::VectorXd& x,
                     const Eigen::VectorXd& u, const bool& recalc) {
    calcDiff(data, x, u, recalc);
  }
  void calcDiff_wrap(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::VectorXd& x,
                     const Eigen::VectorXd& u) {
    calcDiff(data, x, u, true);
  }
  void calcDiff_wrap(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::VectorXd& x) {
    calcDiff(data, x, unone_, true);
  }
  void calcDiff_wrap(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::VectorXd& x,
                     const bool& recalc) {
    calcDiff(data, x, unone_, recalc);
  }

#endif
};

struct DifferentialActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  explicit DifferentialActionDataAbstract(Model* const model)
      : cost(0.),
        xout(model->get_state().get_nv()),
        r(model->get_nr()),
        qcur(model->get_state().get_nq()),
        vcur(model->get_state().get_nv()),
        Fx(model->get_state().get_nv(), model->get_state().get_ndx()),
        Fu(model->get_state().get_nv(), model->get_nu()),
        Lx(model->get_state().get_ndx()),
        Lu(model->get_nu()),
        Lxx(model->get_state().get_ndx(), model->get_state().get_ndx()),
        Lxu(model->get_state().get_ndx(), model->get_nu()),
        Luu(model->get_nu(), model->get_nu()),
        r_ref(&r(0), model->get_nr()),
        Lx_ref(&Lx(0), model->get_state().get_ndx()),
        Lu_ref(&Lu(0), model->get_nu()),
        Lxx_ref(&Lxx(0), model->get_state().get_ndx(), model->get_state().get_ndx()),
        Lxu_ref(&Lxu(0), model->get_state().get_ndx(), model->get_nu()),
        Luu_ref(&Luu(0), model->get_nu(), model->get_nu()) {
    xout.setZero();
    r.setZero();
    qcur.setZero();
    vcur.setZero();
    Fx.setZero();
    Fu.setZero();
    Lx.setZero();
    Lu.setZero();
    Lxx.setZero();
    Lxu.setZero();
    Luu.setZero();
  }

  void shareCostMemory(const boost::shared_ptr<CostDataSum>& costs) {
    // Share memory with the cost data
    new (&r_ref) Eigen::Map<Eigen::VectorXd>(&costs->r(0), costs->r.size());
    new (&Lx_ref) Eigen::Map<Eigen::VectorXd>(&costs->Lx(0), costs->Lx.size());
    new (&Lu_ref) Eigen::Map<Eigen::VectorXd>(&costs->Lu(0), costs->Lu.size());
    new (&Lxx_ref) Eigen::Map<Eigen::MatrixXd>(&costs->Lxx(0), costs->Lxx.rows(), costs->Lxx.cols());
    new (&Lxu_ref) Eigen::Map<Eigen::MatrixXd>(&costs->Lxu(0), costs->Lxu.rows(), costs->Lxu.cols());
    new (&Luu_ref) Eigen::Map<Eigen::MatrixXd>(&costs->Luu(0), costs->Luu.rows(), costs->Luu.cols());
  }

  const double& get_cost() const { return cost; }
  const Eigen::VectorXd& get_xout() const { return xout; }

  Eigen::VectorXd get_r() const { return r_ref; }
  Eigen::VectorXd get_Lx() const { return Lx_ref; }
  Eigen::VectorXd get_Lu() const { return Lu_ref; }
  Eigen::MatrixXd get_Lxx() const { return Lxx_ref; }
  Eigen::MatrixXd get_Lxu() const { return Lxu_ref; }
  Eigen::MatrixXd get_Luu() const { return Luu_ref; }
  const Eigen::MatrixXd& get_Fx() const { return Fx; }
  const Eigen::MatrixXd& get_Fu() const { return Fu; }

  void set_r(Eigen::VectorXd _r) { r_ref = _r; }
  void set_Lx(Eigen::VectorXd _Lx) { Lx_ref = _Lx; }
  void set_Lu(Eigen::VectorXd _Lu) { Lu_ref = _Lu; }
  void set_Lxx(Eigen::MatrixXd _Lxx) { Lxx_ref = _Lxx; }
  void set_Lxu(Eigen::MatrixXd _Lxu) { Lxu_ref = _Lxu; }
  void set_Luu(Eigen::MatrixXd _Luu) { Luu_ref = _Luu; }

  double cost;
  Eigen::VectorXd xout;
  Eigen::VectorXd r;
  Eigen::VectorXd qcur;
  Eigen::VectorXd vcur;
  Eigen::MatrixXd Fx;
  Eigen::MatrixXd Fu;
  Eigen::VectorXd Lx;
  Eigen::VectorXd Lu;
  Eigen::MatrixXd Lxx;
  Eigen::MatrixXd Lxu;
  Eigen::MatrixXd Luu;

  // Using Map for sharing memory with cost-sum data
  Eigen::Map<Eigen::VectorXd> r_ref;
  Eigen::Map<Eigen::VectorXd> Lx_ref;
  Eigen::Map<Eigen::VectorXd> Lu_ref;
  Eigen::Map<Eigen::MatrixXd> Lxx_ref;
  Eigen::Map<Eigen::MatrixXd> Lxu_ref;
  Eigen::Map<Eigen::MatrixXd> Luu_ref;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
