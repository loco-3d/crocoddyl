///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIONS_DIFF_LQR_HPP_
#define CROCODDYL_CORE_ACTIONS_DIFF_LQR_HPP_

#include <stdexcept>

#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/states/euclidean.hpp"

namespace crocoddyl {

template <typename _Scalar>
class DifferentialActionModelLQRTpl
    : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataLQRTpl<Scalar> Data;
  typedef StateVectorTpl<Scalar> StateVector;
  typedef DifferentialActionDataAbstractTpl<Scalar>
      DifferentialActionDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  DifferentialActionModelLQRTpl(const std::size_t nq, const std::size_t nu,
                                const bool drift_free = true);
  virtual ~DifferentialActionModelLQRTpl();

  virtual void calc(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u);
  virtual void calc(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x);
  virtual void calcDiff(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u);
  virtual void calcDiff(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x);
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();
  virtual bool checkData(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data);

  /** @brief Return the position matrix */
  const MatrixXs& get_Aq() const;

  /** @brief Return the velocity matrix */
  const MatrixXs& get_Av() const;

  /** @brief Return the input matrix */
  const MatrixXs& get_B() const;

  /** @brief Return the dynamics drift */
  const VectorXs& get_f() const;

  /** @brief Return the state weight matrix */
  const MatrixXs& get_Q() const;

  /** @brief Return the input weight matrix */
  const MatrixXs& get_R() const;

  /** @brief Return the state-input weight matrix */
  const MatrixXs& get_N() const;

  /** @brief Return the state weight vector */
  const VectorXs& get_q() const;

  /** @brief Return the input weight vector */
  const VectorXs& get_r() const;

  /** @brief Modify the position matrix */
  void set_Aq(const MatrixXs& Aq);

  /** @brief Modify the velocity matrix */
  void set_Av(const MatrixXs& Av);

  /** @brief Modify the input matrix */
  void set_B(const MatrixXs& B);

  /** @brief Modify the dynamics drift */
  void set_f(const VectorXs& f);

  /** @brief Modify the state weight matrix */
  void set_Q(const MatrixXs& Q);

  /** @brief Modify the input weight matrix */
  void set_R(const MatrixXs& R);

  /** @brief Modify the state-input weight matrix */
  void set_N(const MatrixXs& N);

  /** @brief Modify the state weight vector */
  void set_q(const VectorXs& q);

  /** @brief Modify the input weight vector */
  void set_r(const VectorXs& r);

  DEPRECATED("Use get_Aq", const MatrixXs& get_Fq() const { return get_Aq(); })
  DEPRECATED("Use get_Av", const MatrixXs& get_Fv() const { return get_Av(); })
  DEPRECATED("Use get_B", const MatrixXs& get_Fu() const { return get_B(); })
  DEPRECATED("Use get_f", const VectorXs& get_f0() const { return get_f(); })
  DEPRECATED("Use get_q", const VectorXs& get_lx() const { return get_q(); })
  DEPRECATED("Use get_r", const VectorXs& get_lu() const { return get_r(); })
  DEPRECATED("Use get_Q", const MatrixXs& get_Lxx() const { return get_Q(); })
  DEPRECATED("Use get_N", const MatrixXs& get_Lxu() const { return get_N(); })
  DEPRECATED("Use get_R", const MatrixXs& get_Luu() const { return get_R(); })
  DEPRECATED("Use set_Aq", void set_Fq(const MatrixXs& Aq) { set_Aq(Aq); })
  DEPRECATED("Use set_Av", void set_Fv(const MatrixXs& Av) { set_Av(Av); })
  DEPRECATED("Use set_B", void set_Fu(const MatrixXs& B) { set_B(B); })
  DEPRECATED("Use set_f", void set_f0(const VectorXs& f) { set_f(f); })
  DEPRECATED("Use set_q", void set_lx(const VectorXs& q) { set_q(q); })
  DEPRECATED("Use set_r", void set_lu(const VectorXs& r) { set_r(r); })
  DEPRECATED("Use set_Q", void set_Lxx(const MatrixXs& Q) { set_Q(Q); })
  DEPRECATED("Use set_N", void set_Lxu(const MatrixXs& N) { set_N(N); })
  DEPRECATED("Use set_R", void set_Luu(const MatrixXs& R) { set_R(R); })

  /**
   * @brief Print relevant information of the LQR model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::nu_;     //!< Control dimension
  using Base::state_;  //!< Model of the state

 private:
  bool drift_free_;
  MatrixXs Aq_;
  MatrixXs Av_;
  MatrixXs B_;
  MatrixXs Q_;
  MatrixXs R_;
  MatrixXs N_;
  VectorXs f_;
  VectorXs q_;
  VectorXs r_;
};

template <typename _Scalar>
struct DifferentialActionDataLQRTpl
    : public DifferentialActionDataAbstractTpl<_Scalar> {
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataLQRTpl(Model<Scalar>* const model)
      : Base(model) {
    // Setting the linear model and quadratic cost here because they are
    // constant
    Fx.leftCols(model->get_state()->get_nq()) = model->get_Aq();
    Fx.rightCols(model->get_state()->get_nv()) = model->get_Av();
    Fu = model->get_B();
    Lxx = model->get_Q();
    Luu = model->get_R();
    Lxu = model->get_N();
  }

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xout;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/actions/diff-lqr.hxx"
#endif  // CROCODDYL_CORE_ACTIONS_DIFF_LQR_HPP_
