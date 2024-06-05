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

  /**
   * @brief Initialize the LQR action model
   *
   * @param[in] Aq  Position matrix
   * @param[in] Av  Velocity matrix
   * @param[in] B   Input matrix
   * @param[in] Q   State weight matrix
   * @param[in] R   Input weight matrix
   * @param[in] N   State-input weight matrix
   */
  DifferentialActionModelLQRTpl(const MatrixXs& Aq, const MatrixXs& Av,
                                const MatrixXs& B, const MatrixXs& Q,
                                const MatrixXs& R, const MatrixXs& N);

  /**
   * @brief Initialize the LQR action model
   *
   * @param[in] Aq  Position matrix
   * @param[in] Av  Velocity matrix
   * @param[in] B   Input matrix
   * @param[in] Q   State weight matrix
   * @param[in] R   Input weight matrix
   * @param[in] N   State-input weight matrix
   * @param[in] f   Dynamics drift
   * @param[in] q   State weight vector
   * @param[in] r   Input weight vector
   */
  DifferentialActionModelLQRTpl(const MatrixXs& Aq, const MatrixXs& Av,
                                const MatrixXs& B, const MatrixXs& Q,
                                const MatrixXs& R, const MatrixXs& N,
                                const VectorXs& f, const VectorXs& q,
                                const VectorXs& r);

  /**
   * @brief Initialize the LQR action model
   *
   * @param[in] nq         Dimension of position vector
   * @param[in] nu         Dimension of control vector
   * @param[in] drif_free  Enable / disable the bias term of the linear dynamics
   * (default true)
   */
  DifferentialActionModelLQRTpl(const std::size_t nq, const std::size_t nu,
                                const bool drift_free = true);

  /** @brief Copy constructor */
  DifferentialActionModelLQRTpl(const DifferentialActionModelLQRTpl& copy);

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

  /**
   * @brief Create a random LQR model
   *
   * @param[in] nq  Position dimension
   * @param[in] nu  Control dimension
   */
  static DifferentialActionModelLQRTpl Random(const std::size_t nq,
                                              const std::size_t nu);

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

  /**
   * @brief Modify the LQR action model
   *
   * @param[in] Aq  Position matrix
   * @param[in] Av  Velocity matrix
   * @param[in] B   Input matrix
   * @param[in] Q   State weight matrix
   * @param[in] R   Input weight matrix
   * @param[in] N   State-input weight matrix
   * @param[in] f   Dynamics drift
   * @param[in] q   State weight vector
   * @param[in] r   Input weight vector
   */
  void set_LQR(const MatrixXs& Aq, const MatrixXs& Av, const MatrixXs& B,
               const MatrixXs& Q, const MatrixXs& R, const MatrixXs& N,
               const VectorXs& f, const VectorXs& q, const VectorXs& r);

  DEPRECATED("Use get_Aq", const MatrixXs& get_Fq() const { return get_Aq(); })
  DEPRECATED("Use get_Av", const MatrixXs& get_Fv() const { return get_Av(); })
  DEPRECATED("Use get_B", const MatrixXs& get_Fu() const { return get_B(); })
  DEPRECATED("Use get_f", const VectorXs& get_f0() const { return get_f(); })
  DEPRECATED("Use get_q", const VectorXs& get_lx() const { return get_q(); })
  DEPRECATED("Use get_r", const VectorXs& get_lu() const { return get_r(); })
  DEPRECATED("Use get_Q", const MatrixXs& get_Lxx() const { return get_Q(); })
  DEPRECATED("Use get_N", const MatrixXs& get_Lxu() const { return get_N(); })
  DEPRECATED("Use get_R", const MatrixXs& get_Luu() const { return get_R(); })
  DEPRECATED(
      "Use set_LQR", void set_Fq(const MatrixXs& Aq) {
        set_LQR(Aq, Av_, B_, Q_, R_, N_, f_, q_, r_);
      })
  DEPRECATED(
      "Use set_LQR", void set_Fv(const MatrixXs& Av) {
        set_LQR(Aq_, Av, B_, Q_, R_, N_, f_, q_, r_);
      })
  DEPRECATED(
      "Use set_LQR", void set_Fu(const MatrixXs& B) {
        set_LQR(Aq_, Av_, B, Q_, R_, N_, f_, q_, r_);
      })
  DEPRECATED(
      "Use set_LQR", void set_f0(const VectorXs& f) {
        set_LQR(Aq_, Av_, B_, Q_, R_, N_, f, q_, r_);
      })
  DEPRECATED(
      "Use set_LQR", void set_lx(const VectorXs& q) {
        set_LQR(Aq_, Av_, B_, Q_, R_, N_, f_, q, r_);
      })
  DEPRECATED(
      "Use set_LQR", void set_lu(const VectorXs& r) {
        set_LQR(Aq_, Av_, B_, Q_, R_, N_, f_, q_, r);
      })
  DEPRECATED(
      "Use set_LQR", void set_Lxx(const MatrixXs& Q) {
        set_LQR(Aq_, Av_, B_, Q, R_, N_, f_, q_, r_);
      })
  DEPRECATED(
      "Use set_LQR", void set_Lxu(const MatrixXs& N) {
        set_LQR(Aq_, Av_, B_, Q_, R_, N, f_, q_, r_);
      })
  DEPRECATED(
      "Use set_LQR", void set_Luu(const MatrixXs& R) {
        set_LQR(Aq_, Av_, B_, Q_, R, N_, f_, q_, r_);
      })

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
  MatrixXs Aq_;
  MatrixXs Av_;
  MatrixXs B_;
  MatrixXs Q_;
  MatrixXs R_;
  MatrixXs N_;
  VectorXs f_;
  VectorXs q_;
  VectorXs r_;
  MatrixXs H_;
  bool drift_free_;
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
