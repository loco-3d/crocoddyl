///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIONS_DIFF_LQR_HPP_
#define CROCODDYL_CORE_ACTIONS_DIFF_LQR_HPP_

#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/states/euclidean.hpp"

namespace crocoddyl {

/**
 * @brief Linear-quadratic regulator (LQR) differential action model
 *
 * A linear-quadratic regulator (LQR) action has a transition model of the form
 * \f[ \begin{equation}
 *   \mathbf{\dot{v}} = \mathbf{A_q q + A_v v + B u + f}.
 * \end{equation} \f]
 * Its cost function is quadratic of the form:
 * \f[ \begin{equation}
 * \ell(\mathbf{x},\mathbf{u}) = \begin{bmatrix}1
 * \\ \mathbf{x} \\ \mathbf{u}\end{bmatrix}^T \begin{bmatrix}0 &
 * \mathbf{q}^T & \mathbf{r}^T \\ \mathbf{q} & \mathbf{Q}
 * &
 * \mathbf{N}^T \\
 * \mathbf{r} & \mathbf{N} & \mathbf{R}\end{bmatrix}
 * \begin{bmatrix}1 \\ \mathbf{x} \\
 * \mathbf{u}\end{bmatrix}
 * \end{equation} \f]
 * and the linear equality and inequality constraints has the form:
 * \f[ \begin{aligned}
 * \mathbf{g(x,u)} =  \mathbf{G}\begin{bmatrix} \mathbf{x} \\ \mathbf{u}
 * \end{bmatrix} [x,u] + \mathbf{g} \leq \mathbf{0}
 * &\mathbf{h(x,u)} = \mathbf{H}\begin{bmatrix} \mathbf{x} \\ \mathbf{u}
 * \end{bmatrix} [x,u] + \mathbf{h} \end{aligned} \f]
 */
template <typename _Scalar>
class DifferentialActionModelLQRTpl
    : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(DifferentialActionModelBase,
                         DifferentialActionModelLQRTpl)

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
   * @param[in] Aq  Position matrix
   * @param[in] Av  Velocity matrix
   * @param[in] B   Input matrix
   * @param[in] Q   State weight matrix
   * @param[in] R   Input weight matrix
   * @param[in] N   State-input weight matrix
   * @param[in] G   State-input inequality constraint matrix
   * @param[in] H   State-input equality constraint matrix
   * @param[in] f   Dynamics drift
   * @param[in] q   State weight vector
   * @param[in] r   Input weight vector
   * @param[in] g   State-input inequality constraint bias
   * @param[in] h   State-input equality constraint bias
   */
  DifferentialActionModelLQRTpl(const MatrixXs& Aq, const MatrixXs& Av,
                                const MatrixXs& B, const MatrixXs& Q,
                                const MatrixXs& R, const MatrixXs& N,
                                const MatrixXs& G, const MatrixXs& H,
                                const VectorXs& f, const VectorXs& q,
                                const VectorXs& r, const VectorXs& g,
                                const VectorXs& h);

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

  virtual ~DifferentialActionModelLQRTpl() = default;

  virtual void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;
  virtual void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;
  virtual void calcDiff(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override;
  virtual void calcDiff(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x) override;
  virtual std::shared_ptr<DifferentialActionDataAbstract> createData() override;

  /**
   * @brief Cast the differential-LQR model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return DifferentialActionModelLQRTpl<NewScalar> A differential-action
   * model with the new scalar type.
   */
  template <typename NewScalar>
  DifferentialActionModelLQRTpl<NewScalar> cast() const;

  virtual bool checkData(
      const std::shared_ptr<DifferentialActionDataAbstract>& data) override;

  /**
   * @brief Create a random LQR model
   *
   * @param[in] nq  Position dimension
   * @param[in] nu  Control dimension
   * @param[in] ng  Inequality constraint dimension (default 0)
   * @param[in] nh  Equality constraint dimension (default 0)
   */
  static DifferentialActionModelLQRTpl Random(const std::size_t nq,
                                              const std::size_t nu,
                                              const std::size_t ng = 0,
                                              const std::size_t nh = 0);

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

  /** @brief Return the state-input inequality constraint matrix */
  const MatrixXs& get_G() const;

  /** @brief Return the state-input equality constraint matrix */
  const MatrixXs& get_H() const;

  /** @brief Return the state weight vector */
  const VectorXs& get_q() const;

  /** @brief Return the input weight vector */
  const VectorXs& get_r() const;

  /** @brief Return the state-input inequality constraint bias */
  const VectorXs& get_g() const;

  /** @brief Return the state-input equality constraint bias */
  const VectorXs& get_h() const;

  /**
   * @brief Modify the LQR action model
   *
   * @param[in] Aq  Position matrix
   * @param[in] Av  Velocity matrix
   * @param[in] B   Input matrix
   * @param[in] Q   State weight matrix
   * @param[in] R   Input weight matrix
   * @param[in] N   State-input weight matrix
   * @param[in] G   State-input inequality constraint matrix
   * @param[in] H   State-input equality constraint matrix
   * @param[in] f   Dynamics drift
   * @param[in] q   State weight vector
   * @param[in] r   Input weight vector
   * @param[in] g   State-input inequality constraint bias
   * @param[in] h   State-input equality constraint bias
   */
  void set_LQR(const MatrixXs& Aq, const MatrixXs& Av, const MatrixXs& B,
               const MatrixXs& Q, const MatrixXs& R, const MatrixXs& N,
               const MatrixXs& G, const MatrixXs& H, const VectorXs& f,
               const VectorXs& q, const VectorXs& r, const VectorXs& g,
               const VectorXs& h);

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
        set_LQR(Aq, Av_, B_, Q_, R_, N_, G_, H_, f_, q_, r_, g_, h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_Fv(const MatrixXs& Av) {
        set_LQR(Aq_, Av, B_, Q_, R_, N_, G_, H_, f_, q_, r_, g_, h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_Fu(const MatrixXs& B) {
        set_LQR(Aq_, Av_, B, Q_, R_, N_, G_, H_, f_, q_, r_, g_, h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_f0(const VectorXs& f) {
        set_LQR(Aq_, Av_, B_, Q_, R_, N_, G_, H_, f, q_, r_, g_, h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_lx(const VectorXs& q) {
        set_LQR(Aq_, Av_, B_, Q_, R_, N_, G_, H_, f_, q, r_, g_, h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_lu(const VectorXs& r) {
        set_LQR(Aq_, Av_, B_, Q_, R_, N_, G_, H_, f_, q_, r, g_, h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_Lxx(const MatrixXs& Q) {
        set_LQR(Aq_, Av_, B_, Q, R_, N_, G_, H_, f_, q_, r_, g_, h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_Lxu(const MatrixXs& N) {
        set_LQR(Aq_, Av_, B_, Q_, R_, N, G_, H_, f_, q_, r_, g_, h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_Luu(const MatrixXs& R) {
        set_LQR(Aq_, Av_, B_, Q_, R, N_, G_, H_, f_, q_, r_, g_, h_);
      })

  /**
   * @brief Print relevant information of the LQR model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::ng_;     //!< Equality constraint dimension
  using Base::nh_;     //!< Inequality constraint dimension
  using Base::nu_;     //!< Control dimension
  using Base::state_;  //!< Model of the state

 private:
  MatrixXs Aq_;
  MatrixXs Av_;
  MatrixXs B_;
  MatrixXs Q_;
  MatrixXs R_;
  MatrixXs N_;
  MatrixXs G_;
  MatrixXs H_;
  VectorXs f_;
  VectorXs q_;
  VectorXs r_;
  VectorXs g_;
  VectorXs h_;
  MatrixXs L_;
  bool drift_free_;
  bool updated_lqr_;
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
    // Setting the linear model and quadratic cost as they are constant
    const std::size_t nq = model->get_state()->get_nq();
    const std::size_t nu = model->get_nu();
    Fx.leftCols(nq) = model->get_Aq();
    Fx.rightCols(nq) = model->get_Av();
    Fu = model->get_B();
    Lxx = model->get_Q();
    Luu = model->get_R();
    Lxu = model->get_N();
    Gx = model->get_G().leftCols(2 * nq);
    Gu = model->get_G().rightCols(nu);
    Hx = model->get_H().leftCols(2 * nq);
    Hu = model->get_H().rightCols(nu);
  }
  virtual ~DifferentialActionDataLQRTpl() = default;

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Gu;
  using Base::Gx;
  using Base::Hu;
  using Base::Hx;
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

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::DifferentialActionModelLQRTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DifferentialActionDataLQRTpl)

#endif  // CROCODDYL_CORE_ACTIONS_DIFF_LQR_HPP_
