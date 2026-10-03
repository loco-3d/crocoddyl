///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIONS_LQR_HPP_
#define CROCODDYL_CORE_ACTIONS_LQR_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/states/euclidean.hpp"

namespace crocoddyl {

/**
 * @brief Action-parameter model used by parameterized LQR actions
 *
 * The model stores its active parameter segment in the shared action payload.
 * LQR state sensitivity is represented directly by the action model's
 * parameter Jacobian, so computeParamSensitivity() sets dx_dp to zero.
 */
template <typename _Scalar>
class LQRParamsTpl : public ActionModelParamsAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ParamsModelBase, LQRParamsTpl)

  typedef _Scalar Scalar;
  typedef ActionModelParamsAbstractTpl<Scalar> Base;
  typedef StateVectorTpl<Scalar> StateVector;
  typedef typename Base::ActionDataAbstract ActionDataAbstract;
  typedef typename Base::ParamsDataAbstract ParamsDataAbstract;
  typedef typename Base::StateAbstract StateAbstract;
  typedef typename Base::VectorXs VectorXs;
  typedef typename Base::MatrixXs MatrixXs;

  LQRParamsTpl(std::shared_ptr<StateAbstract> state, const std::size_t np);
  LQRParamsTpl(const std::size_t nx, const std::size_t np);
  virtual ~LQRParamsTpl() = default;

  virtual void update(const std::shared_ptr<ParamsDataAbstract>& data,
                      const Eigen::Ref<const VectorXs>& p) override;
  virtual void computeParamSensitivity(
      const std::shared_ptr<ActionDataAbstract>& data,
      const std::shared_ptr<ParamsDataAbstract>& params,
      Eigen::Ref<MatrixXs> dx_dp, const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override;

  template <typename NewScalar>
  LQRParamsTpl<NewScalar> cast() const;

  virtual void print(std::ostream& os) const override;
};

/**
 * @brief Linear-quadratic regulator (LQR) action model
 *
 * With \f$\mathbf{z}=[\mathbf{x};\mathbf{u};\mathbf{p}]\f$, its running
 * dynamics, cost and constraints are
 * \f{equation}{
 * \mathbf{x}^{'}=\mathbf{A x+B u+P p+f}.
 * \f}
 * \f{equation}{
 * \ell=\tfrac12\mathbf{z}^{T}
 * \begin{bmatrix}\mathbf{Q}&\mathbf{N}&\mathbf{Y}\\
 * \mathbf{N}^{T}&\mathbf{R}&\mathbf{V}\\
 * \mathbf{Y}^{T}&\mathbf{V}^{T}&\mathbf{W}\end{bmatrix}\mathbf{z}
 * +[\mathbf{q};\mathbf{r};\mathbf{m}]^{T}\mathbf{z}.
 * \f}
 * \f{equation}{
 * \mathbf{g(x,u,p)}=\mathbf{Gz+g}\leq\mathbf{0},\qquad
 * \mathbf{h(x,u,p)}=\mathbf{Hz+h}.
 * \f}
 * Terminal evaluation keeps only the state and parameter terms. Setting
 * \f$np=0\f$ recovers the legacy LQR model exactly.
 */
template <typename _Scalar>
class ActionModelLQRTpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase, ActionModelLQRTpl)

  typedef _Scalar Scalar;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataLQRTpl<Scalar> Data;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef StateVectorTpl<Scalar> StateVector;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the LQR action model
   *
   * @param[in] A  State matrix
   * @param[in] B  Input matrix
   * @param[in] Q  State weight matrix
   * @param[in] R  Input weight matrix
   * @param[in] N  State-input weight matrix
   */
  ActionModelLQRTpl(const MatrixXs& A, const MatrixXs& B, const MatrixXs& Q,
                    const MatrixXs& R, const MatrixXs& N);

  /**
   * @brief Initialize the LQR action model
   *
   * @param[in] A  State matrix
   * @param[in] B  Input matrix
   * @param[in] Q  State weight matrix
   * @param[in] R  Input weight matrix
   * @param[in] N  State-input weight matrix
   * @param[in] f  Dynamics drift
   * @param[in] q  State weight vector
   * @param[in] r  Input weight vector
   */
  ActionModelLQRTpl(const MatrixXs& A, const MatrixXs& B, const MatrixXs& Q,
                    const MatrixXs& R, const MatrixXs& N, const VectorXs& f,
                    const VectorXs& q, const VectorXs& r);

  /**
   * @brief Initialize the LQR action model
   *
   * @param[in] A  State matrix
   * @param[in] B  Input matrix
   * @param[in] Q  State weight matrix
   * @param[in] R  Input weight matrix
   * @param[in] N  State-input weight matrix
   * @param[in] G  State-input inequality constraint matrix
   * @param[in] H  State-input equality constraint matrix
   * @param[in] f  Dynamics drift
   * @param[in] q  State weight vector
   * @param[in] r  Input weight vector
   * @param[in] g  State-input inequality constraint bias
   * @param[in] h  State-input equality constraint bias
   */
  ActionModelLQRTpl(const MatrixXs& A, const MatrixXs& B, const MatrixXs& Q,
                    const MatrixXs& R, const MatrixXs& N, const MatrixXs& G,
                    const MatrixXs& H, const VectorXs& f, const VectorXs& q,
                    const VectorXs& r, const VectorXs& g, const VectorXs& h);

  /**
   * @brief Initialize a parameterized LQR action model
   *
   * @param[in] A  State matrix
   * @param[in] B  Input matrix
   * @param[in] P  Parameter dynamics matrix
   * @param[in] Q  State weight matrix
   * @param[in] R  Input weight matrix
   * @param[in] N  State-input weight matrix
   * @param[in] W  Parameter weight matrix
   * @param[in] Y  State-parameter weight matrix
   * @param[in] V  Input-parameter weight matrix
   * @param[in] G  State-input-parameter inequality constraint matrix
   * @param[in] H  State-input-parameter equality constraint matrix
   * @param[in] f  Dynamics drift
   * @param[in] q  State weight vector
   * @param[in] r  Input weight vector
   * @param[in] m  Parameter weight vector
   * @param[in] g  Inequality constraint bias
   * @param[in] h  Equality constraint bias
   */
  ActionModelLQRTpl(const MatrixXs& A, const MatrixXs& B, const MatrixXs& P,
                    const MatrixXs& Q, const MatrixXs& R, const MatrixXs& N,
                    const MatrixXs& W, const MatrixXs& Y, const MatrixXs& V,
                    const MatrixXs& G, const MatrixXs& H, const VectorXs& f,
                    const VectorXs& q, const VectorXs& r, const VectorXs& m,
                    const VectorXs& g, const VectorXs& h);

  /**
   * @brief Initialize the LQR action model
   *
   * @param[in] nx         Dimension of state vector
   * @param[in] nu         Dimension of control vector
   * @param[in] drif_free  Enable / disable the bias term of the linear dynamics
   * (default true)
   */
  ActionModelLQRTpl(const std::size_t nx, const std::size_t nu,
                    const bool drift_free = true);

  /**
   * @brief Initialize a parameterized LQR action model
   *
   * @param[in] nx          Dimension of state vector
   * @param[in] nu          Dimension of control vector
   * @param[in] np          Dimension of parameter vector
   * @param[in] ng          Number of inequality constraints (default 0)
   * @param[in] nh          Number of equality constraints (default 0)
   * @param[in] drift_free  Enable / disable the bias term of the linear
   * dynamics
   */
  ActionModelLQRTpl(const std::size_t nx, const std::size_t nu,
                    const std::size_t np, const std::size_t ng = 0,
                    const std::size_t nh = 0, const bool drift_free = true);

  /** @brief Copy constructor */
  ActionModelLQRTpl(const ActionModelLQRTpl& copy);

  virtual ~ActionModelLQRTpl() = default;

  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;
  virtual std::shared_ptr<ActionDataAbstract> createData() override;
  virtual std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;
  virtual void set_params(const std::shared_ptr<ActionDataAbstract>& data,
                          std::shared_ptr<ParameterManager> params) override;
  virtual void update_p(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& p) override;

  /**
   * @brief Cast the LQR model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ActionModelLQRTpl<NewScalar> A action model with the
   * new scalar type.
   */
  template <typename NewScalar>
  ActionModelLQRTpl<NewScalar> cast() const;

  virtual bool checkData(
      const std::shared_ptr<ActionDataAbstract>& data) override;

  /**
   * @brief Create a random LQR model
   *
   * @param[in] nx  State dimension
   * @param[in] nu  Control dimension
   * @param[in] ng  Inequality constraint dimension (default 0)
   * @param[in] nh  Equality constraint dimension (defaul 0)
   */
  static ActionModelLQRTpl Random(const std::size_t nx, const std::size_t nu,
                                  const std::size_t ng = 0,
                                  const std::size_t nh = 0);

  /** @brief Return the state matrix */
  const MatrixXs& get_A() const;

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

  /** @brief Return the parameter dynamics matrix */
  const MatrixXs& get_P() const;

  /** @brief Return the parameter weight matrix */
  const MatrixXs& get_W() const;

  /** @brief Return the state-parameter weight matrix */
  const MatrixXs& get_Y() const;

  /** @brief Return the input-parameter weight matrix */
  const MatrixXs& get_V() const;

  /** @brief Return the state weight vector */
  const VectorXs& get_q() const;

  /** @brief Return the input weight vector */
  const VectorXs& get_r() const;

  /** @brief Return the parameter weight vector */
  const VectorXs& get_m() const;

  /** @brief Return the state-input inequality constraint bias */
  const VectorXs& get_g() const;

  /** @brief Return the state-input equality constraint bias */
  const VectorXs& get_h() const;

  /** @brief Modify the state matrix */
  void set_A(const MatrixXs& A);

  /** @brief Modify the input matrix */
  void set_B(const MatrixXs& B);

  /** @brief Modify the parameter dynamics matrix */
  void set_P(const MatrixXs& P);

  /** @brief Modify the state weight matrix */
  void set_Q(const MatrixXs& Q);

  /** @brief Modify the input weight matrix */
  void set_R(const MatrixXs& R);

  /** @brief Modify the state-input weight matrix */
  void set_N(const MatrixXs& N);

  /** @brief Modify the parameter weight matrix */
  void set_W(const MatrixXs& W);

  /** @brief Modify the state-parameter weight matrix */
  void set_Y(const MatrixXs& Y);

  /** @brief Modify the input-parameter weight matrix */
  void set_V(const MatrixXs& V);

  /** @brief Modify the inequality constraint matrix */
  void set_G(const MatrixXs& G);

  /** @brief Modify the equality constraint matrix */
  void set_H(const MatrixXs& H);

  /** @brief Modify the dynamics drift */
  void set_f(const VectorXs& f);

  /** @brief Modify the state weight vector */
  void set_q(const VectorXs& q);

  /** @brief Modify the input weight vector */
  void set_r(const VectorXs& r);

  /** @brief Modify the parameter weight vector */
  void set_m(const VectorXs& m);

  /** @brief Modify the inequality constraint bias */
  void set_g(const VectorXs& g);

  /** @brief Modify the equality constraint bias */
  void set_h(const VectorXs& h);

  /**
   * @brief Modify the LQR action model
   *
   * @param[in] A  State matrix
   * @param[in] B  Input matrix
   * @param[in] Q  State weight matrix
   * @param[in] R  Input weight matrix
   * @param[in] N  State-input weight matrix
   * @param[in] G  State-input inequality constraint matrix
   * @param[in] H  State-input equality constraint matrix
   * @param[in] f  Dynamics drift
   * @param[in] q  State weight vector
   * @param[in] r  Input weight vector
   * @param[in] g  State-input inequality constraint bias
   * @param[in] h  State-input equality constraint bias
   */
  void set_LQR(const MatrixXs& A, const MatrixXs& B, const MatrixXs& Q,
               const MatrixXs& R, const MatrixXs& N, const MatrixXs& G,
               const MatrixXs& H, const VectorXs& f, const VectorXs& q,
               const VectorXs& r, const VectorXs& g, const VectorXs& h);

  /**
   * @brief Modify the parameterized LQR action model
   */
  void set_LQR(const MatrixXs& A, const MatrixXs& B, const MatrixXs& P,
               const MatrixXs& Q, const MatrixXs& R, const MatrixXs& N,
               const MatrixXs& W, const MatrixXs& Y, const MatrixXs& V,
               const MatrixXs& G, const MatrixXs& H, const VectorXs& f,
               const VectorXs& q, const VectorXs& r, const VectorXs& m,
               const VectorXs& g, const VectorXs& h);

  DEPRECATED("Use get_A", const MatrixXs& get_Fx() const { return get_A(); })
  DEPRECATED("Use get_B", const MatrixXs& get_Fu() const { return get_B(); })
  DEPRECATED("Use get_f", const VectorXs& get_f0() const { return get_f(); })
  DEPRECATED("Use get_q", const VectorXs& get_lx() const { return get_q(); })
  DEPRECATED("Use get_r", const VectorXs& get_lu() const { return get_r(); })
  DEPRECATED("Use get_Q", const MatrixXs& get_Lxx() const { return get_Q(); })
  DEPRECATED("Use get_R", const MatrixXs& get_Lxu() const { return get_R(); })
  DEPRECATED("Use get_N", const MatrixXs& get_Luu() const { return get_N(); })
  DEPRECATED(
      "Use set_LQR", void set_Fx(const MatrixXs& A) {
        set_LQR(A, B_, P_, Q_, R_, N_, W_, Y_, V_, G_, H_, f_, q_, r_, m_, g_,
                h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_Fu(const MatrixXs& B) {
        set_LQR(A_, B, P_, Q_, R_, N_, W_, Y_, V_, G_, H_, f_, q_, r_, m_, g_,
                h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_f0(const VectorXs& f) {
        set_LQR(A_, B_, P_, Q_, R_, N_, W_, Y_, V_, G_, H_, f, q_, r_, m_, g_,
                h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_lx(const VectorXs& q) {
        set_LQR(A_, B_, P_, Q_, R_, N_, W_, Y_, V_, G_, H_, f_, q, r_, m_, g_,
                h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_lu(const VectorXs& r) {
        set_LQR(A_, B_, P_, Q_, R_, N_, W_, Y_, V_, G_, H_, f_, q_, r, m_, g_,
                h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_Lxx(const MatrixXs& Q) {
        set_LQR(A_, B_, P_, Q, R_, N_, W_, Y_, V_, G_, H_, f_, q_, r_, m_, g_,
                h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_Luu(const MatrixXs& R) {
        set_LQR(A_, B_, P_, Q_, R, N_, W_, Y_, V_, G_, H_, f_, q_, r_, m_, g_,
                h_);
      })
  DEPRECATED(
      "Use set_LQR", void set_Lxu(const MatrixXs& N) {
        set_LQR(A_, B_, P_, Q_, R_, N, W_, Y_, V_, G_, H_, f_, q_, r_, m_, g_,
                h_);
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
  using Base::np_;     //!< Parameter dimension
  using Base::nu_;     //!< Control dimension
  using Base::state_;  //!< Model of the state

 private:
  MatrixXs A_;
  MatrixXs B_;
  MatrixXs P_;
  MatrixXs Q_;
  MatrixXs R_;
  MatrixXs N_;
  MatrixXs W_;
  MatrixXs Y_;
  MatrixXs V_;
  MatrixXs G_;
  MatrixXs H_;
  VectorXs f_;
  VectorXs q_;
  VectorXs r_;
  VectorXs m_;
  VectorXs g_;
  VectorXs h_;
  MatrixXs L_;
  bool drift_free_;
  std::shared_ptr<ParameterManager> params_;
};

/**
 * @brief Data for ActionModelLQRTpl
 *
 * Besides the standard action derivatives, this data stores a local parameter
 * vector for standalone use and an optional shared ParameterDataManagerTpl.
 * Calculations use the live aggregate manager value whenever it is attached.
 * The shared pointer retains ownership of the manager data for the lifetime of
 * this data.
 */
template <typename _Scalar>
struct ActionDataLQRTpl : public ActionDataAbstractTpl<_Scalar> {
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef typename MathBase::VectorXs VectorXs;

  template <template <typename Scalar> class Model>
  explicit ActionDataLQRTpl(Model<Scalar>* const model)
      : ActionDataLQRTpl(model, std::shared_ptr<ParameterDataManager>()) {}

  template <template <typename Scalar> class Model>
  explicit ActionDataLQRTpl(
      Model<Scalar>* const model,
      const std::shared_ptr<ParameterDataManager>& params_data)
      : Base(model),
        p(VectorXs::Zero(static_cast<Eigen::Index>(model->get_np()))),
        params(params_data),
        R_u_tmp(VectorXs::Zero(static_cast<Eigen::Index>(model->get_nu()))),
        Q_x_tmp(VectorXs::Zero(
            static_cast<Eigen::Index>(model->get_state()->get_ndx()))),
        W_p_tmp(VectorXs::Zero(static_cast<Eigen::Index>(model->get_np()))) {
    // Setting the linear model and quadratic cost as they are constant
    const std::size_t nx = model->get_state()->get_nx();
    const std::size_t nu = model->get_nu();
    const std::size_t np = model->get_np();
    Fx = model->get_A();
    Fu = model->get_B();
    Fp = model->get_P();
    Lxx = model->get_Q();
    Luu = model->get_R();
    Lxu = model->get_N();
    Lpp = model->get_W();
    Lpx = model->get_Y().transpose();
    Lpu = model->get_V().transpose();
    Gx = model->get_G().leftCols(nx);
    Gu = model->get_G().middleCols(nx, nu);
    Gp = model->get_G().rightCols(np);
    Hx = model->get_H().leftCols(nx);
    Hu = model->get_H().middleCols(nx, nu);
    Hp = model->get_H().rightCols(np);
  }
  virtual ~ActionDataLQRTpl() = default;

  using Base::cost;
  using Base::Fp;
  using Base::Fu;
  using Base::Fx;
  using Base::Gp;
  using Base::Gu;
  using Base::Gx;
  using Base::Hp;
  using Base::Hu;
  using Base::Hx;
  using Base::Lp;
  using Base::Lpp;
  using Base::Lpu;
  using Base::Lpx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xnext;

  VectorXs p;  //!< Standalone LQR parameter vector
  std::shared_ptr<ParameterDataManager> params;
  VectorXs R_u_tmp;  // Temporary variable for storing Hessian-vector product
                     // (size: nu)
  VectorXs Q_x_tmp;  // Temporary variable for storing Hessian-vector product
                     // (size: nx)
  VectorXs W_p_tmp;  // Temporary variable for storing Hessian-vector product
                     // (size: np)
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/actions/lqr.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ActionModelLQRTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ActionDataLQRTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::LQRParamsTpl)

#endif  // CROCODDYL_CORE_ACTIONS_LQR_HPP_
