///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIONS_LQR_HPP_
#define CROCODDYL_CORE_ACTIONS_LQR_HPP_

#include <stdexcept>

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/states/euclidean.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActionModelLQRTpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataLQRTpl<Scalar> Data;
  typedef StateVectorTpl<Scalar> StateVector;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  ActionModelLQRTpl(const std::size_t nx, const std::size_t nu,
                    const bool drift_free = true);
  virtual ~ActionModelLQRTpl();

  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);
  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);
  virtual boost::shared_ptr<ActionDataAbstract> createData();
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract>& data);

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

  /** @brief Return the state weight vector */
  const VectorXs& get_q() const;

  /** @brief Return the input weight vector */
  const VectorXs& get_r() const;

  /** @brief Modify the state matrix */
  void set_A(const MatrixXs& A);

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

  /** @brief Modify the state weigth vector */
  void set_q(const VectorXs& q);

  /** @brief Modify the input weight vector */
  void set_r(const VectorXs& r);

  DEPRECATED("Use get_A", const MatrixXs& get_Fx() const { return get_A(); })
  DEPRECATED("Use get_B", const MatrixXs& get_Fu() const { return get_B(); })
  DEPRECATED("Use get_f", const VectorXs& get_f0() const { return get_f(); })
  DEPRECATED("Use get_q", const VectorXs& get_lx() const { return get_q(); })
  DEPRECATED("Use get_r", const VectorXs& get_lu() const { return get_r(); })
  DEPRECATED("Use get_Q", const MatrixXs& get_Lxx() const { return get_Q(); })
  DEPRECATED("Use get_R", const MatrixXs& get_Lxu() const { return get_R(); })
  DEPRECATED("Use get_N", const MatrixXs& get_Luu() const { return get_N(); })
  DEPRECATED("Use set_A", void set_Fx(const MatrixXs& A) { set_A(A); })
  DEPRECATED("Use set_B", void set_Fu(const MatrixXs& B) { set_B(B); })
  DEPRECATED("Use set_f", void set_f0(const VectorXs& f) { set_f(f); })
  DEPRECATED("Use set_q", void set_lx(const VectorXs& q) { set_q(q); })
  DEPRECATED("Use set_r", void set_lu(const VectorXs& r) { set_r(r); })
  DEPRECATED("Use set_Q", void set_Lxx(const MatrixXs& Q) { set_Q(Q); })
  DEPRECATED("Use set_R", void set_Luu(const MatrixXs& R) { set_R(R); })
  DEPRECATED("Use set_N", void set_Lxu(const MatrixXs& N) { set_N(N); })

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
  MatrixXs A_;
  MatrixXs B_;
  MatrixXs Q_;
  MatrixXs R_;
  MatrixXs N_;
  VectorXs f_;
  VectorXs q_;
  VectorXs r_;
  bool drift_free_;
};

template <typename _Scalar>
struct ActionDataLQRTpl : public ActionDataAbstractTpl<_Scalar> {
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;

  template <template <typename Scalar> class Model>
  explicit ActionDataLQRTpl(Model<Scalar>* const model)
      : Base(model),
        R_u_tmp(VectorXs::Zero(static_cast<Eigen::Index>(model->get_nu()))),
        Q_x_tmp(VectorXs::Zero(
            static_cast<Eigen::Index>(model->get_state()->get_ndx()))) {
    // Setting the linear model and quadratic cost here because they are
    // constant
    Fx = model->get_A();
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
  using Base::xnext;
  VectorXs R_u_tmp;  // Temporary variable for storing Hessian-vector product
                     // (size: nu)
  VectorXs Q_x_tmp;  // Temporary variable for storing Hessian-vector product
                     // (size: nx)
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/actions/lqr.hxx"

#endif  // CROCODDYL_CORE_ACTIONS_LQR_HPP_
