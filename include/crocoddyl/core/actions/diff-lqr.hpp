///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIONS_DIFF_LQR_HPP_
#define CROCODDYL_CORE_ACTIONS_DIFF_LQR_HPP_

#include <stdexcept>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/states/euclidean.hpp"

namespace crocoddyl {

template <typename _Scalar>
class DifferentialActionModelLQRTpl : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataLQRTpl<Scalar> Data;
  typedef StateVectorTpl<Scalar> StateVector;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  DifferentialActionModelLQRTpl(const std::size_t nq, const std::size_t nu, const bool drift_free = true);
  virtual ~DifferentialActionModelLQRTpl();

  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u);
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();
  virtual bool checkData(const boost::shared_ptr<DifferentialActionDataAbstract>& data);

  const MatrixXs& get_Fq() const;
  const MatrixXs& get_Fv() const;
  const MatrixXs& get_Fu() const;
  const VectorXs& get_f0() const;
  const VectorXs& get_lx() const;
  const VectorXs& get_lu() const;
  const MatrixXs& get_Lxx() const;
  const MatrixXs& get_Lxu() const;
  const MatrixXs& get_Luu() const;

  void set_Fq(const MatrixXs& Fq);
  void set_Fv(const MatrixXs& Fv);
  void set_Fu(const MatrixXs& Fu);
  void set_f0(const VectorXs& f0);
  void set_lx(const VectorXs& lx);
  void set_lu(const VectorXs& lu);
  void set_Lxx(const MatrixXs& Lxx);
  void set_Lxu(const MatrixXs& Lxu);
  void set_Luu(const MatrixXs& Luu);

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
  MatrixXs Fq_;
  MatrixXs Fv_;
  MatrixXs Fu_;
  VectorXs f0_;
  MatrixXs Lxx_;
  MatrixXs Lxu_;
  MatrixXs Luu_;
  VectorXs lx_;
  VectorXs lu_;
};

template <typename _Scalar>
struct DifferentialActionDataLQRTpl : public DifferentialActionDataAbstractTpl<_Scalar> {
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataLQRTpl(Model<Scalar>* const model) : Base(model) {
    // Setting the linear model and quadratic cost here because they are constant
    Fx.leftCols(model->get_state()->get_nq()) = model->get_Fq();
    Fx.rightCols(model->get_state()->get_nv()) = model->get_Fv();
    Fu = model->get_Fu();
    Lxx = model->get_Lxx();
    Luu = model->get_Luu();
    Lxu = model->get_Lxu();
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
