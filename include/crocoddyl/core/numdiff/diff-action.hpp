///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University, Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_DIFF_ACTION_HPP_
#define CROCODDYL_CORE_NUMDIFF_DIFF_ACTION_HPP_

#include <vector>
#include <iostream>

#include "crocoddyl/core/diff-action-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class DifferentialActionModelNumDiffTpl : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef DifferentialActionDataNumDiffTpl<Scalar> DifferentialActionDataNumDiff;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit DifferentialActionModelNumDiffTpl(boost::shared_ptr<Base> model, bool with_gauss_approx = false);
  ~DifferentialActionModelNumDiffTpl();

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u);
  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u);
  boost::shared_ptr<DifferentialActionDataAbstract> createData();

  const boost::shared_ptr<Base>& get_model() const;
  const Scalar& get_disturbance() const;
  void set_disturbance(const Scalar& disturbance);
  bool get_with_gauss_approx();

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control limits
  using Base::nr_;                  //!< Dimension of the cost residual
  using Base::nu_;                  //!< Control dimension
  using Base::state_;               //!< Model of the state
  using Base::u_lb_;                //!< Lower control limits
  using Base::u_ub_;                //!< Upper control limits
  using Base::unone_;               //!< Neutral state

 private:
  void assertStableStateFD(const Eigen::Ref<const VectorXs>& x);
  boost::shared_ptr<Base> model_;
  bool with_gauss_approx_;
  Scalar disturbance_;
};

template <typename _Scalar>
struct DifferentialActionDataNumDiffTpl : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Construct a new ActionDataNumDiff object
   *
   * @tparam Model is the type of the ActionModel.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataNumDiffTpl(Model<Scalar>* const model)
      : Base(model),
        Rx(model->get_model()->get_nr(), model->get_model()->get_state()->get_ndx()),
        Ru(model->get_model()->get_nr(), model->get_model()->get_nu()),
        dx(model->get_model()->get_state()->get_ndx()),
        du(model->get_model()->get_nu()),
        xp(model->get_model()->get_state()->get_nx()) {
    Rx.setZero();
    Ru.setZero();
    dx.setZero();
    du.setZero();
    xp.setZero();

    const std::size_t& ndx = model->get_model()->get_state()->get_ndx();
    const std::size_t& nu = model->get_model()->get_nu();
    data_0 = model->get_model()->createData();
    for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(model->get_model()->createData());
    }
    for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(model->get_model()->createData());
    }
  }

  MatrixXs Rx;
  MatrixXs Ru;
  VectorXs dx;
  VectorXs du;
  VectorXs xp;
  boost::shared_ptr<Base> data_0;
  std::vector<boost::shared_ptr<Base> > data_x;
  std::vector<boost::shared_ptr<Base> > data_u;

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
#include "crocoddyl/core/numdiff/diff-action.hxx"

#endif  // CROCODDYL_CORE_NUMDIFF_DIFF_ACTION_HPP_
