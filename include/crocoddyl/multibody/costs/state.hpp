///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_STATE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_STATE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class CostModelStateTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ActivationModelQuadTpl<Scalar> ActivationModelQuad;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  CostModelStateTpl(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
                    const VectorXs& xref, const std::size_t& nu);
  CostModelStateTpl(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
                    const VectorXs& xref);
  CostModelStateTpl(boost::shared_ptr<StateMultibody> state, const VectorXs& xref, const std::size_t& nu);
  CostModelStateTpl(boost::shared_ptr<StateMultibody> state, const VectorXs& xref);
  CostModelStateTpl(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
                    const std::size_t& nu);
  CostModelStateTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& nu);
  CostModelStateTpl(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation);
  explicit CostModelStateTpl(boost::shared_ptr<StateMultibody> state);

  ~CostModelStateTpl();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u);
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  const VectorXs& get_xref() const;
  void set_xref(const VectorXs& xref_in);

 protected:
  using Base::activation_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;
  using Base::with_residuals_;

 private:
  VectorXs xref_;
};

template <typename _Scalar>
struct CostDataStateTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  CostDataStateTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), Arr_Rx(model->get_activation()->get_nr(), model->get_state()->get_ndx()) {
    Arr_Rx.fill(0);
  }

  MatrixXs Arr_Rx;

  using Base::activation;
  using Base::cost;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/state.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_STATE_HPP_
