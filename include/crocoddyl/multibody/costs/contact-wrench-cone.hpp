///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTACT_WRENCH_CONE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTACT_WRENCH_CONE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename _Scalar>
class CostModelContactWrenchConeTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataContactWrenchConeTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ResidualModelContactWrenchConeTpl<Scalar> ResidualModelContactWrenchCone;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef FrameWrenchConeTpl<Scalar> FrameWrenchCone;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixX6s MatrixX6s;

  CostModelContactWrenchConeTpl(boost::shared_ptr<StateMultibody> state,
                                boost::shared_ptr<ActivationModelAbstract> activation, const FrameWrenchCone& fref,
                                const std::size_t nu);
  CostModelContactWrenchConeTpl(boost::shared_ptr<StateMultibody> state,
                                boost::shared_ptr<ActivationModelAbstract> activation, const FrameWrenchCone& fref);
  CostModelContactWrenchConeTpl(boost::shared_ptr<StateMultibody> state, const FrameWrenchCone& fref,
                                const std::size_t nu);
  CostModelContactWrenchConeTpl(boost::shared_ptr<StateMultibody> state, const FrameWrenchCone& fref);
  virtual ~CostModelContactWrenchConeTpl();

  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

 protected:
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);
  virtual void get_referenceImpl(const std::type_info& ti, void* pv);

  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;

 private:
  FrameWrenchCone fref_;
};

template <typename _Scalar>
struct CostDataContactWrenchConeTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  CostDataContactWrenchConeTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data),
        Arr_Rx(model->get_residual()->get_nr(), model->get_state()->get_ndx()),
        Arr_Ru(model->get_residual()->get_nr(), model->get_nu()) {
    Arr_Rx.setZero();
    Arr_Ru.setZero();
  }

  MatrixXs Arr_Rx;
  MatrixXs Arr_Ru;
  using Base::activation;
  using Base::cost;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::residual;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/contact-wrench-cone.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTACT_WRENCH_CONE_HPP_
