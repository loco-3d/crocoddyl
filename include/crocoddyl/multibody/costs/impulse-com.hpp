///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_IMPULSE_COM_HPP_
#define CROCODDYL_MULTIBODY_COSTS_IMPULSE_COM_HPP_
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template<typename _Scalar>
class CostModelImpulseCoMTpl : public CostModelAbstractTpl<_Scalar> {
 public:
    typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ActivationModelQuadTpl<Scalar> ActivationModelQuad;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  
  CostModelImpulseCoMTpl(boost::shared_ptr<StateMultibody> state,
                         boost::shared_ptr<ActivationModelAbstract> activation);
  CostModelImpulseCoMTpl(boost::shared_ptr<StateMultibody> state);
  ~CostModelImpulseCoMTpl();

  void calc(const boost::shared_ptr<CostDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u);
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

protected:
  using Base::state_;
  using Base::activation_;
  using Base::nu_;
  using Base::with_residuals_;
  using Base::unone_;

};

template<typename _Scalar>
struct CostDataImpulseCoMTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix3xs Matrix3xs;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::Vector6s Vector6s;
  
  template <template<typename Scalar> class Model>
  CostDataImpulseCoMTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data),
        Arr_Rx(3, model->get_state()->get_nv()),
        dvc_dq(3, model->get_state()->get_nv()),
        ddv_dv(model->get_state()->get_nv(), model->get_state()->get_nv()),
        pinocchio_internal(pinocchio::DataTpl<Scalar>(model->get_state()->get_pinocchio())) {
    Arr_Rx.fill(0);
    dvc_dq.fill(0);
    ddv_dv.fill(0);
    // Check that proper shared data has been passed
    DataCollectorMultibodyInImpulseTpl<Scalar>* d =
      dynamic_cast<DataCollectorMultibodyInImpulseTpl<Scalar>* >(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibodyInImpulse");
    }
    pinocchio = d->pinocchio;
    impulses = d->impulses;
  }

  pinocchio::DataTpl<Scalar>* pinocchio;
  boost::shared_ptr<crocoddyl::ImpulseDataMultipleTpl<Scalar> > impulses;
  Matrix3xs Arr_Rx;
  Matrix3xs dvc_dq;
  MatrixXs ddv_dv;
  pinocchio::DataTpl<Scalar> pinocchio_internal;
  using Base::shared;
  using Base::activation;
  using Base::cost;
  using Base::Lx;
  using Base::Lu;
  using Base::Lxx;
  using Base::Lxu;
  using Base::Luu;
  using Base::r;
  using Base::Rx;
  using Base::Ru;
  
};

  
}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/impulse-com.hxx"


#endif  // CROCODDYL_MULTIBODY_COSTS_IMPULSE_COM_HPP_
