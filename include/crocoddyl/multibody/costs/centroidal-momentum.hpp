///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_
#define CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template<typename _Scalar>
class CostModelCentroidalMomentumTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ActivationModelQuadTpl<Scalar> ActivationModelQuad;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Vector6s Vector6s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  
  CostModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state,
                                 boost::shared_ptr<ActivationModelAbstract> activation,
                                 const Vector6s& mref,
                                 const std::size_t& nu);
  CostModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state,
                                 boost::shared_ptr<ActivationModelAbstract> activation,
                                 const Vector6s& mref);
  CostModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state,
                              const Vector6s& mref, const std::size_t& nu);
  CostModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state,
                                 const Vector6s& mref);
  ~CostModelCentroidalMomentumTpl();

  void calc(const boost::shared_ptr<CostDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u);
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  const Vector6s& get_href() const;
  void set_href(const Vector6s& mref_in);

protected:
  using Base::state_;
  using Base::activation_;
  using Base::nu_;
  using Base::with_residuals_;
  using Base::unone_;
  
 private:
  Vector6s href_;
};

template<typename _Scalar>
struct CostDataCentroidalMomentumTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::Vector6s Vector6s;

  template <typename Model>
  CostDataCentroidalMomentumTpl(Model* const model, DataCollectorAbstract* const data)
      : Base(model, data),
        dhd_dq(6, model->get_state()->get_nv()),
        dhd_dv(6, model->get_state()->get_nv()) {
    dhd_dq.fill(0);
    dhd_dv.fill(0);

    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar>* d =
      dynamic_cast<DataCollectorMultibodyTpl<Scalar>* >(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::DataTpl<Scalar>* pinocchio;
  Matrix6xs dhd_dq;
  Matrix6xs dhd_dv;
  Matrix6xs Arr_Rx;
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
#include "crocoddyl/multibody/costs/centroidal-momentum.hxx"


#endif  // CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_
