///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_
#define CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_

#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template<typename _Scalar>
class CostModelFramePlacementTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ActivationModelQuadTpl<Scalar> ActivationModelQuad;
  typedef FramePlacementTpl<Scalar> FramePlacement;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  
  CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                             boost::shared_ptr<ActivationModelAbstract> activation,
                             const FramePlacement& Fref,
                             const std::size_t& nu);
  CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                             boost::shared_ptr<ActivationModelAbstract> activation,
                             const FramePlacement& Fref);
  CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                             const FramePlacement& Fref, const std::size_t& nu);
  CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                             const FramePlacement& Fref);
  ~CostModelFramePlacementTpl();

  void calc(const boost::shared_ptr<CostDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u);
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  const FramePlacement& get_Mref() const;
  void set_Mref(const FramePlacement& Mref_in);

protected:
  using Base::state_;
  using Base::activation_;
  using Base::nu_;
  using Base::with_residuals_;
  using Base::unone_;
 
 private:
  FramePlacement Mref_;
  pinocchio::SE3Tpl<Scalar> oMf_inv_;
};

template<typename _Scalar> 
struct CostDataFramePlacementTpl : public CostDataAbstractTpl<_Scalar> {
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
  CostDataFramePlacementTpl(Model* const model, DataCollectorAbstract* const data)
      : CostDataAbstract(model, data),
        J(6, model->get_state()->get_nv()),
        rJf(6, 6),
        fJf(6, model->get_state()->get_nv()),
        Arr_J(6, model->get_state()->get_nv()) {
    r.fill(0);
    J.fill(0);
    rJf.fill(0);
    fJf.fill(0);
    Arr_J.fill(0);
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
  Vector6s r;
  pinocchio::SE3Tpl<Scalar> rMf;
  Matrix6xs J;
  Matrix6s rJf;
  Matrix6xs fJf;
  Matrix6xs Arr_J;

  using Base::shared;
  using Base::activation;
  using Base::cost;
  using Base::Lx;
  using Base::Lu;
  using Base::Lxx;
  using Base::Lxu;
  using Base::Luu;
  //using Base::r;
  using Base::Rx;
  using Base::Ru;
};

typedef CostModelFramePlacementTpl<double> CostModelFramePlacement;
typedef CostDataFramePlacementTpl<double> CostDataFramePlacement;
  
}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/frame-placement.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_
