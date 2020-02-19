
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_
#define CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template<typename _Scalar>
class CostModelFrameTranslationTpl : public CostModelAbstractTpl<_Scalar> {
 public:

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ActivationModelQuadTpl<Scalar> ActivationModelQuad;
  typedef FrameTranslationTpl<Scalar> FrameTranslation;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  
  CostModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state,
                               boost::shared_ptr<ActivationModelAbstract> activation,
                               const FrameTranslation& xref,
                               const std::size_t& nu);
  CostModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state,
                               boost::shared_ptr<ActivationModelAbstract> activation,
                               const FrameTranslation& xref);
  CostModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state,
                               const FrameTranslation& xref,
                               const std::size_t& nu);
  CostModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state,
                               const FrameTranslation& xref);
  ~CostModelFrameTranslationTpl();

  void calc(const boost::shared_ptr<CostDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u);
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  const FrameTranslation& get_xref() const;
  void set_xref(const FrameTranslation& xref_in);

protected:
  using Base::state_;
  using Base::activation_;
  using Base::nu_;
  using Base::with_residuals_;
  using Base::unone_;
  
private:
  FrameTranslation xref_;
};

template<typename _Scalar> 
struct CostDataFrameTranslationTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix3xs Matrix3xs;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  
  template <typename Model>
  CostDataFrameTranslationTpl(Model* const model, DataCollectorAbstract* const data)
      : Base(model, data), J(3, model->get_state()->get_nv()), fJf(6, model->get_state()->get_nv()) {
    J.fill(0);
    fJf.fill(0);
    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar>* d = dynamic_cast<DataCollectorMultibodyTpl<Scalar>* >(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::DataTpl<Scalar>* pinocchio;
  Matrix3xs J;
  Matrix6xs fJf;

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
#include "crocoddyl/multibody/costs/frame-translation.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_
