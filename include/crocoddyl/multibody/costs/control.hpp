///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTROL_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTROL_HPP_

#include "crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {

template<typename _Scalar>
class CostModelControlTpl : public CostModelAbstractTpl<_Scalar> {
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
  
  CostModelControlTpl(boost::shared_ptr<StateMultibody> state,
                      boost::shared_ptr<ActivationModelAbstract> activation,
                      const VectorXs& uref);
  CostModelControlTpl(boost::shared_ptr<StateMultibody> state,
                      boost::shared_ptr<ActivationModelAbstract> activation);
  CostModelControlTpl(boost::shared_ptr<StateMultibody> state,
                      boost::shared_ptr<ActivationModelAbstract> activation,
                      const std::size_t& nu);
  CostModelControlTpl(boost::shared_ptr<StateMultibody> state, const VectorXs& uref);
  explicit CostModelControlTpl(boost::shared_ptr<StateMultibody> state);
  CostModelControlTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& nu);
  ~CostModelControlTpl();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u);

  const VectorXs& get_uref() const;
  void set_uref(const VectorXs& uref_in);

 protected:
  using Base::state_;
  using Base::activation_;
  using Base::nu_;
  using Base::with_residuals_;
  using Base::unone_;
  
 private:
  VectorXs uref_;
};

  typedef CostModelControlTpl<double> CostModelControl;
  
}  // namespace crocoddyl


/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/control.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTROL_HPP_
