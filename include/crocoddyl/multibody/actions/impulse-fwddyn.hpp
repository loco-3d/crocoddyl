///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_IMPULSE_FWDDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_IMPULSE_FWDDYN_HPP_

#include <stdexcept>
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/multibody/costs/cost-sum.hpp"
#include "crocoddyl/multibody/actions/impulse-fwddyn.hpp"

#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>


namespace crocoddyl {

template<typename _Scalar>
class ActionModelImpulseFwdDynamicsTpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ImpulseModelMultipleTpl<Scalar> ImpulseModelMultiple;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  
  ActionModelImpulseFwdDynamicsTpl(boost::shared_ptr<StateMultibody> state,
                                   boost::shared_ptr<ImpulseModelMultiple> impulses,
                                   boost::shared_ptr<CostModelSum> costs,
                                   const Scalar& r_coeff = 0.,
                                   const Scalar& JMinvJt_damping = 0.,
                                   const bool& enable_force = false);
  ~ActionModelImpulseFwdDynamicsTpl();

  void calc(const boost::shared_ptr<ActionDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u);
  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u);
  boost::shared_ptr<ActionDataAbstract> createData();

  const boost::shared_ptr<ImpulseModelMultiple>& get_impulses() const;
  const boost::shared_ptr<CostModelSum>& get_costs() const;
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;
  const VectorXs& get_armature() const;
  const Scalar& get_restitution_coefficient() const;
  const Scalar& get_damping_factor() const;

  void set_armature(const VectorXs& armature);
  void set_restitution_coefficient(const Scalar& r_coeff);
  void set_damping_factor(const Scalar& damping);

protected:
  using Base::nu_;                          //!< Control dimension
  using Base::nr_;                          //!< Dimension of the cost residual
  using Base::state_;  //!< Model of the state
  using Base::unone_;                   //!< Neutral state
  using Base::u_lb_;                    //!< Lower control limits
  using Base::u_ub_;                    //!< Upper control limits
  using Base::has_control_limits_;      //!< Indicates whether any of the control limits
  
 private:
  boost::shared_ptr<ImpulseModelMultiple> impulses_;
  boost::shared_ptr<CostModelSum> costs_;
  pinocchio::ModelTpl<Scalar>& pinocchio_;
  bool with_armature_;
  VectorXs armature_;
  Scalar r_coeff_;
  Scalar JMinvJt_damping_;
  bool enable_force_;
  pinocchio::MotionTpl<Scalar> gravity_;
};
  
template<typename _Scalar>
struct ActionDataImpulseFwdDynamicsTpl : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  
  template <template<typename Scalar> class Model>
  explicit ActionDataImpulseFwdDynamicsTpl(Model<Scalar>* const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(&pinocchio, model->get_impulses()->createData(&pinocchio)),
        costs(model->get_costs()->createData(&multibody)),
        vnone(model->get_state()->get_nv()),
        Kinv(model->get_state()->get_nv() + model->get_impulses()->get_ni(),
             model->get_state()->get_nv() + model->get_impulses()->get_ni()),
        df_dq(model->get_impulses()->get_ni(), model->get_state()->get_nv()) {
    costs->shareMemory(this);
    vnone.fill(0);
    Kinv.fill(0);
    df_dq.fill(0);
  }

  pinocchio::DataTpl<Scalar> pinocchio;
  DataCollectorMultibodyInImpulseTpl<Scalar> multibody;
  boost::shared_ptr<CostDataSumTpl<Scalar> > costs;
  VectorXs vnone;
  MatrixXs Kinv;
  MatrixXs df_dq;
};

  typedef ActionModelImpulseFwdDynamicsTpl<double> ActionModelImpulseFwdDynamics;
  typedef ActionDataImpulseFwdDynamicsTpl<double> ActionDataImpulseFwdDynamics;
  
}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include <crocoddyl/multibody/actions/impulse-fwddyn.hxx>

#endif  // CROCODDYL_MULTIBODY_ACTIONS_IMPULSE_FWDDYN_HPP_
