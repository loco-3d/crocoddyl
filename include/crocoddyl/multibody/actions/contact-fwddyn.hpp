///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_

#include <stdexcept>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/costs/cost-sum.hpp"

namespace crocoddyl {

template <typename _Scalar>
class DifferentialActionModelContactFwdDynamicsTpl : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef ActuationModelFloatingBaseTpl<Scalar> ActuationModelFloatingBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  DifferentialActionModelContactFwdDynamicsTpl(boost::shared_ptr<StateMultibody> state,
                                               boost::shared_ptr<ActuationModelFloatingBase> actuation,
                                               boost::shared_ptr<ContactModelMultiple> contacts,
                                               boost::shared_ptr<CostModelSum> costs,
                                               const Scalar& JMinvJt_damping = 0., const bool& enable_force = false);
  ~DifferentialActionModelContactFwdDynamicsTpl();

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u);
  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u);
  boost::shared_ptr<DifferentialActionDataAbstract> createData();

  const boost::shared_ptr<ActuationModelFloatingBase>& get_actuation() const;
  const boost::shared_ptr<ContactModelMultiple>& get_contacts() const;
  const boost::shared_ptr<CostModelSum>& get_costs() const;
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;
  const VectorXs& get_armature() const;
  const Scalar& get_damping_factor() const;

  void set_armature(const VectorXs& armature);
  void set_damping_factor(const Scalar& damping);

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control limits
  using Base::nr_;                  //!< Dimension of the cost residual
  using Base::nu_;                  //!< Control dimension
  using Base::state_;               //!< Model of the state
  using Base::u_lb_;                //!< Lower control limits
  using Base::u_ub_;                //!< Upper control limits
  using Base::unone_;               //!< Neutral state

 private:
  boost::shared_ptr<ActuationModelFloatingBase> actuation_;
  boost::shared_ptr<ContactModelMultiple> contacts_;
  boost::shared_ptr<CostModelSum> costs_;
  pinocchio::ModelTpl<Scalar>& pinocchio_;
  bool with_armature_;
  VectorXs armature_;
  Scalar JMinvJt_damping_;
  bool enable_force_;
};

template <typename _Scalar>
struct DifferentialActionDataContactFwdDynamicsTpl : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataContactFwdDynamicsTpl(Model<Scalar>* const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(&pinocchio, model->get_actuation()->createData(), model->get_contacts()->createData(&pinocchio)),
        costs(model->get_costs()->createData(&multibody)),
        Kinv(model->get_state()->get_nv() + model->get_contacts()->get_nc(),
             model->get_state()->get_nv() + model->get_contacts()->get_nc()),
        df_dx(model->get_contacts()->get_nc(), model->get_state()->get_ndx()),
        df_du(model->get_contacts()->get_nc(), model->get_nu()) {
    costs->shareMemory(this);
    Kinv.fill(0);
    df_dx.fill(0);
    df_du.fill(0);
  }

  pinocchio::DataTpl<Scalar> pinocchio;
  DataCollectorActMultibodyInContactTpl<Scalar> multibody;
  boost::shared_ptr<CostDataSumTpl<Scalar> > costs;
  MatrixXs Kinv;
  MatrixXs df_dx;
  MatrixXs df_du;

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
#include <crocoddyl/multibody/actions/contact-fwddyn.hxx>

#endif  // CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_
