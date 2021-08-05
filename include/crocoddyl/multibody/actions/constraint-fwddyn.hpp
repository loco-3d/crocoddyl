///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN2_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN2_HPP_

#if PINOCCHIO_VERSION_AT_LEAST(2, 9, 0)

#include <stdexcept>

#include <pinocchio/algorithm/contact-info.hpp>
#include <pinocchio/algorithm/proximal.hpp>
#include <pinocchio/algorithm/constrained-dynamics.hpp>
#include <pinocchio/algorithm/constrained-dynamics-derivatives.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/data/constraint.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"

namespace crocoddyl {

template <typename _Scalar>
class DifferentialActionModelConstraintFwdDynamicsTpl : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef DifferentialActionDataConstraintFwdDynamicsTpl<Scalar> DifferentialActionDataConstraintFwdDynamics;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef pinocchio::RigidConstraintModelTpl<Scalar, 0> RigidConstraintModel;
  typedef pinocchio::ProximalSettingsTpl<Scalar> ProximalSettings;

  DifferentialActionModelConstraintFwdDynamicsTpl(
      boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelAbstract> actuation,
      const PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintModel) & contacts,
      boost::shared_ptr<CostModelSum> costs, const ProximalSettings& settings);
  ~DifferentialActionModelConstraintFwdDynamicsTpl();

  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u);
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();
  // virtual void quasiStatic(const boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
  //                         const Eigen::Ref<const VectorXs>& x, const std::size_t& maxiter = 100, const Scalar& tol =
  //                         1e-9);

  const boost::shared_ptr<ActuationModelAbstract>& get_actuation() const;
  const PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintModel) & get_contacts() const;
  const boost::shared_ptr<CostModelSum>& get_costs() const;
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;
  const VectorXs& get_armature() const;
  void set_armature(const VectorXs& armature);

  virtual bool checkData(const boost::shared_ptr<DifferentialActionDataAbstract>& data);
  
  const ProximalSettings& get_settings() const;
  void set_settings(const ProximalSettings& settings);

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control limits
  using Base::nr_;                  //!< Dimension of the cost residual
  using Base::nu_;                  //!< Control dimension
  using Base::state_;               //!< Model of the state
  using Base::u_lb_;                //!< Lower control limits
  using Base::u_ub_;                //!< Upper control limits
  using Base::unone_;               //!< Neutral state

 private:
  boost::shared_ptr<ActuationModelAbstract> actuation_;
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintModel) contacts_;
  boost::shared_ptr<CostModelSum> costs_;
  pinocchio::ModelTpl<Scalar>& pinocchio_;
  ProximalSettings settings_;
};

template <typename _Scalar>
struct DifferentialActionDataConstraintFwdDynamicsTpl : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef pinocchio::RigidConstraintDataTpl<Scalar, 0> RigidConstraintData;
  typedef pinocchio::RigidConstraintModelTpl<Scalar, 0> RigidConstraintModel;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataConstraintFwdDynamicsTpl(Model<Scalar>* const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(&pinocchio, model->get_actuation()->createData(),
                  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintData)()),
        costs(model->get_costs()->createData(&multibody)) {
    costs->shareMemory(this);

    for (unsigned int i = 0; i < model->get_contacts().size(); i++) {
      multibody.contacts.push_back(RigidConstraintData(model->get_contacts()[i]));
    }
  }

  pinocchio::DataTpl<Scalar> pinocchio;
  DataCollectorActMultibodyInConstraintTpl<Scalar> multibody;
  boost::shared_ptr<CostDataSumTpl<Scalar> > costs;

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
#include <crocoddyl/multibody/actions/constraint-fwddyn.hxx>

#endif  // PINOCCHIO_VERSION_AT_LEAST(2,9,0)

#endif  // CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_
