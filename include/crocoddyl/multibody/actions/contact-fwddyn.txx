///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2025, ???
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


namespace crocoddyl {

#define CROCODDYL_ETI_CLASS DifferentialActionModelContactFwdDynamicsTpl<CROCODDYL_EXPLICIT_INSTANTIATION_SCALAR>
#define CROCODDYL_ETI_OUT(type) \
  CROCODDYL_EXPLICIT_INSTANTIATION_EXTERN \
  template CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
  type \
  CROCODDYL_ETI_CLASS:: \


CROCODDYL_ETI_OUT()
    DifferentialActionModelContactFwdDynamicsTpl(
        std::shared_ptr<StateMultibody> state,
        std::shared_ptr<ActuationModelAbstract> actuation,
        std::shared_ptr<ContactModelMultiple> contacts,
        std::shared_ptr<CostModelSum> costs,
        const Scalar JMinvJt_damping,
        const bool enable_force);

CROCODDYL_ETI_OUT()
    DifferentialActionModelContactFwdDynamicsTpl(
      std::shared_ptr<StateMultibody> state,
      std::shared_ptr<ActuationModelAbstract> actuation,
      std::shared_ptr<ContactModelMultiple> contacts,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints,
      const Scalar JMinvJt_damping,
      const bool enable_force);


CROCODDYL_ETI_OUT(void)
    calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
         const Eigen::Ref<const VectorXs>& x);


CROCODDYL_ETI_OUT(void)
    calcDiff(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u);


CROCODDYL_ETI_OUT(void)
    calcDiff(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x);


CROCODDYL_ETI_OUT(std::shared_ptr<CROCODDYL_ETI_CLASS::DifferentialActionDataAbstract>)
    createData();


CROCODDYL_ETI_OUT(bool)
    checkData(
const std::shared_ptr<DifferentialActionDataAbstract>& data);


CROCODDYL_ETI_OUT(void)
    quasiStatic(
        const std::shared_ptr<DifferentialActionDataAbstract>& data,
        Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
        const std::size_t maxiter,
        const Scalar tol);


CROCODDYL_ETI_OUT(std::size_t)
    get_ng() const;



CROCODDYL_ETI_OUT(std::size_t)
    get_nh() const;


CROCODDYL_ETI_OUT(std::size_t)
    get_ng_T() const;


CROCODDYL_ETI_OUT(std::size_t)
    get_nh_T() const;


CROCODDYL_ETI_OUT(const CROCODDYL_ETI_CLASS::VectorXs&)
  get_g_lb() const;


CROCODDYL_ETI_OUT(const CROCODDYL_ETI_CLASS::VectorXs&)
  get_g_ub() const;


CROCODDYL_ETI_OUT(const std::shared_ptr<CROCODDYL_ETI_CLASS::ActuationModelAbstract>&)
    get_actuation() const;


CROCODDYL_ETI_OUT(const std::shared_ptr<CROCODDYL_ETI_CLASS::ContactModelMultiple>&)
    get_contacts() const;


CROCODDYL_ETI_OUT(const std::shared_ptr<CROCODDYL_ETI_CLASS::CostModelSum>&)
    get_costs() const;


CROCODDYL_ETI_OUT(const std::shared_ptr<CROCODDYL_ETI_CLASS::ConstraintModelManager>&)
 get_constraints() const;


CROCODDYL_ETI_OUT(pinocchio::ModelTpl<CROCODDYL_ETI_CLASS::Scalar>&)
 get_pinocchio() const;


CROCODDYL_ETI_OUT(const CROCODDYL_ETI_CLASS::VectorXs&)
 get_armature() const;


CROCODDYL_ETI_OUT(const CROCODDYL_ETI_CLASS::Scalar)
 get_damping_factor() const;


CROCODDYL_ETI_OUT(void)
  set_armature(const VectorXs& armature);


CROCODDYL_ETI_OUT(void)
  set_damping_factor(const Scalar damping);

CROCODDYL_ETI_OUT(void)
  print(std::ostream& os) const;

CROCODDYL_ETI_OUT(void)
  init();

#undef CROCODDYL_ETI_CLASS
#undef CROCODDYL_ETI_OUT

}  // namespace crocoddyl
