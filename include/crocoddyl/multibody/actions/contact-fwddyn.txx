///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2025, ???
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


namespace crocoddyl {

CROCODDYL_EXPLICIT_INSTANTIATION_EXTERN
template CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
DifferentialActionModelContactFwdDynamicsTpl<CROCODDYL_EXPLICIT_INSTANTIATION_SCALAR>::
    DifferentialActionModelContactFwdDynamicsTpl(
        std::shared_ptr<StateMultibody> state,
        std::shared_ptr<ActuationModelAbstract> actuation,
        std::shared_ptr<ContactModelMultiple> contacts,
        std::shared_ptr<CostModelSum> costs,
        const CROCODDYL_EXPLICIT_INSTANTIATION_SCALAR JMinvJt_damping,
        const bool enable_force);

}  // namespace crocoddyl
