///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FWD_HPP_
#define CROCODDYL_MULTIBODY_FWD_HPP_

#include "crocoddyl/core/utils/deprecate.hpp"
#include "crocoddyl/core/costs/residual.hpp"

namespace crocoddyl {

// actuation
template <typename Scalar>
class ActuationModelFloatingBaseTpl;

template <typename Scalar>
class ActuationModelFullTpl;

template <typename Scalar>
class ActuationModelMultiCopterBaseTpl;

// force
template <typename Scala>
struct ForceDataAbstractTpl;

// contact
template <typename Scalar>
class ContactModelAbstractTpl;
template <typename Scalar>
struct ContactDataAbstractTpl;

// action
template <typename Scalar>
class ActionModelImpulseFwdDynamicsTpl;
template <typename Scalar>
struct ActionDataImpulseFwdDynamicsTpl;

// differential action
template <typename Scalar>
class DifferentialActionModelFreeFwdDynamicsTpl;
template <typename Scalar>
struct DifferentialActionDataFreeFwdDynamicsTpl;

template <typename Scalar>
class DifferentialActionModelContactFwdDynamicsTpl;
template <typename Scalar>
struct DifferentialActionDataContactFwdDynamicsTpl;

// numdiff
template <typename Scalar>
class CostModelNumDiffTpl;
template <typename Scalar>
struct CostDataNumDiffTpl;

template <typename Scalar>
class ContactModelNumDiffTpl;
template <typename Scalar>
struct ContactDataNumDiffTpl;

// frame
template <typename Scalar>
struct FrameTranslationTpl;

template <typename Scalar>
struct FrameRotationTpl;

template <typename Scalar>
struct FramePlacementTpl;

template <typename Scalar>
struct FrameMotionTpl;

template <typename Scalar>
struct FrameForceTpl;

template <typename Scalar>
struct FrameFrictionConeTpl;

template <typename Scalar>
struct FrameWrenchConeTpl;

template <typename Scalar>
struct FrameCoPSupportTpl;

// residual
template <typename Scalar>
class ResidualModelCentroidalMomentumTpl;
template <typename Scalar>
struct ResidualDataCentroidalMomentumTpl;

template <typename Scalar>
class ResidualModelCoMPositionTpl;
template <typename Scalar>
struct ResidualDataCoMPositionTpl;

template <typename Scalar>
class ResidualModelContactForceTpl;
template <typename Scalar>
struct ResidualDataContactForceTpl;

template <typename Scalar>
class ResidualModelContactFrictionConeTpl;
template <typename Scalar>
struct ResidualDataContactFrictionConeTpl;

template <typename Scalar>
class ResidualModelContactCoPPositionTpl;
template <typename Scalar>
struct ResidualDataContactCoPPositionTpl;

template <typename Scalar>
class ResidualModelContactWrenchConeTpl;
template <typename Scalar>
struct ResidualDataContactWrenchConeTpl;

template <typename Scalar>
class ResidualModelContactControlGravTpl;
template <typename Scalar>
struct ResidualDataContactControlGravTpl;

template <typename Scalar>
class ResidualModelControlGravTpl;
template <typename Scalar>
struct ResidualDataControlGravTpl;

template <typename Scalar>
class ResidualModelFramePlacementTpl;
template <typename Scalar>
struct ResidualDataFramePlacementTpl;

template <typename Scalar>
class ResidualModelFrameRotationTpl;
template <typename Scalar>
struct ResidualDataFrameRotationTpl;

template <typename Scalar>
class ResidualModelFrameTranslationTpl;
template <typename Scalar>
struct ResidualDataFrameTranslationTpl;

template <typename Scalar>
class ResidualModelFrameVelocityTpl;
template <typename Scalar>
struct ResidualDataFrameVelocityTpl;

template <typename Scalar>
class ResidualModelImpulseCoMTpl;
template <typename Scalar>
struct ResidualDataImpulseCoMTpl;

template <typename Scalar>
class ResidualModelStateTpl;
template <typename Scalar>
struct ResidualDataStateTpl;

// cost
template <typename Scalar>
class CostModelFrameTranslationTpl;

template <typename Scalar>
using CostDataFrameTranslationTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelCentroidalMomentumTpl;
template <typename Scalar>
using CostDataCentroidalMomentumTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelCoMPositionTpl;
template <typename Scalar>
using CostDataCoMPositionTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelFramePlacementTpl;
template <typename Scalar>
using CostDataFramePlacementTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelImpulseCoMTpl;
template <typename Scalar>
using CostDataImpulseCoMTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelStateTpl;
template <typename Scalar>
using CostDataStateTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelControlGravTpl;
template <typename Scalar>
using CostDataControlGravTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelControlGravContactTpl;
template <typename Scalar>
using CostDataControlGravContactTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelFrameVelocityTpl;
template <typename Scalar>
using CostDataFrameVelocityTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelContactFrictionConeTpl;
template <typename Scalar>
using CostDataContactFrictionConeTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelContactWrenchConeTpl;
template <typename Scalar>
using CostDataContactWrenchConeTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelContactForceTpl;
template <typename Scalar>
using CostDataContactForceTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelContactImpulseTpl;
template <typename Scalar>
using CostDataContactImpulseTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelFrameRotationTpl;
template <typename Scalar>
using CostDataFrameRotationTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelImpulseFrictionConeTpl;
template <typename Scalar>
using CostDataImpulseFrictionConeTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelImpulseWrenchConeTpl;
template <typename Scalar>
using CostDataImpulseWrenchConeTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelContactCoPPositionTpl;
template <typename Scalar>
using CostDataContactCoPPositionTpl = CostDataResidualTpl<Scalar>;

template <typename Scalar>
class CostModelImpulseCoPPositionTpl;
template <typename Scalar>
using CostDataImpulseCoPPositionTpl = CostDataResidualTpl<Scalar>;

// impulse
template <typename Scalar>
class ImpulseModelAbstractTpl;
template <typename Scalar>
struct ImpulseDataAbstractTpl;

// contact
template <typename Scalar>
struct ContactItemTpl;
template <typename Scalar>
class ContactModelMultipleTpl;
template <typename Scalar>
struct ContactDataMultipleTpl;

template <typename Scalar>
class ContactModel2DTpl;
template <typename Scalar>
struct ContactData2DTpl;

template <typename Scalar>
class ContactModel3DTpl;
template <typename Scalar>
struct ContactData3DTpl;

template <typename Scalar>
class ContactModel6DTpl;
template <typename Scalar>
struct ContactData6DTpl;

// friction
template <typename Scalar>
class FrictionConeTpl;
template <typename Scalar>
class WrenchConeTpl;

// cop support
template <typename Scalar>
class CoPSupportTpl;

// state
template <typename Scalar>
class StateMultibodyTpl;

// data collector
template <typename Scalar>
struct DataCollectorMultibodyTpl;

template <typename Scalar>
struct DataCollectorActMultibodyTpl;

template <typename Scalar>
struct DataCollectorContactTpl;

template <typename Scalar>
struct DataCollectorMultibodyInContactTpl;

template <typename Scalar>
struct DataCollectorActMultibodyInContactTpl;

template <typename Scalar>
struct DataCollectorImpulseTpl;

template <typename Scalar>
struct DataCollectorMultibodyInImpulseTpl;

// impulse
template <typename Scalar>
class ImpulseModel6DTpl;
template <typename Scalar>
struct ImpulseData6DTpl;

template <typename Scalar>
class ImpulseModel3DTpl;
template <typename Scalar>
struct ImpulseData3DTpl;

template <typename Scalar>
struct ImpulseItemTpl;
template <typename Scalar>
class ImpulseModelMultipleTpl;
template <typename Scalar>
struct ImpulseDataMultipleTpl;

/*******************************Template Instantiation**************************/

typedef ActuationModelFloatingBaseTpl<double> ActuationModelFloatingBase;
typedef ActuationModelFullTpl<double> ActuationModelFull;
typedef ActuationModelMultiCopterBaseTpl<double> ActuationModelMultiCopterBase;

typedef ForceDataAbstractTpl<double> ForceDataAbstract;

typedef ContactModelAbstractTpl<double> ContactModelAbstract;
typedef ContactDataAbstractTpl<double> ContactDataAbstract;

typedef ActionModelImpulseFwdDynamicsTpl<double> ActionModelImpulseFwdDynamics;
typedef ActionDataImpulseFwdDynamicsTpl<double> ActionDataImpulseFwdDynamics;

typedef DifferentialActionModelFreeFwdDynamicsTpl<double> DifferentialActionModelFreeFwdDynamics;
typedef DifferentialActionDataFreeFwdDynamicsTpl<double> DifferentialActionDataFreeFwdDynamics;
typedef DifferentialActionModelContactFwdDynamicsTpl<double> DifferentialActionModelContactFwdDynamics;
typedef DifferentialActionDataContactFwdDynamicsTpl<double> DifferentialActionDataContactFwdDynamics;

typedef CostModelNumDiffTpl<double> CostModelNumDiff;
typedef CostDataNumDiffTpl<double> CostDataNumDiff;
typedef ContactModelNumDiffTpl<double> ContactModelNumDiff;
typedef ContactDataNumDiffTpl<double> ContactDataNumDiff;

typedef FrictionConeTpl<double> FrictionCone;
typedef WrenchConeTpl<double> WrenchCone;
typedef CoPSupportTpl<double> CoPSupport;

typedef FrameTranslationTpl<double> FrameTranslation;
typedef FrameRotationTpl<double> FrameRotation;
typedef FramePlacementTpl<double> FramePlacement;
typedef FrameMotionTpl<double> FrameMotion;
typedef FrameForceTpl<double> FrameForce;
typedef FrameFrictionConeTpl<double> FrameFrictionCone;
typedef FrameWrenchConeTpl<double> FrameWrenchCone;
typedef FrameCoPSupportTpl<double> FrameCoPSupport;

typedef ResidualModelCentroidalMomentumTpl<double> ResidualModelCentroidalMomentum;
typedef ResidualDataCentroidalMomentumTpl<double> ResidualDataCentroidalMomentum;
typedef ResidualModelCoMPositionTpl<double> ResidualModelCoMPosition;
typedef ResidualDataCoMPositionTpl<double> ResidualDataCoMPosition;
typedef ResidualModelContactForceTpl<double> ResidualModelContactForce;
typedef ResidualDataContactForceTpl<double> ResidualDataContactForce;
typedef ResidualModelContactFrictionConeTpl<double> ResidualModelContactFrictionCone;
typedef ResidualDataContactFrictionConeTpl<double> ResidualDataContactFrictionCone;
typedef ResidualModelContactCoPPositionTpl<double> ResidualModelContactCoPPosition;
typedef ResidualDataContactCoPPositionTpl<double> ResidualDataContactCoPPosition;
typedef ResidualModelContactWrenchConeTpl<double> ResidualModelContactWrenchCone;
typedef ResidualDataContactWrenchConeTpl<double> ResidualDataContactWrenchCone;
typedef ResidualModelContactControlGravTpl<double> ResidualModelContactControlGrav;
typedef ResidualDataContactControlGravTpl<double> ResidualDataContactControlGrav;
typedef ResidualModelControlGravTpl<double> ResidualModelControlGrav;
typedef ResidualDataControlGravTpl<double> ResidualDataControlGrav;
typedef ResidualModelFramePlacementTpl<double> ResidualModelFramePlacement;
typedef ResidualDataFramePlacementTpl<double> ResidualDataFramePlacement;
typedef ResidualModelFrameRotationTpl<double> ResidualModelFrameRotation;
typedef ResidualDataFrameRotationTpl<double> ResidualDataFrameRotation;
typedef ResidualModelFrameTranslationTpl<double> ResidualModelFrameTranslation;
typedef ResidualDataFrameTranslationTpl<double> ResidualDataFrameTranslation;
typedef ResidualModelFrameVelocityTpl<double> ResidualModelFrameVelocity;
typedef ResidualDataFrameVelocityTpl<double> ResidualDataFrameVelocity;
typedef ResidualModelImpulseCoMTpl<double> ResidualModelImpulseCoM;
typedef ResidualDataImpulseCoMTpl<double> ResidualDataImpulseCoM;
typedef ResidualModelStateTpl<double> ResidualModelState;
typedef ResidualDataStateTpl<double> ResidualDataState;

DEPRECATED("Use ResidualModelFrameTranslation with CostModelResidual",
           typedef CostModelFrameTranslationTpl<double> CostModelFrameTranslation;)
DEPRECATED("Use ResidualDataFrameTranslation", typedef CostDataResidualTpl<double> CostDataFrameTranslation;)
DEPRECATED("Use ResidualModelCentroidalMomentum with CostModelResidual",
           typedef CostModelCentroidalMomentumTpl<double> CostModelCentroidalMomentum;)
DEPRECATED("Use ResidualDataCentroidalMomentum", typedef CostDataResidualTpl<double> CostDataCentroidalMomentum;)
DEPRECATED("Use ResidualModelCoMPosition with CostModelResidual",
           typedef CostModelCoMPositionTpl<double> CostModelCoMPosition;)
DEPRECATED("Use ResidualDataCoMPosition", typedef CostDataResidualTpl<double> CostDataCoMPosition;)
DEPRECATED("Use ResidualModelFramePlacement with CostModelResidual",
           typedef CostModelFramePlacementTpl<double> CostModelFramePlacement;)
DEPRECATED("Use ResidualDataFramePlacement", typedef CostDataResidualTpl<double> CostDataFramePlacement;)
DEPRECATED("Use ResidualModelImpulseCoM with CostModelResidual",
           typedef CostModelImpulseCoMTpl<double> CostModelImpulseCoM;)
DEPRECATED("Use ResidualDataImpulseCoM", typedef CostDataResidualTpl<double> CostDataImpulseCoM;)
DEPRECATED("Use ResidualModelState with CostModelResidual", typedef CostModelStateTpl<double> CostModelState;)
DEPRECATED("Use ResidualDataState", typedef CostDataResidualTpl<double> CostDataState;)
DEPRECATED("Use ResidualModelControlGrav with CostModelResidual",
           typedef CostModelControlGravTpl<double> CostModelControlGrav;)
DEPRECATED("Use ResidualDataControlGrav", typedef CostDataResidualTpl<double> CostDataControlGrav;)
DEPRECATED("Use ResidualModelContactControlGrav with CostModelResidual",
           typedef CostModelControlGravContactTpl<double> CostModelControlGravContact;)
DEPRECATED("Use ResidualDataContactControlGrav", typedef CostDataResidualTpl<double> CostDataControlGravContact;)
DEPRECATED("Use ResidualModelFrameVelocity with CostModelResidual",
           typedef CostModelFrameVelocityTpl<double> CostModelFrameVelocity;)
DEPRECATED("Use ResidualDataFrameVelocity", typedef CostDataResidualTpl<double> CostDataFrameVelocity;)
DEPRECATED("Use ResidualModelFrameRotation with CostModelResidual",
           typedef CostModelFrameRotationTpl<double> CostModelFrameRotation;)
DEPRECATED("Use ResidualDataFrameRotation", typedef CostDataResidualTpl<double> CostDataFrameRotation;)
DEPRECATED("Use ResidualModelCoPPosition with CostModelResidual",
           typedef CostModelContactCoPPositionTpl<double> CostModelContactCoPPosition;)
DEPRECATED("Use ResidualDataCoPPosition", typedef CostDataResidualTpl<double> CostDataContactCoPPosition;)
DEPRECATED("Use ResidualModelContactFrictionCone with CostModelResidual",
           typedef CostModelContactFrictionConeTpl<double> CostModelContactFrictionCone;)
DEPRECATED("Use ResidualDataContactFrictionCone", typedef CostDataResidualTpl<double> CostDataContactFrictionCone;)
DEPRECATED("Use ResidualModelContactWrenchCone with CostModelResidual",
           typedef CostModelContactWrenchConeTpl<double> CostModelContactWrenchCone;)
DEPRECATED("Use ResidualDataContactWrenchCone", typedef CostDataResidualTpl<double> CostDataContactWrenchCone;)
DEPRECATED("Use ResidualModelContactForce with CostModelResidual",
           typedef CostModelContactForceTpl<double> CostModelContactForce;)
DEPRECATED("Use ResidualDataContactForce", typedef CostDataResidualTpl<double> CostDataContactForce;)
DEPRECATED("Use ResidualModelContactForce with CostModelResidual",
           typedef CostModelContactImpulseTpl<double> CostModelContactImpulse;)
DEPRECATED("Use ResidualDataContactForce", typedef CostDataResidualTpl<double> CostDataContactImpulse;)
DEPRECATED("Use ResidualModelFrictionCone with CostModelResidual",
           typedef CostModelImpulseFrictionConeTpl<double> CostModelImpulseFrictionCone;)
DEPRECATED("Use ResidualDataContactFrictionCone", typedef CostDataResidualTpl<double> CostDataImpulseFrictionCone;)
DEPRECATED("Use ResidualModelContactCoPPosition with CostModelResidual",
           typedef CostModelImpulseCoPPositionTpl<double> CostModelImpulseCoPPosition;)
DEPRECATED("Use ResidualDataContactCoPPosition", typedef CostDataResidualTpl<double> CostDataImpulseCoPPosition;)
DEPRECATED("Use ResidualModelContactWrenchCone with CostModelResidual",
           typedef CostModelImpulseWrenchConeTpl<double> CostModelImpulseWrenchCone;)
DEPRECATED("Use ResidualDataContactWrenchCone", typedef CostDataResidualTpl<double> CostDataImpulseWrenchCone;)

typedef ImpulseModelAbstractTpl<double> ImpulseModelAbstract;
typedef ImpulseDataAbstractTpl<double> ImpulseDataAbstract;

enum ContactType { Contact2D, Contact3D, Contact6D, ContactUndefined };
enum ImpulseType { Impulse3D, Impulse6D, ImpulseUndefined };

typedef ContactItemTpl<double> ContactItem;
typedef ContactModelMultipleTpl<double> ContactModelMultiple;
typedef ContactDataMultipleTpl<double> ContactDataMultiple;
typedef ContactModel2DTpl<double> ContactModel2D;
typedef ContactData2DTpl<double> ContactData2D;
typedef ContactModel3DTpl<double> ContactModel3D;
typedef ContactData3DTpl<double> ContactData3D;
typedef ContactModel6DTpl<double> ContactModel6D;
typedef ContactData6DTpl<double> ContactData6D;

typedef StateMultibodyTpl<double> StateMultibody;

typedef DataCollectorMultibodyTpl<double> DataCollectorMultibody;
typedef DataCollectorActMultibodyTpl<double> DataCollectorActMultibody;
typedef DataCollectorContactTpl<double> DataCollectorContact;
typedef DataCollectorMultibodyInContactTpl<double> DataCollectorMultibodyInContact;
typedef DataCollectorActMultibodyInContactTpl<double> DataCollectorActMultibodyInContact;
typedef DataCollectorImpulseTpl<double> DataCollectorImpulse;
typedef DataCollectorMultibodyInImpulseTpl<double> DataCollectorMultibodyInImpulse;

typedef ImpulseModel6DTpl<double> ImpulseModel6D;
typedef ImpulseData6DTpl<double> ImpulseData6D;
typedef ImpulseModel3DTpl<double> ImpulseModel3D;
typedef ImpulseData3DTpl<double> ImpulseData3D;
typedef ImpulseItemTpl<double> ImpulseItem;
typedef ImpulseModelMultipleTpl<double> ImpulseModelMultiple;
typedef ImpulseDataMultipleTpl<double> ImpulseDataMultiple;

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FWD_HPP_
