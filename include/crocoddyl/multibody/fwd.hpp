///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FWD_HPP_
#define CROCODDYL_MULTIBODY_FWD_HPP_

#include "crocoddyl/core/utils/deprecate.hpp"

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
class ForceDataAbstractTpl;

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
struct CostDataFrameTranslationTpl;

template <typename Scalar>
class CostModelCentroidalMomentumTpl;
template <typename Scalar>
struct CostDataCentroidalMomentumTpl;

template <typename Scalar>
class CostModelCoMPositionTpl;
template <typename Scalar>
struct CostDataCoMPositionTpl;

template <typename Scalar>
class CostModelFramePlacementTpl;
template <typename Scalar>
struct CostDataFramePlacementTpl;

template <typename Scalar>
class CostModelImpulseCoMTpl;
template <typename Scalar>
struct CostDataImpulseCoMTpl;

template <typename Scalar>
class CostModelStateTpl;
template <typename Scalar>
struct CostDataStateTpl;

template <typename Scalar>
class CostModelControlGravTpl;
template <typename Scalar>
struct CostDataControlGravTpl;

template <typename Scalar>
class CostModelControlGravContactTpl;
template <typename Scalar>
struct CostDataControlGravContactTpl;

template <typename Scalar>
class CostModelFrameVelocityTpl;
template <typename Scalar>
struct CostDataFrameVelocityTpl;

template <typename Scalar>
class CostModelContactFrictionConeTpl;
template <typename Scalar>
struct CostDataContactFrictionConeTpl;

template <typename Scalar>
class CostModelContactWrenchConeTpl;
template <typename Scalar>
struct CostDataContactWrenchConeTpl;

template <typename Scalar>
class CostModelContactForceTpl;
template <typename Scalar>
struct CostDataContactForceTpl;

template <typename Scalar>
class CostModelContactImpulseTpl;
template <typename Scalar>
struct CostDataContactImpulseTpl;

template <typename Scalar>
class CostModelFrameRotationTpl;
template <typename Scalar>
struct CostDataFrameRotationTpl;

template <typename Scalar>
class CostModelImpulseFrictionConeTpl;
template <typename Scalar>
struct CostDataImpulseFrictionConeTpl;

template <typename Scalar>
class CostModelImpulseWrenchConeTpl;
template <typename Scalar>
struct CostDataImpulseWrenchConeTpl;

template <typename Scalar>
class CostModelContactCoPPositionTpl;
template <typename Scalar>
struct CostDataContactCoPPositionTpl;

template <typename Scalar>
class CostModelImpulseCoPPositionTpl;
template <typename Scalar>
struct CostDataImpulseCoPPositionTpl;

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

typedef CostModelFrameTranslationTpl<double> CostModelFrameTranslation;
typedef CostDataFrameTranslationTpl<double> CostDataFrameTranslation;
typedef CostModelCentroidalMomentumTpl<double> CostModelCentroidalMomentum;
typedef CostDataCentroidalMomentumTpl<double> CostDataCentroidalMomentum;
typedef CostModelCoMPositionTpl<double> CostModelCoMPosition;
typedef CostDataCoMPositionTpl<double> CostDataCoMPosition;
typedef CostModelFramePlacementTpl<double> CostModelFramePlacement;
typedef CostDataFramePlacementTpl<double> CostDataFramePlacement;
typedef CostModelImpulseCoMTpl<double> CostModelImpulseCoM;
typedef CostDataImpulseCoMTpl<double> CostDataImpulseCoM;
typedef CostModelStateTpl<double> CostModelState;
typedef CostDataStateTpl<double> CostDataState;
typedef CostModelControlGravTpl<double> CostModelControlGrav;
typedef CostDataControlGravTpl<double> CostDataControlGrav;
typedef CostModelControlGravContactTpl<double> CostModelControlGravContact;
typedef CostDataControlGravContactTpl<double> CostDataControlGravContact;
typedef CostModelFrameVelocityTpl<double> CostModelFrameVelocity;
typedef CostDataFrameVelocityTpl<double> CostDataFrameVelocity;
typedef CostModelFrameRotationTpl<double> CostModelFrameRotation;
typedef CostDataFrameRotationTpl<double> CostDataFrameRotation;
typedef CostModelContactCoPPositionTpl<double> CostModelContactCoPPosition;
typedef CostDataContactCoPPositionTpl<double> CostDataContactCoPPosition;
typedef CostModelContactFrictionConeTpl<double> CostModelContactFrictionCone;
typedef CostDataContactFrictionConeTpl<double> CostDataContactFrictionCone;
typedef CostModelContactWrenchConeTpl<double> CostModelContactWrenchCone;
typedef CostDataContactWrenchConeTpl<double> CostDataContactWrenchCone;
typedef CostModelContactForceTpl<double> CostModelContactForce;
typedef CostDataContactForceTpl<double> CostDataContactForce;
DEPRECATED("Use CostModelContactImpulse", typedef CostModelContactImpulseTpl<double> CostModelContactImpulse;)
DEPRECATED("Use CostDataContactImpulse", typedef CostDataContactImpulseTpl<double> CostDataContactImpulse;)
DEPRECATED("Use CostModelImpulseFrictionCone",
           typedef CostModelImpulseFrictionConeTpl<double> CostModelImpulseFrictionCone;)
DEPRECATED("Use CostDataImpulseFrictionCone",
           typedef CostDataImpulseFrictionConeTpl<double> CostDataImpulseFrictionCone;)
DEPRECATED("Use CostModelImpulseCoPPosition",
           typedef CostModelImpulseCoPPositionTpl<double> CostModelImpulseCoPPosition;)
DEPRECATED("Use CostDataImpulseCoPPosition", typedef CostDataImpulseCoPPositionTpl<double> CostDataImpulseCoPPosition;)
DEPRECATED("Use CostModelImpulseWrenchCone", typedef CostModelImpulseWrenchConeTpl<double> CostModelImpulseWrenchCone;)
DEPRECATED("Use CostDataImpulseWrenchCone", typedef CostDataImpulseWrenchConeTpl<double> CostDataImpulseWrenchCone;)

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
