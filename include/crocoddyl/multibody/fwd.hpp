///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FWD_HPP_
#define CROCODDYL_MULTIBODY_FWD_HPP_

namespace crocoddyl {

// actuation
template <typename Scalar>
class ActuationModelFloatingBaseTpl;

template <typename Scalar>
class ActuationModelFullTpl;

template <typename Scalar>
class ActuationModelMultiCopterBaseTpl;

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
struct FrameCoPSupportTpl;

// cost
template <typename Scalar>
class CostModelAbstractTpl;
template <typename Scalar>
struct CostDataAbstractTpl;

template <typename Scalar>
class CostModelFrameTranslationTpl;
template <typename Scalar>
struct CostDataFrameTranslationTpl;

template <typename Scalar>
struct CostItemTpl;
template <typename Scalar>
class CostModelSumTpl;
template <typename Scalar>
struct CostDataSumTpl;

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
class CostModelFrameVelocityTpl;
template <typename Scalar>
struct CostDataFrameVelocityTpl;

template <typename Scalar>
class CostModelContactFrictionConeTpl;
template <typename Scalar>
struct CostDataContactFrictionConeTpl;

template <typename Scalar>
class CostModelContactForceTpl;
template <typename Scalar>
struct CostDataContactForceTpl;

template <typename Scalar>
class CostModelContactImpulseTpl;
template <typename Scalar>
struct CostDataContactImpulseTpl;

template <typename Scalar>
class CostModelControlTpl;

template <typename Scalar>
class CostModelFrameRotationTpl;
template <typename Scalar>
struct CostDataFrameRotationTpl;

template <typename Scalar>
class CostModelImpulseFrictionConeTpl;
template <typename Scalar>
struct CostDataImpulseFrictionConeTpl;

template <typename Scalar>
class CostModelContactCoPPositionTpl;
template <typename Scalar>
struct CostDataContactCoPPositionTpl;

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

typedef FrameTranslationTpl<double> FrameTranslation;
typedef FrameRotationTpl<double> FrameRotation;
typedef FramePlacementTpl<double> FramePlacement;
typedef FrameMotionTpl<double> FrameMotion;
typedef FrameForceTpl<double> FrameForce;
typedef FrameFrictionConeTpl<double> FrameFrictionCone;
typedef FrameCoPSupportTpl<double> FrameCoPSupport;

typedef CostModelAbstractTpl<double> CostModelAbstract;
typedef CostDataAbstractTpl<double> CostDataAbstract;
typedef CostModelFrameTranslationTpl<double> CostModelFrameTranslation;
typedef CostDataFrameTranslationTpl<double> CostDataFrameTranslation;
typedef CostItemTpl<double> CostItem;
typedef CostModelSumTpl<double> CostModelSum;
typedef CostDataSumTpl<double> CostDataSum;
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
typedef CostModelFrameVelocityTpl<double> CostModelFrameVelocity;
typedef CostDataFrameVelocityTpl<double> CostDataFrameVelocity;
typedef CostModelContactCoPPositionTpl<double> CostModelContactCoPPosition;
typedef CostDataContactCoPPositionTpl<double> CostDataContactCoPPosition;
typedef CostModelContactFrictionConeTpl<double> CostModelContactFrictionCone;
typedef CostDataContactFrictionConeTpl<double> CostDataContactFrictionCone;
typedef CostModelContactForceTpl<double> CostModelContactForce;
typedef CostDataContactForceTpl<double> CostDataContactForce;
typedef CostModelContactImpulseTpl<double> CostModelContactImpulse;
typedef CostDataContactImpulseTpl<double> CostDataContactImpulse;
typedef CostModelControlTpl<double> CostModelControl;
typedef CostModelFrameRotationTpl<double> CostModelFrameRotation;
typedef CostModelImpulseFrictionConeTpl<double> CostModelImpulseFrictionCone;
typedef CostDataImpulseFrictionConeTpl<double> CostDataImpulseFrictionCone;

typedef CostDataFrameRotationTpl<double> CostDataFrameRotation;

typedef ImpulseModelAbstractTpl<double> ImpulseModelAbstract;
typedef ImpulseDataAbstractTpl<double> ImpulseDataAbstract;

typedef ContactItemTpl<double> ContactItem;
typedef ContactModelMultipleTpl<double> ContactModelMultiple;
typedef ContactDataMultipleTpl<double> ContactDataMultiple;
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
