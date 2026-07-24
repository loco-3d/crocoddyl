///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh, INRIA
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_FWD_HPP_
#define CROCODDYL_MULTIBODY_FWD_HPP_

#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/pch.hpp"

namespace crocoddyl {

// actuation
template <typename Scalar>
class ActuationModelFloatingBaseTpl;

template <typename Scalar>
class ActuationModelFullTpl;

template <typename Scalar>
struct ThrusterTpl;

template <typename Scalar>
class ActuationModelFloatingBaseThrustersTpl;

template <typename Scalar>
class JointDynamicsModelAbstractTpl;
template <typename Scalar>
struct JointDynamicsDataAbstractTpl;
template <typename Scalar>
class JointDynamicsModelIdentityTpl;
template <typename Scalar>
struct JointDynamicsDataFrictionTpl;
template <typename Scalar>
class JointDynamicsModelFrictionTpl;
template <typename Scalar>
class JointDynamicsModelThrusterTpl;
template <typename Scalar>
class ActuationModelMultibodyTpl;
template <typename Scalar>
struct ActuationDataMultibodyTpl;
template <typename Scalar>
class ActuationMultibodyParamsTpl;
template <typename Scalar>
struct ActuationMultibodyParamsDataTpl;

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
class DifferentialActionModelFreeInvDynamicsTpl;
template <typename Scalar>
struct DifferentialActionDataFreeInvDynamicsTpl;

template <typename Scalar>
class DifferentialActionModelContactFwdDynamicsTpl;
template <typename Scalar>
struct DifferentialActionDataContactFwdDynamicsTpl;

template <typename Scalar>
class DifferentialActionModelContactInvDynamicsTpl;
template <typename Scalar>
struct DifferentialActionDataContactInvDynamicsTpl;

// numdiff
template <typename Scalar>
class CostModelNumDiffTpl;
template <typename Scalar>
struct CostDataNumDiffTpl;

template <typename Scalar>
class ContactModelNumDiffTpl;
template <typename Scalar>
struct ContactDataNumDiffTpl;

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

#ifdef PINOCCHIO_WITH_HPP_FCL
template <typename Scalar>
class ResidualModelPairCollisionTpl;
template <typename Scalar>
struct ResidualDataPairCollisionTpl;
#endif

// task
template <typename Scalar>
class TaskModelCentroidalMomentumTpl;
template <typename Scalar>
struct TaskDataCentroidalMomentumTpl;

template <typename Scalar>
class TaskModelFrameRotationTpl;
template <typename Scalar>
struct TaskDataFrameRotationTpl;

template <typename Scalar>
class TaskModelFrameTranslationTpl;
template <typename Scalar>
struct TaskDataFrameTranslationTpl;

template <typename Scalar>
class TaskModelFramePlacementTpl;
template <typename Scalar>
struct TaskDataFramePlacementTpl;

template <typename Scalar>
class TaskModelCoMPositionTpl;
template <typename Scalar>
struct TaskDataCoMPositionTpl;

template <typename Scalar>
class TaskModelJointPositionTpl;
template <typename Scalar>
struct TaskDataJointPositionTpl;

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
class ContactModel1DTpl;
template <typename Scalar>
struct ContactData1DTpl;

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

// implicit constraint
template <typename Scalar>
class ImplicitConstraintModelAbstractTpl;
template <typename Scalar>
struct ImplicitConstraintDataAbstractTpl;
template <typename Scalar>
struct ImplicitConstraintItemTpl;
template <typename Scalar>
class ImplicitConstraintModelMultipleTpl;
template <typename Scalar>
struct ImplicitConstraintDataMultipleTpl;
template <typename Scalar>
class KinematicLoopModelTpl;
template <typename Scalar>
struct KinematicLoopDataTpl;
template <typename Scalar>
class ContactModelTpl;
template <typename Scalar>
struct ContactDataTpl;

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
struct DataCollectorMultibodyParamsTpl;

template <typename Scalar>
struct DataCollectorActMultibodyTpl;

template <typename Scalar>
struct DataCollectorActMultibodyParamsTpl;

template <typename Scalar>
struct DataCollectorJointActMultibodyTpl;

template <typename Scalar>
struct DataCollectorJointActMultibodyParamsTpl;

template <typename Scalar>
struct DataCollectorContactTpl;

template <typename Scalar>
struct DataCollectorMultibodyInContactTpl;

template <typename Scalar>
struct DataCollectorMultibodyInContactParamsTpl;

template <typename Scalar>
struct DataCollectorActMultibodyInContactTpl;

template <typename Scalar>
struct DataCollectorActMultibodyInContactParamsTpl;

template <typename Scalar>
struct DataCollectorJointActMultibodyInContactTpl;

template <typename Scalar>
struct DataCollectorJointActMultibodyInContactParamsTpl;

template <typename Scalar>
struct DataCollectorImplicitConstraintTpl;

template <typename Scalar>
struct DataCollectorMultibodyInImplicitConstraintTpl;

template <typename Scalar>
struct DataCollectorMultibodyInImplicitConstraintParamsTpl;

template <typename Scalar>
struct DataCollectorActMultibodyInImplicitConstraintTpl;

template <typename Scalar>
struct DataCollectorActMultibodyInImplicitConstraintParamsTpl;

template <typename Scalar>
struct DataCollectorJointMultibodyInImplicitConstraintTpl;

template <typename Scalar>
struct DataCollectorJointMultibodyInImplicitConstraintParamsTpl;

template <typename Scalar>
struct DataCollectorJointActMultibodyInImplicitConstraintTpl;

template <typename Scalar>
struct DataCollectorJointActMultibodyInImplicitConstraintParamsTpl;

template <typename Scalar>
struct DataCollectorImpulseTpl;

template <typename Scalar>
struct DataCollectorMultibodyInImpulseTpl;

template <typename Scalar>
struct DataCollectorMultibodyInImpulseParamsTpl;

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

/**** Template Instantiation ****/

typedef ActuationModelFloatingBaseTpl<double> ActuationModelFloatingBase;
typedef ActuationModelFullTpl<double> ActuationModelFull;
typedef ThrusterTpl<double> Thruster;
typedef ActuationModelFloatingBaseThrustersTpl<double>
    ActuationModelFloatingBaseThrusters;
typedef JointDynamicsModelAbstractTpl<double> JointDynamicsModelAbstract;
typedef JointDynamicsDataAbstractTpl<double> JointDynamicsDataAbstract;
typedef JointDynamicsModelIdentityTpl<double> JointDynamicsModelIdentity;
typedef JointDynamicsDataFrictionTpl<double> JointDynamicsDataFriction;
typedef JointDynamicsModelFrictionTpl<double> JointDynamicsModelFriction;
typedef JointDynamicsModelThrusterTpl<double> JointDynamicsModelThruster;
typedef ActuationModelMultibodyTpl<double> ActuationModelMultibody;
typedef ActuationDataMultibodyTpl<double> ActuationDataMultibody;
typedef ActuationMultibodyParamsTpl<double> ActuationMultibodyParams;
typedef ActuationMultibodyParamsDataTpl<double> ActuationMultibodyParamsData;

typedef ForceDataAbstractTpl<double> ForceDataAbstract;

typedef ContactModelAbstractTpl<double> ContactModelAbstract;
typedef ContactDataAbstractTpl<double> ContactDataAbstract;

typedef ActionModelImpulseFwdDynamicsTpl<double> ActionModelImpulseFwdDynamics;
typedef ActionDataImpulseFwdDynamicsTpl<double> ActionDataImpulseFwdDynamics;

typedef DifferentialActionModelFreeFwdDynamicsTpl<double>
    DifferentialActionModelFreeFwdDynamics;
typedef DifferentialActionDataFreeFwdDynamicsTpl<double>
    DifferentialActionDataFreeFwdDynamics;
typedef DifferentialActionModelFreeInvDynamicsTpl<double>
    DifferentialActionModelFreeInvDynamics;
typedef DifferentialActionDataFreeInvDynamicsTpl<double>
    DifferentialActionDataFreeInvDynamics;
typedef DifferentialActionModelContactFwdDynamicsTpl<double>
    DifferentialActionModelContactFwdDynamics;
typedef DifferentialActionDataContactFwdDynamicsTpl<double>
    DifferentialActionDataContactFwdDynamics;
typedef DifferentialActionModelContactInvDynamicsTpl<double>
    DifferentialActionModelContactInvDynamics;
typedef DifferentialActionDataContactInvDynamicsTpl<double>
    DifferentialActionDataContactInvDynamics;

typedef CostModelNumDiffTpl<double> CostModelNumDiff;
typedef CostDataNumDiffTpl<double> CostDataNumDiff;
typedef ContactModelNumDiffTpl<double> ContactModelNumDiff;
typedef ContactDataNumDiffTpl<double> ContactDataNumDiff;

typedef FrictionConeTpl<double> FrictionCone;
typedef WrenchConeTpl<double> WrenchCone;
typedef CoPSupportTpl<double> CoPSupport;

typedef ResidualModelCentroidalMomentumTpl<double>
    ResidualModelCentroidalMomentum;
typedef ResidualDataCentroidalMomentumTpl<double>
    ResidualDataCentroidalMomentum;
typedef ResidualModelCoMPositionTpl<double> ResidualModelCoMPosition;
typedef ResidualDataCoMPositionTpl<double> ResidualDataCoMPosition;
typedef ResidualModelContactForceTpl<double> ResidualModelContactForce;
typedef ResidualDataContactForceTpl<double> ResidualDataContactForce;
typedef ResidualModelContactFrictionConeTpl<double>
    ResidualModelContactFrictionCone;
typedef ResidualDataContactFrictionConeTpl<double>
    ResidualDataContactFrictionCone;
typedef ResidualModelContactCoPPositionTpl<double>
    ResidualModelContactCoPPosition;
typedef ResidualDataContactCoPPositionTpl<double>
    ResidualDataContactCoPPosition;
typedef ResidualModelContactWrenchConeTpl<double>
    ResidualModelContactWrenchCone;
typedef ResidualDataContactWrenchConeTpl<double> ResidualDataContactWrenchCone;
typedef ResidualModelContactControlGravTpl<double>
    ResidualModelContactControlGrav;
typedef ResidualDataContactControlGravTpl<double>
    ResidualDataContactControlGrav;
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

#ifdef PINOCCHIO_WITH_HPP_FCL
typedef ResidualModelPairCollisionTpl<double> ResidualModelPairCollision;
typedef ResidualDataPairCollisionTpl<double> ResidualDataPairCollision;
#endif

typedef TaskModelCentroidalMomentumTpl<double> TaskModelCentroidalMomentum;
typedef TaskDataCentroidalMomentumTpl<double> TaskDataCentroidalMomentum;
typedef TaskModelFrameRotationTpl<double> TaskModelFrameRotation;
typedef TaskDataFrameRotationTpl<double> TaskDataFrameRotation;
typedef TaskModelFrameTranslationTpl<double> TaskModelFrameTranslation;
typedef TaskDataFrameTranslationTpl<double> TaskDataFrameTranslation;
typedef TaskModelFramePlacementTpl<double> TaskModelFramePlacement;
typedef TaskDataFramePlacementTpl<double> TaskDataFramePlacement;
typedef TaskModelCoMPositionTpl<double> TaskModelCoMPosition;
typedef TaskDataCoMPositionTpl<double> TaskDataCoMPosition;
typedef TaskModelJointPositionTpl<double> TaskModelJointPosition;
typedef TaskDataJointPositionTpl<double> TaskDataJointPosition;

typedef ImpulseModelAbstractTpl<double> ImpulseModelAbstract;
typedef ImpulseDataAbstractTpl<double> ImpulseDataAbstract;

enum ContactType {
  ContactUndefined,
  Contact1D,
  Contact2D,
  Contact3D,
  Contact6D
};
enum ImpulseType { ImpulseUndefined, Impulse3D, Impulse6D };

typedef ContactItemTpl<double> ContactItem;
typedef ContactModelMultipleTpl<double> ContactModelMultiple;
typedef ContactDataMultipleTpl<double> ContactDataMultiple;
typedef ContactModel1DTpl<double> ContactModel1D;
typedef ContactData1DTpl<double> ContactData1D;
typedef ContactModel2DTpl<double> ContactModel2D;
typedef ContactData2DTpl<double> ContactData2D;
typedef ContactModel3DTpl<double> ContactModel3D;
typedef ContactData3DTpl<double> ContactData3D;
typedef ContactModel6DTpl<double> ContactModel6D;
typedef ContactData6DTpl<double> ContactData6D;
typedef ImplicitConstraintModelAbstractTpl<double>
    ImplicitConstraintModelAbstract;
typedef ImplicitConstraintDataAbstractTpl<double>
    ImplicitConstraintDataAbstract;
typedef ImplicitConstraintItemTpl<double> ImplicitConstraintItem;
typedef ImplicitConstraintModelMultipleTpl<double>
    ImplicitConstraintModelMultiple;
typedef ImplicitConstraintDataMultipleTpl<double>
    ImplicitConstraintDataMultiple;
typedef KinematicLoopModelTpl<double> KinematicLoopModel;
typedef KinematicLoopDataTpl<double> KinematicLoopData;
typedef ContactModelTpl<double> ContactModel;
typedef ContactDataTpl<double> ContactData;

typedef StateMultibodyTpl<double> StateMultibody;

typedef DataCollectorMultibodyTpl<double> DataCollectorMultibody;
typedef DataCollectorMultibodyParamsTpl<double> DataCollectorMultibodyParams;
typedef DataCollectorActMultibodyTpl<double> DataCollectorActMultibody;
typedef DataCollectorActMultibodyParamsTpl<double>
    DataCollectorActMultibodyParams;
typedef DataCollectorJointActMultibodyTpl<double>
    DataCollectorJointActMultibody;
typedef DataCollectorJointActMultibodyParamsTpl<double>
    DataCollectorJointActMultibodyParams;
typedef DataCollectorContactTpl<double> DataCollectorContact;
typedef DataCollectorMultibodyInContactTpl<double>
    DataCollectorMultibodyInContact;
typedef DataCollectorMultibodyInContactParamsTpl<double>
    DataCollectorMultibodyInContactParams;
typedef DataCollectorActMultibodyInContactTpl<double>
    DataCollectorActMultibodyInContact;
typedef DataCollectorActMultibodyInContactParamsTpl<double>
    DataCollectorActMultibodyInContactParams;
typedef DataCollectorJointActMultibodyInContactTpl<double>
    DataCollectorJointActMultibodyInContact;
typedef DataCollectorJointActMultibodyInContactParamsTpl<double>
    DataCollectorJointActMultibodyInContactParams;
typedef DataCollectorImplicitConstraintTpl<double>
    DataCollectorImplicitConstraint;
typedef DataCollectorMultibodyInImplicitConstraintTpl<double>
    DataCollectorMultibodyInImplicitConstraint;
typedef DataCollectorMultibodyInImplicitConstraintParamsTpl<double>
    DataCollectorMultibodyInImplicitConstraintParams;
typedef DataCollectorActMultibodyInImplicitConstraintTpl<double>
    DataCollectorActMultibodyInImplicitConstraint;
typedef DataCollectorActMultibodyInImplicitConstraintParamsTpl<double>
    DataCollectorActMultibodyInImplicitConstraintParams;
typedef DataCollectorJointMultibodyInImplicitConstraintTpl<double>
    DataCollectorJointMultibodyInImplicitConstraint;
typedef DataCollectorJointMultibodyInImplicitConstraintParamsTpl<double>
    DataCollectorJointMultibodyInImplicitConstraintParams;
typedef DataCollectorJointActMultibodyInImplicitConstraintTpl<double>
    DataCollectorJointActMultibodyInImplicitConstraint;
typedef DataCollectorJointActMultibodyInImplicitConstraintParamsTpl<double>
    DataCollectorJointActMultibodyInImplicitConstraintParams;
typedef DataCollectorImpulseTpl<double> DataCollectorImpulse;
typedef DataCollectorMultibodyInImpulseTpl<double>
    DataCollectorMultibodyInImpulse;
typedef DataCollectorMultibodyInImpulseParamsTpl<double>
    DataCollectorMultibodyInImpulseParams;

typedef ImpulseModel6DTpl<double> ImpulseModel6D;
typedef ImpulseData6DTpl<double> ImpulseData6D;
typedef ImpulseModel3DTpl<double> ImpulseModel3D;
typedef ImpulseData3DTpl<double> ImpulseData3D;
typedef ImpulseItemTpl<double> ImpulseItem;
typedef ImpulseModelMultipleTpl<double> ImpulseModelMultiple;
typedef ImpulseDataMultipleTpl<double> ImpulseDataMultiple;

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_FWD_HPP_
