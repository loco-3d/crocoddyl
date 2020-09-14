///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_FWD_HPP_
#define CROCODDYL_CORE_FWD_HPP_

namespace crocoddyl {

// differential action
template <typename Scalar>
class DifferentialActionModelAbstractTpl;
template <typename Scalar>
struct DifferentialActionDataAbstractTpl;

template <typename Scalar>
class DifferentialActionModelLQRTpl;
template <typename Scalar>
struct DifferentialActionDataLQRTpl;

// action
template <typename Scalar>
class ActionModelUnicycleTpl;
template <typename Scalar>
struct ActionDataUnicycleTpl;

template <typename Scalar>
class ActionModelLQRTpl;
template <typename Scalar>
struct ActionDataLQRTpl;

// data collector
template <typename Scalar>
struct DataCollectorAbstractTpl;

// activation
template <typename Scalar>
struct ActivationBoundsTpl;
template <typename Scalar>
class ActivationModelQuadraticBarrierTpl;
template <typename Scalar>
struct ActivationDataQuadraticBarrierTpl;

template <typename Scalar>
class ActivationModelWeightedQuadraticBarrierTpl;

template <typename Scalar>
class ActivationModelQuadTpl;

template <typename Scalar>
class ActivationModelWeightedQuadTpl;
template <typename Scalar>
struct ActivationDataWeightedQuadTpl;

template <typename Scalar>
class ActivationModelSmoothAbsTpl;
template <typename Scalar>
struct ActivationDataSmoothAbsTpl;

template <typename Scalar>
class ActivationModelCollisionTpl;
template <typename Scalar>
struct ActivationDataCollisionTpl;

template <typename Scalar>
class ActivationModelAbstractTpl;
template <typename Scalar>
struct ActivationDataAbstractTpl;

// state
template <typename Scalar>
class StateAbstractTpl;

// actuation
template <typename Scalar>
class ActuationModelAbstractTpl;
template <typename Scalar>
struct ActuationDataAbstractTpl;

template <typename Scalar>
class ActuationSquashingModelTpl;
template <typename Scalar>
struct ActuationSquashingDataTpl;

// squashing
template <typename Scalar>
class SquashingModelAbstractTpl;
template <typename Scalar>
struct SquashingDataAbstractTpl;

template <typename Scalar>
class SquashingModelSmoothSatTpl;

// shooting
template <typename Scalar>
class ShootingProblemTpl;

// integrated action
template <typename Scalar>
class IntegratedActionModelEulerTpl;
template <typename Scalar>
struct IntegratedActionDataEulerTpl;

template <typename Scalar>
class IntegratedActionModelRK4Tpl;
template <typename Scalar>
struct IntegratedActionDataRK4Tpl;

// state
template <typename Scalar>
class StateVectorTpl;

// data collector
template <typename Scalar>
struct DataCollectorActuationTpl;

// ActionData
template <typename Scalar>
struct ActionDataAbstractTpl;

template <typename Scalar>
class ActionModelAbstractTpl;

// Numdiff
template <typename Scalar>
class ActionModelNumDiffTpl;
template <typename Scalar>
struct ActionDataNumDiffTpl;

template <typename Scalar>
class DifferentialActionModelNumDiffTpl;
template <typename Scalar>
struct DifferentialActionDataNumDiffTpl;

template <typename Scalar>
class ActivationModelNumDiffTpl;
template <typename Scalar>
struct ActivationDataNumDiffTpl;

template <typename Scalar>
class StateNumDiffTpl;

template <typename Scalar>
class ActuationModelNumDiffTpl;
template <typename Scalar>
struct ActuationDataNumDiffTpl;

template <typename Scalar>
class ActionModelCodeGenTpl;

template <typename Scalar>
struct ActionDataCodeGenTpl;

/********************Template Instantiation*************/
typedef DifferentialActionModelAbstractTpl<double> DifferentialActionModelAbstract;
typedef DifferentialActionDataAbstractTpl<double> DifferentialActionDataAbstract;
typedef DifferentialActionModelLQRTpl<double> DifferentialActionModelLQR;
typedef DifferentialActionDataLQRTpl<double> DifferentialActionDataLQR;

typedef ActionModelUnicycleTpl<double> ActionModelUnicycle;
typedef ActionDataUnicycleTpl<double> ActionDataUnicycle;
typedef ActionModelLQRTpl<double> ActionModelLQR;
typedef ActionDataLQRTpl<double> ActionDataLQR;

typedef DataCollectorAbstractTpl<double> DataCollectorAbstract;

typedef ActivationDataQuadraticBarrierTpl<double> ActivationDataQuadraticBarrier;
typedef ActivationModelQuadraticBarrierTpl<double> ActivationModelQuadraticBarrier;
typedef ActivationBoundsTpl<double> ActivationBounds;
typedef ActivationModelWeightedQuadraticBarrierTpl<double> ActivationModelWeightedQuadraticBarrier;
typedef ActivationModelQuadTpl<double> ActivationModelQuad;
typedef ActivationModelWeightedQuadTpl<double> ActivationModelWeightedQuad;
typedef ActivationDataWeightedQuadTpl<double> ActivationDataWeightedQuad;
typedef ActivationModelSmoothAbsTpl<double> ActivationModelSmoothAbs;
typedef ActivationDataSmoothAbsTpl<double> ActivationDataSmoothAbs;
typedef ActivationModelCollisionTpl<double> ActivationModelCollision;
typedef ActivationDataCollisionTpl<double> ActivationDataCollision;
typedef ActivationModelAbstractTpl<double> ActivationModelAbstract;
typedef ActivationDataAbstractTpl<double> ActivationDataAbstract;

typedef StateAbstractTpl<double> StateAbstract;

typedef ActuationDataAbstractTpl<double> ActuationDataAbstract;
typedef ActuationModelAbstractTpl<double> ActuationModelAbstract;
typedef ActuationSquashingDataTpl<double> ActuationSquashingData;
typedef ActuationSquashingModelTpl<double> ActuationSquashingModel;

typedef SquashingDataAbstractTpl<double> SquashingDataAbstract;
typedef SquashingModelAbstractTpl<double> SquashingModelAbstract;
typedef SquashingModelSmoothSatTpl<double> SquashingModelSmoothSat;

typedef ShootingProblemTpl<double> ShootingProblem;

typedef IntegratedActionModelEulerTpl<double> IntegratedActionModelEuler;
typedef IntegratedActionDataEulerTpl<double> IntegratedActionDataEuler;
typedef IntegratedActionModelRK4Tpl<double> IntegratedActionModelRK4;
typedef IntegratedActionDataRK4Tpl<double> IntegratedActionDataRK4;

typedef StateVectorTpl<double> StateVector;

typedef DataCollectorActuationTpl<double> DataCollectorActuation;

typedef ActionDataAbstractTpl<double> ActionDataAbstract;
typedef ActionModelAbstractTpl<double> ActionModelAbstract;

typedef ActionModelNumDiffTpl<double> ActionModelNumDiff;
typedef ActionDataNumDiffTpl<double> ActionDataNumDiff;
typedef DifferentialActionModelNumDiffTpl<double> DifferentialActionModelNumDiff;
typedef DifferentialActionDataNumDiffTpl<double> DifferentialActionDataNumDiff;
typedef ActivationModelNumDiffTpl<double> ActivationModelNumDiff;
typedef ActivationDataNumDiffTpl<double> ActivationDataNumDiff;
typedef StateNumDiffTpl<double> StateNumDiff;
typedef ActuationModelNumDiffTpl<double> ActuationModelNumDiff;
typedef ActuationDataNumDiffTpl<double> ActuationDataNumDiff;

typedef ActionModelCodeGenTpl<double> ActionModelCodeGen;
typedef ActionDataCodeGenTpl<double> ActionDataCodeGen;

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_FWD_HPP_
