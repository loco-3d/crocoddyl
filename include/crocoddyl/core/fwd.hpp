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

// DiffAction
template <typename Scalar>
class DifferentialActionModelAbstractTpl;

template <typename Scalar>
class DifferentialActionDataAbstractTpl;

template <typename Scalar>
class DifferentialActionModelLQRTpl;

template <typename Scalar>
class DifferentialActionDataLQRTpl;

// Actions
template <typename Scalar>
class ActionModelUnicycleTpl;

template <typename Scalar>
class ActionDataUnicycleTpl;

template <typename Scalar>
class ActionModelLQRTpl;

template <typename Scalar>
class ActionDataLQRTpl;

// DataCollector
template <typename Scalar>
class DataCollectorAbstractTpl;

// Activations
template <typename Scalar>
class ActivationDataQuadraticBarrierTpl;

template <typename Scalar>
class ActivationModelQuadraticBarrierTpl;

template <typename Scalar>
class ActivationBoundsTpl;

template <typename Scalar>
class ActivationModelWeightedQuadraticBarrierTpl;

template <typename Scalar>
class ActivationModelQuadTpl;

template <typename Scalar>
class ActivationModelWeightedQuadTpl;

template <typename Scalar>
class ActivationDataWeightedQuadTpl;

template <typename Scalar>
class ActivationModelSmoothAbsTpl;

template <typename Scalar>
class ActivationDataSmoothAbsTpl;

template <typename Scalar>
class ActivationModelAbstractTpl;

template <typename Scalar>
class ActivationDataAbstractTpl;

// State
template <typename Scalar>
class StateAbstractTpl;

// Actuations
template <typename Scalar>
class ActuationDataAbstractTpl;

template <typename Scalar>
class ActuationModelAbstractTpl;

// Squashing
template <typename Scalar>
class SquashingDataAbstractTpl;

template <typename Scalar>
class SquashingModelAbstractTpl;

template <typename Scalar>
class SquashingModelSmoothSatTpl;

// shooting
template <typename Scalar>
class ShootingProblemTpl;

// IAMs
template <typename Scalar>
class IntegratedActionModelEulerTpl;

template <typename Scalar>
class IntegratedActionDataEulerTpl;

// State
template <typename Scalar>
class StateVectorTpl;

// Datacollect
template <typename Scalar>
class DataCollectorActuationTpl;

// ActionData
template <typename Scalar>
class ActionDataAbstractTpl;

template <typename Scalar>
class ActionModelAbstractTpl;

// Numdiff
template <typename Scalar>
class ActionModelNumDiffTpl;

template <typename Scalar>
class ActionDataNumDiffTpl;

template <typename Scalar>
class DifferentialActionModelNumDiffTpl;

template <typename Scalar>
class DifferentialActionDataNumDiffTpl;

template <typename Scalar>
class ActivationModelNumDiffTpl;

template <typename Scalar>
class ActivationDataNumDiffTpl;

template <typename Scalar>
class StateNumDiffTpl;

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
typedef ActivationModelAbstractTpl<double> ActivationModelAbstract;
typedef ActivationDataAbstractTpl<double> ActivationDataAbstract;

typedef StateAbstractTpl<double> StateAbstract;

typedef ActuationDataAbstractTpl<double> ActuationDataAbstract;
typedef ActuationModelAbstractTpl<double> ActuationModelAbstract;

typedef SquashingDataAbstractTpl<double> SquashingDataAbstract;
typedef SquashingModelAbstractTpl<double> SquashingModelAbstract;
typedef SquashingModelSmoothSatTpl<double> SquashingModelSmoothSat;

typedef ShootingProblemTpl<double> ShootingProblem;

typedef IntegratedActionModelEulerTpl<double> IntegratedActionModelEuler;
typedef IntegratedActionDataEulerTpl<double> IntegratedActionDataEuler;

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

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_FWD_HPP_
