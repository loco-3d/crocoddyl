// BSD 3-Clause License
//
// Copyright (C) 2024, LAAS-CNRS.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.

#ifndef COLMPC_FWDL_HPP_
#define COLMPC_FWDL_HPP_

#include <pinocchio/fwd.hpp>

namespace colmpc {

template <typename Scalar>
class ResidualDistanceCollisionTpl;
typedef ResidualDistanceCollisionTpl<double> ResidualDistanceCollision;
template <typename Scalar>
class ResidualDataDistanceCollisionTpl;
typedef ResidualDataDistanceCollisionTpl<double> ResidualDataDistanceCollision;

template <typename Scalar>
class ResidualDistanceCollision2Tpl;
typedef ResidualDistanceCollision2Tpl<double> ResidualDistanceCollision2;
template <typename Scalar>
class ResidualDataDistanceCollision2Tpl;
typedef ResidualDataDistanceCollision2Tpl<double>
    ResidualDataDistanceCollision2;

template <typename Scalar>
class ResidualModelVelocityAvoidanceTpl;
typedef ResidualModelVelocityAvoidanceTpl<double>
    ResidualModelVelocityAvoidance;
template <typename Scalar>
class ResidualDataVelocityAvoidanceTpl;
typedef ResidualDataVelocityAvoidanceTpl<double> ResidualDataVelocityAvoidance;

template <typename Scalar>
class StateMultibodyTpl;
typedef StateMultibodyTpl<double> StateMultibody;

template <typename Scalar>
class DifferentialActionModelFreeFwdDynamicsTpl;
template <typename Scalar>
class DifferentialActionDataFreeFwdDynamicsTpl;
typedef DifferentialActionModelFreeFwdDynamicsTpl<double>
    DifferentialActionModelFreeFwdDynamics;

template <typename Scalar, int N>
class ActivationModelExpTpl;
typedef ActivationModelExpTpl<double, 1> ActivationModelExp;
typedef ActivationModelExpTpl<double, 2> ActivationModelQuadExp;
template <typename Scalar, int N>
class ActivationDataExpTpl;
typedef ActivationDataExpTpl<double, 1> ActivationDataExp;
typedef ActivationDataExpTpl<double, 2> ActivationDataQuadExp;

template <typename Scalar>
class ActivationModelDistanceQuadTpl;
typedef ActivationModelDistanceQuadTpl<double> ActivationModelDistanceQuad;
template <typename Scalar>
class ActivationDataDistanceQuadTpl;
typedef ActivationDataDistanceQuadTpl<double> ActivationDataDistanceQuad;

}  // namespace colmpc

#endif  // COLMPC_FWDL_HPP_
