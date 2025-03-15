///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_PINOCCHIO_MODEL_FACTORY_HXX_
#define CROCODDYL_PINOCCHIO_MODEL_FACTORY_HXX_

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal-derivatives.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/multibody/model.hpp>

namespace crocoddyl {
namespace unittest {
/**
 * @brief Compute all the pinocchio data needed for the numerical
 * differentiation. We use the address of the object to avoid a copy from the
 * "boost::bind".
 *
 * @param model is the rigid body robot model.
 * @param data contains the results of the computations.
 * @param x is the state vector.
 */
template <typename Scalar, int Options,
          template <typename, int> class JointCollectionTpl>
void updateAllPinocchio(
    pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl>* const model,
    pinocchio::DataTpl<Scalar, Options, JointCollectionTpl>* data,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>&) {
  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& q = x.segment(0, model->nq);
  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& v =
      x.segment(model->nq, model->nv);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> a =
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(model->nv);
  Eigen::Matrix<Scalar, 6, Eigen::Dynamic> tmp;
  tmp.resize(6, model->nv);
  pinocchio::forwardKinematics(*model, *data, q, v, a);
  pinocchio::computeForwardKinematicsDerivatives(*model, *data, q, v, a);
  pinocchio::computeJointJacobians(*model, *data, q);
  pinocchio::updateFramePlacements(*model, *data);
  pinocchio::centerOfMass(*model, *data, q, v, a);
  pinocchio::jacobianCenterOfMass(*model, *data, q);
  pinocchio::computeCentroidalMomentum(*model, *data, q, v);
  pinocchio::computeCentroidalDynamicsDerivatives(*model, *data, q, v, a, tmp,
                                                  tmp, tmp, tmp);
  pinocchio::computeRNEADerivatives(*model, *data, q, v, a);
}

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_PINOCCHIO_MODEL_FACTORY_HXX_
