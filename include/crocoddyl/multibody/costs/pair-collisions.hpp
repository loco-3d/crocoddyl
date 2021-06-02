///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_PAIR_COLLISIONS_HPP_
#define CROCODDYL_MULTIBODY_COSTS_PAIR_COLLISIONS_HPP_

#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/residuals/pair-collisions.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {


template <typename _Scalar>
class CostModelPairCollisionsTpl : public CostModelResidualTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelResidualTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ActivationModelNorm2BarrierTpl<Scalar> ActivationModelNorm2Barrier;
  typedef ResidualModelPairCollisionsTpl<Scalar> ResidualModelPairCollisions;
  typedef pinocchio::GeometryModel GeometryModel;
  
  typedef typename MathBase::VectorXs Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;


  CostModelPairCollisionsTpl(boost::shared_ptr<StateMultibody> state,
                          const Scalar& threshold,
                          const std::size_t& nu,
                          boost::shared_ptr<GeometryModel> geom_model,
                          const pinocchio::PairIndex& pair_id, // const std::size_t col_id, // The id of the pair of colliding objects
                          const pinocchio::JointIndex& joint_id);

  virtual ~CostModelPairCollisionsTpl();

 protected:
  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;
};

} // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/pair-collisions.hxx"

#endif // CROCODDYL_MULTIBODY_COSTS_PAIR_COLLISIONS_HPP_
