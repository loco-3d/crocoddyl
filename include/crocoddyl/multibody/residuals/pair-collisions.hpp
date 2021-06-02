///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_PAIR_COLLISIONS_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_PAIR_COLLISIONS_HPP_

#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "pinocchio/multibody/geometry.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/multibody/fcl.hpp"

namespace crocoddyl {

/**
 * @brief Pair collisions residual
 *
 * This residual function defines the euclidean distance between a geometric collision pair as 
 * \f$\mathbf{r}=\mathbf{p}_1-\mathbf{p}_2^*\f$, where \f$\mathbf{p}_1,\mathbf{p}_2^*\in~\mathbb{R}^3\f$
 * are the nearest points on each collision object with respect to the other object. One of the objects is
 * a body frame of the robot, the other is an external object. Note that for the
 * sake of fast computation, it is easier to define the collision objects as inflated capsules.
 * Note also that the dimension of the residual vector is obtained from 3. Furthermore, the Jacobians 
 * of the residual function are computed analytically.
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its Jacobians are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelPairCollisionsTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataPairCollisionsTpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef pinocchio::GeometryModel GeometryModel;
  
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the pair collisions residual model
   *
   * @param[in] state  State of the multibody system
   * @param[in] nu   Dimension of the control vector
   * @param[in] geom_model   Pinocchio geometry model containing the collision pair
   * @param[in] pair_id   Index of the collision pair in the geometry model
   * @param[in] joint_id   Index of the nearest joint on which the collision link is attached
   */
  
  ResidualModelPairCollisionsTpl(boost::shared_ptr<StateMultibody> state,
                             const std::size_t& nu,
                             boost::shared_ptr<GeometryModel> geom_model,
                             const pinocchio::PairIndex& pair_id, 
                             const pinocchio::JointIndex& joint_id);

 
  virtual ~ResidualModelPairCollisionsTpl();

   /**
   * @brief Compute the pair collisions residual
   *
   * @param[in] data  Pair collisions residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);

  /**
   * @brief Compute the derivatives of the pair collisions residual
   *
   * @param[in] data  Pair collisions residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);
  
  virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract *const data);
  
  /**
   * @brief Return the Pinocchio geometry model
   */
  const pinocchio::GeometryModel& get_geometryModel() const;

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::unone_;
  using Base::v_dependent_;

 private:
  typename StateMultibody::PinocchioModel pin_model_;  //!< Pinocchio model used for internal computations
  boost::shared_ptr<pinocchio::GeometryModel > geom_model_; //!< Pinocchio geometry model containing collision pair
  pinocchio::PairIndex pair_id_; //!< Index of the collision pair in geometry model
  pinocchio::JointIndex joint_id_; //!< Index of joint on which the collision body frame of the robot is attached
};

template <typename _Scalar>
struct ResidualDataPairCollisionsTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  typedef typename MathBase::Matrix6xs Matrix6xs;

  template <template <typename Scalar> class Model>
  ResidualDataPairCollisionsTpl(Model<Scalar> *const model, DataCollectorAbstract *const data)
   : Base(model, data),
     geom_data(pinocchio::GeometryData(model->get_geometryModel())),
     J(Matrix6xs::Zero(6, model->get_state()->get_nv())) {
    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar> *d = dynamic_cast<DataCollectorMultibodyTpl<Scalar> *>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorActMultibodyTpl");
    }
    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }
  pinocchio::GeometryData geom_data;    //!< Pinocchio geometry data
  pinocchio::DataTpl<Scalar>* pinocchio;   //!< Pinocchio data
  Matrix6xs J;           //!< Jacobian at the collision joint
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/pair-collisions.hxx"

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_PAIR_COLLISIONS_HPP_
