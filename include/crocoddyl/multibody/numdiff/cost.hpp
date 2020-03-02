///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh, New York University,
//                          Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_NUMDIFF_COST_HPP_
#define CROCODDYL_MULTIBODY_NUMDIFF_COST_HPP_

#include <boost/function.hpp>
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/cost-base.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/centroidal-derivatives.hpp"

namespace crocoddyl {

// Simple renaming that ease the code writing.
typedef boost::function<void(const Eigen::VectorXd&)> ReevaluationFunction;

/**
 * @brief Compute all the pinocchio data needed for the numerical
 * differentiation. We use the address of the object to avoid a copy from the
 * "boost::bind".
 *
 * @param model is the rigid body robot model.
 * @param data contains the results of the computations.
 * @param x is the state vector.
 */
void updateAllPinocchio(pinocchio::Model* const model, pinocchio::Data* data, const Eigen::VectorXd& x) {
  const Eigen::VectorXd& q = x.segment(0, model->nq);
  const Eigen::VectorXd& v = x.segment(model->nq, model->nv);
  Eigen::VectorXd a = Eigen::VectorXd::Zero(model->nv);
  Eigen::Matrix<double, 6, Eigen::Dynamic> tmp;
  tmp.resize(6, model->nv);
  pinocchio::forwardKinematics(*model, *data, q);
  pinocchio::computeJointJacobians(*model, *data, q);
  pinocchio::updateFramePlacements(*model, *data);
  pinocchio::jacobianCenterOfMass(*model, *data, q);
  pinocchio::computeCentroidalMomentum(*model, *data, q, v);
  pinocchio::computeCentroidalDynamicsDerivatives(*model, *data, q, v, a, tmp, tmp, tmp, tmp);
}

template <typename _Scalar>
class CostNumDiffModelTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;

  /**
   * @brief Construct a new CostNumDiffModel object from a CostModelAbstract.
   *
   * @param model
   */
  CostNumDiffModelTpl(const boost::shared_ptr<Base>& model);

  /**
   * @brief Default destructor of the CostNumDiffModel object
   */
  ~CostNumDiffModelTpl();

  /**
   * @brief @copydoc ActionModelAbstract::calc()
   */
  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc ActionModelAbstract::calcDiff()
   */
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u, const bool& recalc = true);

  /**
   * @brief Create a Data object
   *
   * @param data is the DataCollector used by the original model.
   * @return boost::shared_ptr<CostModelAbstract>
   */
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Get the model_ object
   *
   * @return CostModelAbstract&
   */
  const boost::shared_ptr<Base>& get_model() const;

  /**
   * @brief Get the disturbance_ object
   *
   * @return const Scalar&
   */
  const Scalar& get_disturbance() const;

  /**
   * @brief Set the disturbance_ object
   *
   * @param disturbance is the value used to find the numerical derivative
   */
  void set_disturbance(const Scalar& disturbance);

  /**
   * @brief Identify if the Gauss approximation is going to be used or not.
   *
   * @return true
   * @return false
   */
  bool get_with_gauss_approx();

  /**
   * @brief Register functions that take a pinocchio model, a pinocchio
   * data, a state and a control. These function are called during the
   * evaluation of the gradient and hessian.
   *
   * @param reevals are the registered functions.
   */
  void set_reevals(const std::vector<ReevaluationFunction>& reevals);

 protected:
  /** @brief Model of the cost. */
  boost::shared_ptr<Base> model_;

  /** @brief Numerical disturbance used in the numerical differentiation. */
  Scalar disturbance_;

  /** @brief Functions that needs execution before calc or calcDiff. */
  std::vector<ReevaluationFunction> reevals_;

 private:
  /**
   * @brief Make sure that when we finite difference the Cost Model, the user
   * does not face unknown behaviour because of the finite differencing of a
   * quaternion around pi. This behaviour might occur if CostModelState and
   * FloatingInContact differential model are used together.
   *
   * For full discussions see issue
   * https://gepgitlab.laas.fr/loco-3d/crocoddyl/issues/139
   *
   * @param x is the state at which the check is performed.
   */
  virtual void assertStableStateFD(const Eigen::Ref<const Eigen::VectorXd>& /*x*/){};
};

template <typename _Scalar>
struct CostDataNumDiffTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;

  template <template <typename Scalar> class Model>
  explicit CostDataNumDiffTpl(Model<Scalar>* const model, DataCollectorAbstract* const shared_data)
      : Base(model, shared_data),
        dx(model->get_state()->get_ndx()),
        xp(model->get_state()->get_nx()),
        du(model->get_nu()),
        up(model->get_nu()) {
    dx.setZero();
    xp.setZero();
    du.setZero();
    up.setZero();

    const std::size_t& ndx = model->get_model()->get_state()->get_ndx();
    const std::size_t& nu = model->get_model()->get_nu();
    data_0 = model->get_model()->createData(shared_data);
    for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(model->get_model()->createData(shared_data));
    }
    for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(model->get_model()->createData(shared_data));
    }
  }

  virtual ~CostDataNumDiffTpl() {}

  using Base::shared;
  using Base::activation;
  using Base::cost;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::Ru;
  using Base::Rx;

  VectorXs dx;  //!< State disturbance.
  VectorXs xp;  //!< The integrated state from the disturbance on one DoF "\f$ \int x dx_i \f$".
  VectorXs du;  //!< Control disturbance.
  VectorXs up;  //!< The integrated control from the disturbance on one DoF "\f$ \int u du_i = u + du \f$".
  boost::shared_ptr<Base> data_0;                //!< The data at the approximation point.
  std::vector<boost::shared_ptr<Base> > data_x;  //!< The temporary data associated with the state variation.
  std::vector<boost::shared_ptr<Base> > data_u;  //!< The temporary data associated with the control variation.
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_NUMDIFF_COST_HPP_
