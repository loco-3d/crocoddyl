///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_NUMDIFF_CONTACT_HPP_
#define CROCODDYL_MULTIBODY_NUMDIFF_CONTACT_HPP_

#include <boost/function.hpp>
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/contact-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ContactModelNumDiffTpl : public ContactModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ContactDataAbstractTpl<Scalar> ContactDataAbstract;
  typedef ContactModelAbstractTpl<Scalar> Base;
  typedef ContactDataNumDiffTpl<Scalar> Data;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef boost::function<void(const typename MathBaseTpl<Scalar>::VectorXs&)> ReevaluationFunction;

  /**
   * @brief Construct a new ContactModelNumDiff object from a ContactModelAbstract.
   *
   * @param model
   */
  explicit ContactModelNumDiffTpl(const boost::shared_ptr<Base>& model);

  /**
   * @brief Default destructor of the ContactModelNumDiff object
   */
  virtual ~ContactModelNumDiffTpl();

  /**
   * @brief @copydoc ContactModelAbstract::calc()
   */
  void calc(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief @copydoc ContactModelAbstract::calcDiff()
   */
  void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief @copydoc ContactModelAbstract::updateForce()
   */
  void updateForce(const boost::shared_ptr<ContactDataAbstract>& data, const VectorXs& force);

  /**
   * @brief Create a Data object
   *
   * @param data is the Pinocchio data
   * @return boost::shared_ptr<ContactModelAbstract>
   */
  boost::shared_ptr<ContactDataAbstract> createData(pinocchio::DataTpl<Scalar>* const data);

  /**
   * @brief Get the model_ object
   *
   * @return ContactModelAbstract&
   */
  const boost::shared_ptr<Base>& get_model() const;

  /**
   * @brief Get the disturbance_ object
   *
   * @return Scalar
   */
  Scalar get_disturbance() const;

  /**
   * @brief Set the disturbance_ object
   *
   * @param disturbance is the value used to find the numerical derivative
   */
  void set_disturbance(Scalar disturbance);

  /**
   * @brief Register functions that take a pinocchio model, a pinocchio data, a state and a control.
   * The updated data is used to evaluate of the gradient and hessian.
   *
   * @param reevals are the registered functions.
   */
  void set_reevals(const std::vector<ReevaluationFunction>& reevals);

 protected:
  using Base::nc_;
  using Base::nu_;
  using Base::state_;

  /** @brief Model of the Contact. */
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
  void assertStableStateFD(const Eigen::Ref<const VectorXs>& /*x*/);
};

template <typename _Scalar>
struct ContactDataNumDiffTpl : public ContactDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ContactDataAbstractTpl<Scalar> Base;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;

  template <template <typename Scalar> class Model>
  explicit ContactDataNumDiffTpl(Model<Scalar>* const model, pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data), dx(model->get_state()->get_ndx()), xp(model->get_state()->get_nx()) {
    dx.setZero();
    xp.setZero();

    std::size_t ndx = model->get_model()->get_state()->get_ndx();
    data_0 = model->get_model()->createData(data);
    for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(model->get_model()->createData(data));
    }
  }

  virtual ~ContactDataNumDiffTpl() {}

  using Base::a0;
  using Base::da0_dx;
  using Base::f;
  using Base::pinocchio;

  VectorXs dx;                     //!< State disturbance.
  VectorXs xp;                     //!< The integrated state from the disturbance on one DoF "\f$ \int x dx_i \f$".
  boost::shared_ptr<Base> data_0;  //!< The data at the approximation point.
  std::vector<boost::shared_ptr<Base> > data_x;  //!< The temporary data associated with the state variation.
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/numdiff/contact.hxx"

#endif  // CROCODDYL_MULTIBODY_NUMDIFF_CONTACT_HPP_
