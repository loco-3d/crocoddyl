///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, LAAS-CNRS,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_NUMDIFF_CONTACT_HPP_
#define CROCODDYL_MULTIBODY_NUMDIFF_CONTACT_HPP_

#include <boost/function.hpp>

#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ContactModelNumDiffTpl : public ContactModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ContactModelBase, ContactModelNumDiffTpl)

  typedef _Scalar Scalar;
  typedef ContactDataAbstractTpl<Scalar> ContactDataAbstract;
  typedef ContactModelAbstractTpl<Scalar> Base;
  typedef ContactDataNumDiffTpl<Scalar> Data;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef boost::function<void(const VectorXs&, const VectorXs&)>
      ReevaluationFunction;

  /**
   * @brief Construct a new ContactModelNumDiff object from a
   * ContactModelAbstract.
   *
   * @param model
   */
  explicit ContactModelNumDiffTpl(const std::shared_ptr<Base>& model);

  /**
   * @brief Default destructor of the ContactModelNumDiff object
   */
  virtual ~ContactModelNumDiffTpl() = default;

  /**
   * @brief @copydoc ContactModelAbstract::calc()
   */
  void calc(const std::shared_ptr<ContactDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief @copydoc ContactModelAbstract::calcDiff()
   */
  void calcDiff(const std::shared_ptr<ContactDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief @copydoc ContactModelAbstract::updateForce()
   */
  void updateForce(const std::shared_ptr<ContactDataAbstract>& data,
                   const VectorXs& force) override;

  /**
   * @brief Create a Data object
   *
   * @param data is the Pinocchio data
   * @return std::shared_ptr<ContactModelAbstract>
   */
  std::shared_ptr<ContactDataAbstract> createData(
      pinocchio::DataTpl<Scalar>* const data) override;

  template <typename NewScalar>
  ContactModelNumDiffTpl<NewScalar> cast() const;

  /**
   * @brief Return the acton model that we use to numerical differentiate
   */
  const std::shared_ptr<Base>& get_model() const;

  /**
   * @brief Return the disturbance constant used in the numerical
   * differentiation routine
   */
  const Scalar get_disturbance() const;

  /**
   * @brief Modify the disturbance constant used in the numerical
   * differentiation routine
   */
  void set_disturbance(const Scalar disturbance);

  /**
   * @brief Register functions that take a pinocchio model, a pinocchio data, a
   * state and a control. The updated data is used to evaluate of the gradient
   * and Hessian.
   *
   * @param reevals are the registered functions.
   */
  void set_reevals(const std::vector<ReevaluationFunction>& reevals);

 protected:
  using Base::nc_;
  using Base::nu_;
  using Base::state_;

  std::shared_ptr<Base> model_;  //!<  contact model to differentiate
  Scalar e_jac_;  //!< Constant used for computing disturbances in Jacobian
                  //!< calculation
  std::vector<ReevaluationFunction>
      reevals_;  //!< functions that need execution before calc or calcDiff

 private:
  /**
   * @brief Make sure that when we finite difference the Cost Model, the user
   * does not face unknown behaviour because of the finite differencing of a
   * quaternion around pi. This behaviour might occur if state cost in
   * floating systems.
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
  explicit ContactDataNumDiffTpl(Model<Scalar>* const model,
                                 pinocchio::DataTpl<Scalar>* const data)
      : Base(model, data),
        dx(model->get_state()->get_ndx()),
        xp(model->get_state()->get_nx()) {
    dx.setZero();
    xp.setZero();

    const std::size_t ndx = model->get_model()->get_state()->get_ndx();
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

  Scalar x_norm;  //!< Norm of the state vector
  Scalar
      xh_jac;  //!< Disturbance value used for computing \f$ \ell_\mathbf{x} \f$
  VectorXs dx;  //!< State disturbance.
  VectorXs xp;  //!< The integrated state from the disturbance on one DoF "\f$
                //!< \int x dx_i \f$".
  std::shared_ptr<Base> data_0;  //!< The data at the approximation point.
  std::vector<std::shared_ptr<Base> >
      data_x;  //!< The temporary data associated with the state variation.
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/numdiff/contact.hxx"

#endif  // CROCODDYL_MULTIBODY_NUMDIFF_CONTACT_HPP_
