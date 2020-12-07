///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_ACTUATION_HPP_
#define CROCODDYL_CORE_NUMDIFF_ACTUATION_HPP_

#include <vector>
#include <iostream>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/actuation-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActuationModelNumDiffTpl : public ActuationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationDataNumDiffTpl<Scalar> Data;
  typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Construct a new ActuationModelNumDiff object
   *
   * @param model
   */
  explicit ActuationModelNumDiffTpl(boost::shared_ptr<Base> model);

  /**
   * @brief Destroy the ActuationModelNumDiff object
   */
  virtual ~ActuationModelNumDiffTpl();

  /**
   * @brief @copydoc Base::calc()
   */
  virtual void calc(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc Base::calcDiff()
   */
  virtual void calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create a Data object from the given model.
   *
   * @return boost::shared_ptr<ActuationDataAbstract>
   */
  virtual boost::shared_ptr<ActuationDataAbstract> createData();

  /**
   * @brief Get the model_ object
   *
   * @return Base&
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

 private:
  /**
   * @brief This is the model to compute the finite differentiation from
   */
  boost::shared_ptr<Base> model_;

  /**
   * @brief This is the numerical disturbance value used during the numerical
   * differentiation
   */
  Scalar disturbance_;

 protected:
  using Base::nu_;
};

template <typename _Scalar>
struct ActuationDataNumDiffTpl : public ActuationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef ActuationDataAbstractTpl<Scalar> Base;

  /**
   * @brief Construct a new ActuationDataNumDiff object
   *
   * @tparam Model is the type of the ActuationModel.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <template <typename Scalar> class Model>
  explicit ActuationDataNumDiffTpl(Model<Scalar>* const model)
      : Base(model),
        dx(model->get_model()->get_state()->get_ndx()),
        du(model->get_model()->get_nu()),
        xp(model->get_model()->get_state()->get_nx()) {
    dx.setZero();
    du.setZero();
    xp.setZero();
    std::size_t ndx = model->get_model()->get_state()->get_ndx();
    std::size_t nu = model->get_model()->get_nu();
    data_0 = model->get_model()->createData();
    for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(model->get_model()->createData());
    }
    for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(model->get_model()->createData());
    }
  }

  VectorXs dx;                     //!< State disturbance
  VectorXs du;                     //!< Control disturbance
  VectorXs xp;                     //!< The integrated state from the disturbance on one DoF "\f$ \int x dx_i \f$"
  boost::shared_ptr<Base> data_0;  //!< The data that contains the final results
  std::vector<boost::shared_ptr<Base> > data_x;  //!< The temporary data associated with the state variation
  std::vector<boost::shared_ptr<Base> > data_u;  //!< The temporary data associated with the control variation

  using Base::dtau_du;
  using Base::dtau_dx;
  using Base::tau;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/actuation.hxx"

#endif  // CROCODDYL_CORE_NUMDIFF_ACTUATION_HPP_
