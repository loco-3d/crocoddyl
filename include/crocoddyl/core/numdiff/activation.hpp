///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University, Max Planck Gesellschaft,
// University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_ACTIVATION_HPP_
#define CROCODDYL_CORE_NUMDIFF_ACTIVATION_HPP_

#include <vector>
#include <iostream>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/activation-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActivationModelNumDiffTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Construct a new ActivationModelNumDiff object
   *
   * @param model
   */
  explicit ActivationModelNumDiffTpl(boost::shared_ptr<Base> model);

  /**
   * @brief Destroy the ActivationModelNumDiff object
   */
  ~ActivationModelNumDiffTpl();

  /**
   * @brief @copydoc Base::calc()
   */
  virtual void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r);

  /**
   * @brief @copydoc Base::calcDiff()
   */
  virtual void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r);

  /**
   * @brief Create a Data object from the given model.
   *
   * @return boost::shared_ptr<ActivationDataAbstract>
   */
  virtual boost::shared_ptr<ActivationDataAbstract> createData();

  /**
   * @brief Get the model_ object
   *
   * @return Base&
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
  using Base::nr_;
};

template <typename _Scalar>
struct ActivationDataNumDiffTpl : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef ActivationDataAbstractTpl<Scalar> Base;

  /**
   * @brief Construct a new ActivationDataNumDiff object
   *
   * @tparam Model is the type of the ActivationModel.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <template <typename Scalar> class Model>
  explicit ActivationDataNumDiffTpl(Model<Scalar>* const model)
      : Base(model), dr(model->get_model()->get_nr()), rp(model->get_model()->get_nr()) {
    dr.setZero();
    rp.setZero();
    data_0 = model->get_model()->createData();
    const std::size_t& nr = model->get_model()->get_nr();
    data_rp.clear();
    for (std::size_t i = 0; i < nr; ++i) {
      data_rp.push_back(model->get_model()->createData());
    }

    data_r2p.clear();
    for (std::size_t i = 0; i < 4; ++i) {
      data_r2p.push_back(model->get_model()->createData());
    }
  }

  VectorXs dr;                     //!< disturbance: \f$ [\hdot \;\; disturbance \;\; \hdot] \f$
  VectorXs rp;                     //!< The input + the disturbance on one DoF "\f$ r^+ = rp =  \int r + dr \f$"
  boost::shared_ptr<Base> data_0;  //!< The data that contains the final results
  std::vector<boost::shared_ptr<Base> > data_rp;   //!< The temporary data associated with the input variation
  std::vector<boost::shared_ptr<Base> > data_r2p;  //!< The temporary data associated with the input variation

  using Base::a_value;
  using Base::Ar;
  using Base::Arr;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/activation.hxx"

#endif  // CROCODDYL_CORE_NUMDIFF_ACTIVATION_HPP_
