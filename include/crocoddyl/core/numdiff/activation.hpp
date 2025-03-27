///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_ACTIVATION_HPP_
#define CROCODDYL_CORE_NUMDIFF_ACTIVATION_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActivationModelNumDiffTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActivationModelBase, ActivationModelNumDiffTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataNumDiffTpl<Scalar> Data;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Construct a new ActivationModelNumDiff object
   *
   * @param model
   */
  explicit ActivationModelNumDiffTpl(std::shared_ptr<Base> model);

  /**
   * @brief Destroy the ActivationModelNumDiff object
   */
  virtual ~ActivationModelNumDiffTpl();

  /**
   * @brief @copydoc Base::calc()
   */
  virtual void calc(const std::shared_ptr<ActivationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& r) override;

  /**
   * @brief @copydoc Base::calcDiff()
   */
  virtual void calcDiff(const std::shared_ptr<ActivationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& r) override;

  /**
   * @brief Create a Data object from the given model.
   *
   * @return std::shared_ptr<ActivationDataAbstract>
   */
  virtual std::shared_ptr<ActivationDataAbstract> createData() override;

  template <typename NewScalar>
  ActivationModelNumDiffTpl<NewScalar> cast() const;

  /**
   * @brief Get the model_ object
   *
   * @return Base&
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

 private:
  std::shared_ptr<Base>
      model_;     //!< model to compute the finite differentiation from
  Scalar e_jac_;  //!< Constant used for computing disturbances in Jacobian
                  //!< calculation

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
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Construct a new ActivationDataNumDiff object
   *
   * @tparam Model is the type of the ActivationModel.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <template <typename Scalar> class Model>
  explicit ActivationDataNumDiffTpl(Model<Scalar>* const model)
      : Base(model),
        dr(model->get_model()->get_nr()),
        rp(model->get_model()->get_nr()),
        Arr_(Arr.rows(), Arr.cols()) {
    dr.setZero();
    rp.setZero();
    Arr_.setZero();
    data_0 = model->get_model()->createData();
    const std::size_t nr = model->get_model()->get_nr();
    data_rp.clear();
    for (std::size_t i = 0; i < nr; ++i) {
      data_rp.push_back(model->get_model()->createData());
    }

    data_r2p.clear();
    for (std::size_t i = 0; i < 4; ++i) {
      data_r2p.push_back(model->get_model()->createData());
    }
  }

  VectorXs dr;  //!< disturbance: \f$ [\hdot \;\; disturbance \;\; \hdot] \f$
  VectorXs rp;  //!< The input + the disturbance on one DoF "\f$ r^+ = rp = \int
                //!< r + dr \f$"
  std::shared_ptr<Base> data_0;  //!< The data that contains the final results
  std::vector<std::shared_ptr<Base> >
      data_rp;  //!< The temporary data associated with the input variation
  std::vector<std::shared_ptr<Base> >
      data_r2p;  //!< The temporary data associated with the input variation

  MatrixXs Arr_;
  using Base::a_value;
  using Base::Ar;
  using Base::Arr;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/activation.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ActivationModelNumDiffTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ActivationDataNumDiffTpl)

#endif  // CROCODDYL_CORE_NUMDIFF_ACTIVATION_HPP_
