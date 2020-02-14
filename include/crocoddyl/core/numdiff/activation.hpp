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
#include "crocoddyl/core/activation-base.hpp"

namespace crocoddyl {

class ActivationModelNumDiff : public ActivationModelAbstract {
 public:
  /**
   * @brief Construct a new ActivationModelNumDiff object
   *
   * @param model
   */
  explicit ActivationModelNumDiff(boost::shared_ptr<ActivationModelAbstract> model);

  /**
   * @brief Destroy the ActivationModelNumDiff object
   */
  ~ActivationModelNumDiff();

  /**
   * @brief @copydoc ActivationModelAbstract::calc()
   */
  void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r);

  /**
   * @brief @copydoc ActivationModelAbstract::calcDiff()
   */
  void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r);

  /**
   * @brief Create a Data object from the given model.
   *
   * @return boost::shared_ptr<ActivationDataAbstract>
   */
  boost::shared_ptr<ActivationDataAbstract> createData();

  /**
   * @brief Get the model_ object
   *
   * @return ActivationModelAbstract&
   */
  const boost::shared_ptr<ActivationModelAbstract>& get_model() const;

  /**
   * @brief Get the disturbance_ object
   *
   * @return const double&
   */
  const double& get_disturbance() const;

  /**
   * @brief Set the disturbance_ object
   *
   * @param disturbance is the value used to find the numerical derivative
   */
  void set_disturbance(const double& disturbance);

 private:
  /**
   * @brief This is the model to compute the finite differentiation from
   */
  boost::shared_ptr<ActivationModelAbstract> model_;

  /**
   * @brief This is the numerical disturbance value used during the numerical
   * differentiation
   */
  double disturbance_;
};

struct ActivationDataNumDiff : public ActivationDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * @brief Construct a new ActivationDataNumDiff object
   *
   * @tparam Model is the type of the ActivationModel.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <typename Model>
  explicit ActivationDataNumDiff(Model* const model)
      : ActivationDataAbstract(model), dr(model->get_model()->get_nr()), rp(model->get_model()->get_nr()) {
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

  Eigen::VectorXd dr;  //!< disturbance: \f$ [\hdot \;\; disturbance \;\; \hdot] \f$
  Eigen::VectorXd rp;  //!< The input + the disturbance on one DoF "\f$ r^+ = rp =  \int r + dr \f$"
  boost::shared_ptr<ActivationDataAbstract> data_0;  //!< The data that contains the final results
  std::vector<boost::shared_ptr<ActivationDataAbstract> >
      data_rp;  //!< The temporary data associated with the input variation
  std::vector<boost::shared_ptr<ActivationDataAbstract> >
      data_r2p;  //!< The temporary data associated with the input variation
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_NUMDIFF_ACTIVATION_HPP_
