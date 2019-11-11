///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_SMOOTH_ABS_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_SMOOTH_ABS_HPP_

#include "crocoddyl/core/activation-base.hpp"

namespace crocoddyl {

class ActivationModelSmoothAbs : public ActivationModelAbstract {
 public:
  explicit ActivationModelSmoothAbs(const std::size_t& nr);
  ~ActivationModelSmoothAbs();

  void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r);
  void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r,
                const bool& recalc = true);
  boost::shared_ptr<ActivationDataAbstract> createData();
};

struct ActivationDataSmoothAbs : public ActivationDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Activation>
  explicit ActivationDataSmoothAbs(Activation* const activation)
      : ActivationDataAbstract(activation), a(Eigen::VectorXd::Zero(activation->get_nr())) {
    Arr = 2 * Eigen::MatrixXd::Identity(activation->get_nr(), activation->get_nr());
  }

  Eigen::VectorXd a;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_SMOOTH_ABS_HPP_
