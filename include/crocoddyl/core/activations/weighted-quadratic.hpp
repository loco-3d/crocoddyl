///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include <stdexcept>

namespace crocoddyl {

class ActivationModelWeightedQuad : public ActivationModelAbstract {
 public:
  explicit ActivationModelWeightedQuad(const Eigen::VectorXd& weights);
  ~ActivationModelWeightedQuad();

  void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r);
  void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r);
  boost::shared_ptr<ActivationDataAbstract> createData();

  const Eigen::VectorXd& get_weights() const;
  void set_weights(const Eigen::VectorXd& weights);

 private:
  Eigen::VectorXd weights_;

#ifndef NDEBUG
  Eigen::MatrixXd Arr_;
#endif
};

struct ActivationDataWeightedQuad : public ActivationDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Activation>
  explicit ActivationDataWeightedQuad(Activation* const activation)
      : ActivationDataAbstract(activation), Wr(Eigen::VectorXd::Zero(activation->get_nr())) {}

  Eigen::VectorXd Wr;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_HPP_
