///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIVATION_BASE_HPP_
#define CROCODDYL_MULTIBODY_ACTIVATION_BASE_HPP_

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace crocoddyl {

struct ActivationDataAbstract;  // forward declaration

class ActivationModelAbstract {
 public:
  ActivationModelAbstract();
  virtual ~ActivationModelAbstract();

  virtual void calc(boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r) = 0;
  virtual void calcDiff(boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r,
                        const bool& recalc = true) = 0;
  virtual boost::shared_ptr<ActivationDataAbstract> createData() = 0;

 protected:
  unsigned int ncost_;

#ifdef PYTHON_BINDINGS
 public:
  void calc_wrap(boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::VectorXd& r) { calc(data, r); }

  void calcDiff_wrap(boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::VectorXd& r, const bool& recalc) {
    calcDiff(data, r, recalc);
  }
  void calcDiff_wrap(boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::VectorXd& r) {
    calcDiff(data, r, true);
  }
#endif
};

struct ActivationDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ActivationDataAbstract(Model* const model) {}
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTIVATION_BASE_HPP_
