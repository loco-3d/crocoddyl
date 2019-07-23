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
  ActivationModelAbstract(const unsigned int& nr);
  virtual ~ActivationModelAbstract();

  virtual void calc(boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r) = 0;
  virtual void calcDiff(boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& r,
                        const bool& recalc = true) = 0;
  virtual boost::shared_ptr<ActivationDataAbstract> createData() = 0;

  unsigned int get_nr() const;

 protected:
  unsigned int nr_;

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

  template <typename Activation>
  ActivationDataAbstract(Activation* const activation) : a_norm(0.), Ar(Eigen::VectorXd::Zero(activation->get_nr())), Arr(Eigen::MatrixXd::Zero(activation->get_nr(), activation->get_nr())) {}

  double a_norm;
  Eigen::VectorXd Ar;
  Eigen::MatrixXd Arr;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTIVATION_BASE_HPP_
