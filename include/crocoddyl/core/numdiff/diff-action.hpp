///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_DIFF_ACTION_HPP_
#define CROCODDYL_CORE_NUMDIFF_DIFF_ACTION_HPP_

#include "crocoddyl/core/diff-action-base.hpp"
#include <vector>
#include <iostream>

namespace crocoddyl {

class DifferentialActionModelNumDiff : public DifferentialActionModelAbstract {
 public:
  explicit DifferentialActionModelNumDiff(DifferentialActionModelAbstract& model, bool with_gauss_approx = false);
  ~DifferentialActionModelNumDiff();

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc = true);
  boost::shared_ptr<DifferentialActionDataAbstract> createData();

  DifferentialActionModelAbstract& get_model() const { return model_; }

  double get_disturbance() { return disturbance_; }
  bool get_with_gauss_approx() { return with_gauss_approx_; }

 private:
  void assertStableStateFD(const Eigen::Ref<const Eigen::VectorXd>& x);

  bool with_gauss_approx_;
  double disturbance_;
  Eigen::VectorXd dx_;
  Eigen::VectorXd du_;
  Eigen::VectorXd tmp_x_;
  DifferentialActionModelAbstract& model_;
};

struct DifferentialActionDataNumDiff : public DifferentialActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * @brief Construct a new ActionDataNumDiff object
   *
   * @tparam Model is the type of the ActionModel.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <typename Model>
  explicit DifferentialActionDataNumDiff(Model* const model_num_diff)
      : DifferentialActionDataAbstract(model_num_diff) {
    const unsigned& ndx = model_num_diff->get_model().get_state().get_ndx();
    const unsigned& nu = model_num_diff->get_model().get_nu();
    const unsigned& nr = model_num_diff->get_model().get_nr();
    data_0 = model_num_diff->get_model().createData();
    for (unsigned i = 0; i < ndx; ++i) {
      data_x.push_back(model_num_diff->get_model().createData());
    }
    for (unsigned i = 0; i < nu; ++i) {
      data_u.push_back(model_num_diff->get_model().createData());
    }
    Rx.resize(nr, ndx);
    Ru.resize(nr, nu);
    Rx.setZero();
    Ru.setZero();
  }

  Eigen::MatrixXd Rx;
  Eigen::MatrixXd Ru;
  boost::shared_ptr<DifferentialActionDataAbstract> data_0;
  std::vector<boost::shared_ptr<DifferentialActionDataAbstract> > data_x;
  std::vector<boost::shared_ptr<DifferentialActionDataAbstract> > data_u;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_NUMDIFF_DIFF_ACTION_HPP_
