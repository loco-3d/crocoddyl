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
  explicit DifferentialActionModelNumDiff(boost::shared_ptr<DifferentialActionModelAbstract> model,
                                          bool with_gauss_approx = false);
  ~DifferentialActionModelNumDiff();

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc = true);
  boost::shared_ptr<DifferentialActionDataAbstract> createData();

  const boost::shared_ptr<DifferentialActionModelAbstract>& get_model() const;
  const double& get_disturbance() const;
  void set_disturbance(const double& disturbance);
  bool get_with_gauss_approx();

 private:
  void assertStableStateFD(const Eigen::Ref<const Eigen::VectorXd>& x);

  boost::shared_ptr<DifferentialActionModelAbstract> model_;
  bool with_gauss_approx_;
  double disturbance_;
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
  explicit DifferentialActionDataNumDiff(Model* const model)
      : DifferentialActionDataAbstract(model),
        Rx(model->get_model()->get_nr(), model->get_model()->get_state()->get_ndx()),
        Ru(model->get_model()->get_nr(), model->get_model()->get_nu()),
        dx(model->get_model()->get_state()->get_ndx()),
        du(model->get_model()->get_nu()),
        xp(model->get_model()->get_state()->get_nx()) {
    Rx.setZero();
    Ru.setZero();
    dx.setZero();
    du.setZero();
    xp.setZero();

    const std::size_t& ndx = model->get_model()->get_state()->get_ndx();
    const std::size_t& nu = model->get_model()->get_nu();
    data_0 = model->get_model()->createData();
    for (std::size_t i = 0; i < ndx; ++i) {
      data_x.push_back(model->get_model()->createData());
    }
    for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(model->get_model()->createData());
    }
  }

  Eigen::MatrixXd Rx;
  Eigen::MatrixXd Ru;
  Eigen::VectorXd dx;
  Eigen::VectorXd du;
  Eigen::VectorXd xp;
  boost::shared_ptr<DifferentialActionDataAbstract> data_0;
  std::vector<boost::shared_ptr<DifferentialActionDataAbstract> > data_x;
  std::vector<boost::shared_ptr<DifferentialActionDataAbstract> > data_u;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_NUMDIFF_DIFF_ACTION_HPP_
