///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIONS_ACTION_NUM_DIFF_HPP_
#define CROCODDYL_CORE_ACTIONS_ACTION_NUM_DIFF_HPP_

#include <crocoddyl/core/action-base.hpp>
#include <vector>
#include <iostream>

namespace crocoddyl {

class ActionModelNumDiff : public ActionModelAbstract {
 public:
  ActionModelNumDiff(ActionModelAbstract& model, bool with_gauss_approx = false);
  ~ActionModelNumDiff();

  void calc(std::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) override;
  void calcDiff(std::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true) override;
  std::shared_ptr<ActionDataAbstract> createData() override;

  ActionModelAbstract& get_model() { return model_; }

  /**
   * @brief Get the disturbance_ object
   *
   * @return double
   */
  double get_disturbance() { return disturbance_; }
  /**
   * @brief Get the with_gauss_approx_ object
   *
   * @return true
   * @return false
   */
  bool get_with_gauss_approx() { return with_gauss_approx_; }

 private:
  /**
   * @brief Make sure that when we finite difference the Action Model, the user
   * does not face unknown behaviour because of the finite differencing of a
   * quaternion around pi. This behaviour might occur if CostModelState and
   * FloatingInContact differential model are used together.
   *
   * For full discussions see issue
   * https://gepgitlab.laas.fr/loco-3d/crocoddyl/issues/139
   *
   * @param model object to be checked.
   * @param x is the state at which the check is performed.
   */
  void assertStableStateFD(const Eigen::Ref<const Eigen::VectorXd>& x);

  /**
   * @brief Defines id we use a Gauss approximation of the cost or not.
   */
  bool with_gauss_approx_;
  /**
   * @brief This the increment used in the finite differentation and integration.
   */
  double disturbance_;
  /**
   * @brief This is the vector containing the small element during the finite
   * differentiation and integration. This is a temporary variable but used
   * quiet often. For sake of memory management we allocate it once in the
   * constructor of this class.
   */
  Eigen::VectorXd dx_;
  /**
   * @brief This is the vector containing the small element during the finite
   * differentiation and integration. This is a temporary variable but used
   * quiet often. For sake of memory management we allocate it once in the
   * constructor of this class.
   */
  Eigen::VectorXd du_;
  /**
   * @brief This is the vector containing the result of an integration. This is
   * a temporary variable but used quiet often. For sake of memory management we
   * allocate it once in the constructor of this class.
   */
  Eigen::VectorXd tmp_x_;
  /**
   * @brief This is the model from which to compute the finite differentiation
   * from.
   */
  ActionModelAbstract& model_;
};

struct ActionDataNumDiff : public ActionDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * @brief Construct a new ActionDataNumDiff object
   *
   * @tparam Model is the type of the ActionModel.
   * @param model is the object to compute the numerical differentiation from.
   */
  template <typename Model>
  ActionDataNumDiff(Model* const model_num_diff) : ActionDataAbstract(model_num_diff) {
    // simple renaming for conveniency
    const unsigned& ndx = model_num_diff->get_model().get_ndx();
    const unsigned& nu = model_num_diff->get_model().get_nu();
    const unsigned& ncost = model_num_diff->get_model().get_ncost();
    // data_0_
    data_0 = model_num_diff->get_model().createData();
    // data_x_
    for (unsigned i = 0; i < ndx; ++i) {
      data_x.push_back(model_num_diff->get_model().createData());
    }
    // data_u_
    for (unsigned i = 0; i < nu; ++i) {
      data_u.push_back(model_num_diff->get_model().createData());
    }
    // Rx
    Rx.resize(ncost, ndx);
    Rx.setZero();
    // Ru
    Ru.resize(ncost, nu);
    Ru.setZero();
  }

  /**
   * @brief Destroy the ActionDataNumDiff object
   */
  ~ActionDataNumDiff() {}

  /**
   * @brief @todo write the doc
   */
  Eigen::MatrixXd Rx;
  /**
   * @brief @todo write the doc
   */
  Eigen::MatrixXd Ru;
  /**
   * @brief One set of data used to compute the state 0.
   */
  std::shared_ptr<ActionDataAbstract> data_0;
  /**
   * @brief The data to compute the derivation around the state x.
   */
  std::vector<std::shared_ptr<ActionDataAbstract>> data_x;
  /**
   * @brief The data to compute the derivation around the state u.
   */
  std::vector<std::shared_ptr<ActionDataAbstract>> data_u;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIONS_NUM_DIFF_HPP_
