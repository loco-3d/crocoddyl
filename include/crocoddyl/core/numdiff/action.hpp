///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellshaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_ACTION_HPP_
#define CROCODDYL_CORE_NUMDIFF_ACTION_HPP_

#include "crocoddyl/core/action-base.hpp"
#include <vector>
#include <iostream>

namespace crocoddyl {

class ActionModelNumDiff : public ActionModelAbstract {
 public:
  explicit ActionModelNumDiff(ActionModelAbstract& model, bool with_gauss_approx = false);
  ~ActionModelNumDiff();

  void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<ActionDataAbstract> createData();

  ActionModelAbstract& get_model() const { return model_; }

  const double& get_disturbance() const { return disturbance_; }
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

  ActionModelAbstract& model_;
  bool with_gauss_approx_;
  double disturbance_;
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
  explicit ActionDataNumDiff(Model* const model)
      : ActionDataAbstract(model),
        Rx(model->get_model().get_nr(), model->get_model().get_state().get_ndx()),
        Ru(model->get_model().get_nr(), model->get_model().get_nu()),
        dx(model->get_model().get_state().get_ndx()),
        du(model->get_model().get_nu()),
        xp(model->get_model().get_state().get_nx()) {
    Rx.setZero();
    Ru.setZero();
    dx.setZero();
    du.setZero();
    xp.setZero();

    const unsigned& ndx = model->get_model().get_state().get_ndx();
    const unsigned& nu = model->get_model().get_nu();
    data_0 = model->get_model().createData();
    for (unsigned i = 0; i < ndx; ++i) {
      data_x.push_back(model->get_model().createData());
    }
    for (unsigned i = 0; i < nu; ++i) {
      data_u.push_back(model->get_model().createData());
    }
  }

  Eigen::MatrixXd Rx;
  Eigen::MatrixXd Ru;
  Eigen::VectorXd dx;
  Eigen::VectorXd du;
  Eigen::VectorXd xp;
  boost::shared_ptr<ActionDataAbstract> data_0;
  std::vector<boost::shared_ptr<ActionDataAbstract> > data_x;
  std::vector<boost::shared_ptr<ActionDataAbstract> > data_u;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_NUMDIFF_ACTION_HPP_
