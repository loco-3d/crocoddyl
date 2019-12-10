///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_NUMDIFF_COST_HPP_
#define CROCODDYL_MULTIBODY_NUMDIFF_COST_HPP_

#include <boost/function.hpp>
#include "crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {

struct CostNumDiffData;  // forward declaration

// Simple renaming that ease the code writing.
typedef boost::function<void(const pinocchio::Model&, pinocchio::Data&, const Eigen::VectorXd&, const Eigen::VectorXd&)> ReevaluationFunction;

class CostNumDiffModel: CostModelAbstract {

 public:
  /**
   * @brief Construct a new CostNumDiffModel object from a CostModelAbstract.
   * 
   * @param model 
   */
  CostNumDiffModel(const boost::shared_ptr<CostModelAbstract>& model);
  
  /**
   * @brief Default destructor of the CostNumDiffModel object
   */
  ~CostNumDiffModel();

  /**
   * @brief @copydoc ActionModelAbstract::calc()
   */
  void calc(const boost::shared_ptr<CostNumDiffData>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                    const Eigen::Ref<const Eigen::VectorXd>& u);

  /**
   * @brief @copydoc ActionModelAbstract::calcDiff()
   */
  void calcDiff(const boost::shared_ptr<CostNumDiffData>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                        const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);

  /**
   * @brief Create a Data object
   * 
   * @param data is the DataCollector used by the original model.
   * @return boost::shared_ptr<CostModelAbstract> 
   */
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Get the model_ object
   *
   * @return CostModelAbstract&
   */
  const boost::shared_ptr<CostModelAbstract>& get_model() const;

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

  /**
   * @brief Identify if the Gauss approximation is going to be used or not.
   *
   * @return true
   * @return false
   */
  bool get_with_gauss_approx();

  /**
   * @brief Register functions that take a pinocchio model, a pinocchio
   * data, a state and a control. These function are called during the
   * evaluation of the gradient and hessian.
   * 
   * @param reevals are the registered functions.
   */
  void set_reevals(const std::vector<ReevaluationFunction>& reevals);

 protected:
  boost::shared_ptr<CostModelAbstract> model_; /*< Model of the cost. */
  double disturbance_; /*< Numerical disturbance used in the numerical differentiation. */
  std::vector<ReevaluationFunction> reevals_; /*< Functions that needs execution before calc or calcDiff. */
};

struct CostNumDiffData : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  explicit CostNumDiffData(Model* const model, DataCollectorAbstract* const data)
      : CostDataAbstract(model, data),
        dx(model->get_model()->get_state()->get_ndx()),
        du(model->get_model()->get_nu()),
        xp(model->get_model()->get_state()->get_nx()) {
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

  virtual ~CostNumDiffData() {}
  Eigen::VectorXd dx;  //!< State disturbance.
  Eigen::VectorXd du;  //!< Control disturbance.
  Eigen::VectorXd xp;  //!< The integrated state from the disturbance on one DoF "\f$ \int x dx_i \f$".
  boost::shared_ptr<CostDataAbstract> data_0;  //!< The data at the approximation point.
  std::vector<boost::shared_ptr<CostDataAbstract> >
      data_x;  //!< The temporary data associated with the state variation.
  std::vector<boost::shared_ptr<CostDataAbstract> >
      data_u;  //!< The temporary data associated with the control variation.
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_NUMDIFF_COST_HPP_
