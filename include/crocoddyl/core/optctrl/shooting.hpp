///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_
#define CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_

#include <stdexcept>
#include <vector>
#include <boost/circular_buffer.hpp>
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/utils/to-string.hpp"

namespace crocoddyl {

/**
 * @brief This class encapsulates a shooting problem
 *
 * A shooting problem encapsulates the initial state \f$\mathbf{x}_{0}\in\mathcal{M}\f$, a set of running action models
 * and a terminal action model for a discretized trajectory into \f$T\f$ nodes. It has three main methods - `calc`,
 * `calcDiff` and `rollout`. The first computes the set of next states and cost values per each node \f$k\f$. Instead,
 * `calcDiff` updates the derivatives of all action models. Finally, `rollout` integrates the system dynamics. This
 * class is used to decouple problem formulation and resolution.
 */
template <typename _Scalar>
class ShootingProblemTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActionModelAbstractTpl<Scalar> ActionModelAbstract;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the shooting problem and allocate its data
   *
   * @param[in] x0              Initial state
   * @param[in] running_models  Running action models (size \f$T\f$)
   * @param[in] terminal_model  Terminal action model
   */
  ShootingProblemTpl(const VectorXs& x0,
                     const boost::circular_buffer<boost::shared_ptr<ActionModelAbstract> >& running_models,
                     boost::shared_ptr<ActionModelAbstract> terminal_model);

  /**
   * @brief Initialize the shooting problem (models and datas)
   *
   * @param[in] x0              Initial state
   * @param[in] running_models  Running action models (size \f$T\f$)
   * @param[in] terminal_model  Terminal action model
   * @param[in] running_datas   Running action datas (size \f$T\f$)
   * @param[in] terminal_data   Terminal action data
   */
  ShootingProblemTpl(const VectorXs& x0,
                     const boost::circular_buffer<boost::shared_ptr<ActionModelAbstract> >& running_models,
                     boost::shared_ptr<ActionModelAbstract> terminal_model,
                     const boost::circular_buffer<boost::shared_ptr<ActionDataAbstract> >& running_datas,
                     boost::shared_ptr<ActionDataAbstract> terminal_data);
  /**
   * @brief Initialize the shooting problem
   */
  ShootingProblemTpl(const ShootingProblemTpl<Scalar>& problem);
  ~ShootingProblemTpl();

  /**
   * @brief Compute the cost and the next states
   *
   * For each node \f$k\f$, and along the state \f$\mathbf{x_{s}}\f$ and control \f$\mathbf{u_{s}}\f$ trajectory, it
   * computes the next state \f$\mathbf{x}_{k+1}\f$ and cost \f$l_{k}\f$.
   *
   * @param[in] xs  time-discrete state trajectory \f$\mathbf{x_{s}}\f$ (size \f$T+1\f$)
   * @param[in] us  time-discrete control sequence \f$\mathbf{u_{s}}\f$ (size \f$T\f$)
   * @return The total cost value \f$l_{k}\f$
   */
  Scalar calc(const std::vector<VectorXs>& xs, const std::vector<VectorXs>& us);

  /**
   * @brief Compute the derivatives of the cost and dynamics
   *
   * For each node \f$k\f$, and along the state \f$\mathbf{x_{s}}\f$ and control \f$\mathbf{u_{s}}\f$ trajectory, it
   * computes the derivatives of the cost
   * \f$(\mathbf{l}_{\mathbf{x}}, \mathbf{l}_{\mathbf{u}}, \mathbf{l}_{\mathbf{xx}}, \mathbf{l}_{\mathbf{xu}},
   * \mathbf{l}_{\mathbf{uu}})\f$ and dynamics \f$(\mathbf{f}_{\mathbf{x}}, \mathbf{f}_{\mathbf{u}})\f$.
   *
   * @param[in] xs  time-discrete state trajectory \f$\mathbf{x_{s}}\f$ (size \f$T+1\f$)
   * @param[in] us  time-discrete control sequence \f$\mathbf{u_{s}}\f$ (size \f$T\f$)
   * @return The total cost value \f$l_{k}\f$
   */
  Scalar calcDiff(const std::vector<VectorXs>& xs, const std::vector<VectorXs>& us);

  /**
   * @brief Integrate the dynamics given a control sequence
   *
   * @param[in] xs  time-discrete state trajectory \f$\mathbf{x_{s}}\f$ (size \f$T+1\f$)
   * @param[in] us  time-discrete control sequence \f$\mathbf{u_{s}}\f$ (size \f$T\f$)
   */
  void rollout(const std::vector<VectorXs>& us, std::vector<VectorXs>& xs);

  /**
   * @copybrief rollout
   *
   * @param[in] us  time-discrete control sequence \f$\mathbf{u_{s}}\f$ (size \f$T\f$)
   */
  std::vector<VectorXs> rollout_us(const std::vector<VectorXs>& us);

  void pushBackRunningNode(boost::shared_ptr<ActionModelAbstract> model, boost::shared_ptr<ActionDataAbstract> data);

  /**
   * @brief Update the model and data for a specific node
   *
   * @param[in] i      Node index \f$(0\leq i \leq T+1)\f$
   * @param[in] model  Action model
   * @param[in] data   Action data
   */
  void updateNode(std::size_t i, boost::shared_ptr<ActionModelAbstract> model,
                  boost::shared_ptr<ActionDataAbstract> data);

  /**
   * @brief Update a model and allocated new data for a specific node
   *
   * @param[in] i      Node index \f$(0\leq i \leq T+1)\f$
   * @param[in] model  Action model
   */
  void updateModel(std::size_t i, boost::shared_ptr<ActionModelAbstract> model);

  /**
   * @brief Return the number of running nodes
   */
  const std::size_t& get_T() const;

  /**
   * @brief Return the initial state
   */
  const VectorXs& get_x0() const;

  /**
   * @brief Return the running models
   */
  const boost::circular_buffer<boost::shared_ptr<ActionModelAbstract> >& get_runningModels() const;

  /**
   * @brief Return the terminal model
   */
  const boost::shared_ptr<ActionModelAbstract>& get_terminalModel() const;

  /**
   * @brief Return the running datas
   */
  const boost::circular_buffer<boost::shared_ptr<ActionDataAbstract> >& get_runningDatas() const;

  /**
   * @brief Return the terminal data
   */
  const boost::shared_ptr<ActionDataAbstract>& get_terminalData() const;

  /**
   * @brief Modify the initial state
   */
  void set_x0(const VectorXs& x0_in);

  /**
   * @brief Modify the running models and allocate new data
   */
  void set_runningModels(const boost::circular_buffer<boost::shared_ptr<ActionModelAbstract> >& models);

  /**
   * @brief Modify the terminal model and allocate new data
   */
  void set_terminalModel(boost::shared_ptr<ActionModelAbstract> model);

 protected:
  Scalar cost_;                                                                     //!< Total cost
  std::size_t T_;                                                                   //!< number of running nodes
  VectorXs x0_;                                                                     //!< Initial state
  boost::shared_ptr<ActionModelAbstract> terminal_model_;                           //!< Terminal action model
  boost::shared_ptr<ActionDataAbstract> terminal_data_;                             //!< Terminal action data
  boost::circular_buffer<boost::shared_ptr<ActionModelAbstract> > running_models_;  //!< Running action model
  boost::circular_buffer<boost::shared_ptr<ActionDataAbstract> > running_datas_;    //!< Running action data

 private:
  void allocateData();
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/optctrl/shooting.hxx"

#endif  // CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_
