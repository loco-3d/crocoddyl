///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONSTRAINTS_CONSTRAINT_MANAGER_HPP_
#define CROCODDYL_CORE_CONSTRAINTS_CONSTRAINT_MANAGER_HPP_

#include "crocoddyl/core/constraint-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
struct ConstraintItemTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ConstraintModelAbstractTpl<Scalar> ConstraintModelAbstract;

  ConstraintItemTpl() {}
  ConstraintItemTpl(const std::string& name,
                    std::shared_ptr<ConstraintModelAbstract> constraint,
                    bool active = true)
      : name(name), constraint(constraint), active(active) {}

  template <typename NewScalar>
  ConstraintItemTpl<NewScalar> cast() const {
    typedef ConstraintItemTpl<NewScalar> ReturnType;
    ReturnType ret(name, constraint->template cast<NewScalar>(), active);
    return ret;
  }

  /**
   * @brief Print information on the constraint item
   */
  friend std::ostream& operator<<(std::ostream& os,
                                  const ConstraintItemTpl<Scalar>& model) {
    os << "{" << *model.constraint << "}";
    return os;
  }

  std::string name;
  std::shared_ptr<ConstraintModelAbstract> constraint;
  bool active;
};

/**
 * @brief Manage the individual constraint models
 *
 * This class serves to manage a set of added constraint models. The constraint
 * functions might active or inactive, with this approach we avoid dynamic
 * allocation of memory. Each constraint model is added through `addConstraint`,
 * where its status can be defined.
 *
 * The main computations are carring out in `calc` and `calcDiff` routines.
 * `calc` computes the constraint residuals and `calcDiff` computes the
 * Jacobians of the constraint functions. Concretely speaking, `calcDiff` builds
 * a linear approximation of the total constraint function (both inequality and
 * equality) with the form: \f$\mathbf{g_u}\in\mathbb{R}^{ng\times nu}\f$,
 * \f$\mathbf{h_x}\in\mathbb{R}^{nh\times ndx}\f$
 * \f$\mathbf{h_u}\in\mathbb{R}^{nh\times nu}\f$.
 *
 * \sa `StateAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ConstraintModelManagerTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef ConstraintModelAbstractTpl<Scalar> ConstraintModelAbstract;
  typedef ConstraintDataAbstractTpl<Scalar> ConstraintDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ConstraintItemTpl<Scalar> ConstraintItem;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  typedef std::map<std::string, std::shared_ptr<ConstraintItem> >
      ConstraintModelContainer;
  typedef std::map<std::string, std::shared_ptr<ConstraintDataAbstract> >
      ConstraintDataContainer;

  /**
   * @brief Initialize the constraint-manager model
   *
   * @param[in] state  State of the multibody system
   * @param[in] nu     Dimension of control vector
   */
  ConstraintModelManagerTpl(std::shared_ptr<StateAbstract> state,
                            const std::size_t nu);

  /**
   * @brief Initialize the constraint-manager model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   */
  explicit ConstraintModelManagerTpl(std::shared_ptr<StateAbstract> state);
  ~ConstraintModelManagerTpl();

  /**
   * @brief Add a constraint item
   *
   * @param[in] name        Constraint name
   * @param[in] constraint  Constraint model
   * @param[in] weight      Constraint weight
   * @param[in] active      True if the constraint is activated (default true)
   */
  void addConstraint(const std::string& name,
                     std::shared_ptr<ConstraintModelAbstract> constraint,
                     const bool active = true);

  /**
   * @brief Remove a constraint item
   *
   * @param[in] name  Constraint name
   */
  void removeConstraint(const std::string& name);

  /**
   * @brief Change the constraint status
   *
   * @param[in] name    Constraint name
   * @param[in] active  Constraint status (true for active and false for
   * inactive)
   */
  void changeConstraintStatus(const std::string& name, bool active);

  /**
   * @brief Compute the total constraint value
   *
   * @param[in] data  Constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  void calc(const std::shared_ptr<ConstraintDataManager>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the total constraint value for nodes that depends only on
   * the state
   *
   * It updates the constraint based on the state only. This function is
   * commonly used in the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  void calc(const std::shared_ptr<ConstraintDataManager>& data,
            const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the Jacobian of the total constraint
   *
   * @param[in] data  Constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  void calcDiff(const std::shared_ptr<ConstraintDataManager>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the Jacobian of the total constraint with respect to the
   * state only
   *
   * It computes the Jacobian of the constraint function based on the state
   * only. This function is commonly used in the terminal nodes of an optimal
   * control problem.
   *
   * @param[in] data  Constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  void calcDiff(const std::shared_ptr<ConstraintDataManager>& data,
                const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the constraint data
   *
   * The default data contains objects to store the values of the constraint and
   * their derivatives (i.e. Jacobians). However, it is possible to specialize
   * this function is we need to create additional data, for instance, to avoid
   * dynamic memory allocation.
   *
   * @param data  Data collector
   * @return the constraint data
   */
  std::shared_ptr<ConstraintDataManager> createData(
      DataCollectorAbstract* const data);

  /**
   * @brief Cast the constraint-manager model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ConstraintModelManagerTpl<NewScalar> A constraint-manager model
   * with the new scalar type.
   */
  template <typename NewScalar>
  ConstraintModelManagerTpl<NewScalar> cast() const;

  /**
   * @brief Return the state
   */
  const std::shared_ptr<StateAbstract>& get_state() const;

  /**
   * @brief Return the stack of constraint models
   */
  const ConstraintModelContainer& get_constraints() const;

  /**
   * @brief Return the dimension of the control input
   */
  std::size_t get_nu() const;

  /**
   * @brief Return the number of active inequality constraints
   */
  std::size_t get_ng() const;

  /**
   * @brief Return the number of active equality constraints
   */
  std::size_t get_nh() const;

  /**
   * @brief Return the number of active inequality terminal constraints
   */
  std::size_t get_ng_T() const;

  /**
   * @brief Return the number of active equality terminal constraints
   */
  std::size_t get_nh_T() const;

  /**
   * @brief Return the names of the set of active constraints
   */
  const std::set<std::string>& get_active_set() const;

  /**
   * @brief Return the names of the set of inactive constraints
   */
  const std::set<std::string>& get_inactive_set() const;

  /**
   * @brief Return the state lower bound
   */
  const VectorXs& get_lb() const;

  /**
   * @brief Return the state upper bound
   */
  const VectorXs& get_ub() const;

  /**
   * @brief Return the status of a given constraint name
   *
   * @param[in] name  Constraint name
   */
  bool getConstraintStatus(const std::string& name) const;

  /**
   * @brief Print information on the stack of constraints
   */
  template <class Scalar>
  friend std::ostream& operator<<(
      std::ostream& os, const ConstraintModelManagerTpl<Scalar>& model);

 private:
  std::shared_ptr<StateAbstract> state_;  //!< State description
  ConstraintModelContainer constraints_;  //!< Stack of constraint items
  VectorXs lb_;                           //!< Lower bound of the constraint
  VectorXs ub_;                           //!< Upper bound of the constraint
  std::size_t nu_;                        //!< Dimension of the control input
  std::size_t ng_;    //!< Number of the active inequality constraints
  std::size_t nh_;    //!< Number of the active equality constraints
  std::size_t ng_T_;  //!< Number of the active inequality terminal constraints
  std::size_t nh_T_;  //!< Number of the active equality terminal constraints
  std::set<std::string> active_set_;  //!< Names of the active constraint items
  std::set<std::string>
      inactive_set_;  //!< Names of the inactive constraint items
};

template <typename _Scalar>
struct ConstraintDataManagerTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ConstraintItemTpl<Scalar> ConstraintItem;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  ConstraintDataManagerTpl(Model<Scalar>* const model,
                           DataCollectorAbstract* const data)
      : g_internal(model->get_ng()),
        Gx_internal(model->get_ng(), model->get_state()->get_ndx()),
        Gu_internal(model->get_ng(), model->get_nu()),
        h_internal(model->get_nh()),
        Hx_internal(model->get_nh(), model->get_state()->get_ndx()),
        Hu_internal(model->get_nh(), model->get_nu()),
        shared(data),
        g(g_internal.data(), model->get_ng()),
        Gx(Gx_internal.data(), model->get_ng(), model->get_state()->get_ndx()),
        Gu(Gu_internal.data(), model->get_ng(), model->get_nu()),
        h(h_internal.data(), model->get_nh()),
        Hx(Hx_internal.data(), model->get_nh(), model->get_state()->get_ndx()),
        Hu(Hu_internal.data(), model->get_nh(), model->get_nu()) {
    g.setZero();
    Gx.setZero();
    Gu.setZero();
    h.setZero();
    Hx.setZero();
    Hu.setZero();
    for (typename ConstraintModelManagerTpl<
             Scalar>::ConstraintModelContainer::const_iterator it =
             model->get_constraints().begin();
         it != model->get_constraints().end(); ++it) {
      const std::shared_ptr<ConstraintItem>& item = it->second;
      constraints.insert(
          std::make_pair(item->name, item->constraint->createData(data)));
    }
  }

  template <class ActionData>
  void shareMemory(ActionData* const data) {
    // Save memory by setting the internal variables with null dimension
    g_internal.resize(0);
    Gx_internal.resize(0, 0);
    Gu_internal.resize(0, 0);
    h_internal.resize(0);
    Hx_internal.resize(0, 0);
    Hu_internal.resize(0, 0);
    // Share memory with the differential action data
    new (&g) Eigen::Map<VectorXs>(data->g.data(), data->g.size());
    new (&Gx)
        Eigen::Map<MatrixXs>(data->Gx.data(), data->Gx.rows(), data->Gx.cols());
    new (&Gu)
        Eigen::Map<MatrixXs>(data->Gu.data(), data->Gu.rows(), data->Gu.cols());
    new (&h) Eigen::Map<VectorXs>(data->h.data(), data->h.size());
    new (&Hx)
        Eigen::Map<MatrixXs>(data->Hx.data(), data->Hx.rows(), data->Hx.cols());
    new (&Hu)
        Eigen::Map<MatrixXs>(data->Hu.data(), data->Hu.rows(), data->Hu.cols());
  }

  template <class Model>
  void resize(Model* const model, const bool running_node = true) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nu = model->get_nu();
    const std::size_t ng = running_node ? model->get_ng() : model->get_ng_T();
    const std::size_t nh = running_node ? model->get_nh() : model->get_nh_T();
    new (&g) Eigen::Map<VectorXs>(g_internal.data(), ng);
    new (&Gx) Eigen::Map<MatrixXs>(Gx_internal.data(), ng, ndx);
    new (&Gu) Eigen::Map<MatrixXs>(Gu_internal.data(), ng, nu);
    new (&h) Eigen::Map<VectorXs>(h_internal.data(), nh);
    new (&Hx) Eigen::Map<MatrixXs>(Hx_internal.data(), nh, ndx);
    new (&Hu) Eigen::Map<MatrixXs>(Hu_internal.data(), nh, nu);
  }

  template <class ActionModel, class ActionData>
  void resize(ActionModel* const model, ActionData* const data,
              const bool running_node = true) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nu = model->get_nu();
    const std::size_t ng = running_node ? model->get_ng() : model->get_ng_T();
    const std::size_t nh = running_node ? model->get_nh() : model->get_nh_T();
    data->g.conservativeResize(ng);
    data->Gx.conservativeResize(ng, ndx);
    data->Gu.conservativeResize(ng, nu);
    data->h.conservativeResize(nh);
    data->Hx.conservativeResize(nh, ndx);
    data->Hu.conservativeResize(nh, nu);
    new (&g) Eigen::Map<VectorXs>(data->g.data(), ng);
    new (&Gx) Eigen::Map<MatrixXs>(data->Gx.data(), ng, ndx);
    new (&Gu) Eigen::Map<MatrixXs>(data->Gu.data(), ng, nu);
    new (&h) Eigen::Map<VectorXs>(data->h.data(), nh);
    new (&Hx) Eigen::Map<MatrixXs>(data->Hx.data(), nh, ndx);
    new (&Hu) Eigen::Map<MatrixXs>(data->Hu.data(), nh, nu);
  }

  VectorXs get_g() const { return g; }
  MatrixXs get_Gx() const { return Gx; }
  MatrixXs get_Gu() const { return Gu; }
  VectorXs get_h() const { return h; }
  MatrixXs get_Hx() const { return Hx; }
  MatrixXs get_Hu() const { return Hu; }

  void set_g(const VectorXs& _g) {
    if (g.size() != _g.size()) {
      throw_pretty(
          "Invalid argument: " << "g has wrong dimension (it should be " +
                                      std::to_string(g.size()) + ")");
    }
    g = _g;
  }
  void set_Gx(const MatrixXs& _Gx) {
    if (Gx.rows() != _Gx.rows() || Gx.cols() != _Gx.cols()) {
      throw_pretty(
          "Invalid argument: " << "Gx has wrong dimension (it should be " +
                                      std::to_string(Gx.rows()) + ", " +
                                      std::to_string(Gx.cols()) + ")");
    }
    Gx = _Gx;
  }
  void set_Gu(const MatrixXs& _Gu) {
    if (Gu.rows() != _Gu.rows() || Gu.cols() != _Gu.cols()) {
      throw_pretty(
          "Invalid argument: " << "Gu has wrong dimension (it should be " +
                                      std::to_string(Gu.rows()) + ", " +
                                      std::to_string(Gu.cols()) + ")");
    }
    Gu = _Gu;
  }
  void set_h(const VectorXs& _h) {
    if (h.size() != _h.size()) {
      throw_pretty(
          "Invalid argument: " << "h has wrong dimension (it should be " +
                                      std::to_string(h.size()) + ")");
    }
    h = _h;
  }
  void set_Hx(const MatrixXs& _Hx) {
    if (Hx.rows() != _Hx.rows() || Hx.cols() != _Hx.cols()) {
      throw_pretty(
          "Invalid argument: " << "Hx has wrong dimension (it should be " +
                                      std::to_string(Hx.rows()) + ", " +
                                      std::to_string(Hx.cols()) + ")");
    }
    Hx = _Hx;
  }
  void set_Hu(const MatrixXs& _Hu) {
    if (Hu.rows() != _Hu.rows() || Hu.cols() != _Hu.cols()) {
      throw_pretty(
          "Invalid argument: " << "Hu has wrong dimension (it should be " +
                                      std::to_string(Hu.rows()) + ", " +
                                      std::to_string(Hu.cols()) + ")");
    }
    Hu = _Hu;
  }

  // Creates internal data in case we don't share it externally
  VectorXs g_internal;
  MatrixXs Gx_internal;
  MatrixXs Gu_internal;
  VectorXs h_internal;
  MatrixXs Hx_internal;
  MatrixXs Hu_internal;

  typename ConstraintModelManagerTpl<Scalar>::ConstraintDataContainer
      constraints;
  DataCollectorAbstract* shared;
  Eigen::Map<VectorXs> g;
  Eigen::Map<MatrixXs> Gx;
  Eigen::Map<MatrixXs> Gu;
  Eigen::Map<VectorXs> h;
  Eigen::Map<MatrixXs> Hx;
  Eigen::Map<MatrixXs> Hu;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/constraints/constraint-manager.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ConstraintItemTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ConstraintModelManagerTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ConstraintDataManagerTpl)

#endif  // CROCODDYL_CORE_CONSTRAINTS_CONSTRAINT_MANAGER_HPP_
