///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_COSTS_COST_SUM_HPP_
#define CROCODDYL_CORE_COSTS_COST_SUM_HPP_

#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
struct CostItemTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef CostModelAbstractTpl<Scalar> CostModelAbstract;

  CostItemTpl() {}
  CostItemTpl(const std::string& name, std::shared_ptr<CostModelAbstract> cost,
              const Scalar weight, const bool active = true)
      : name(name), cost(cost), weight(weight), active(active) {}

  template <typename NewScalar>
  CostItemTpl<NewScalar> cast() const {
    typedef CostItemTpl<NewScalar> ReturnType;
    ReturnType ret(name, cost->template cast<NewScalar>(),
                   scalar_cast<NewScalar>(weight), active);
    return ret;
  }

  /**
   * @brief Print information on the cost item
   */
  friend std::ostream& operator<<(std::ostream& os,
                                  const CostItemTpl<Scalar>& model) {
    os << "{w=" << model.weight << ", " << *model.cost << "}";
    return os;
  }

  std::string name;
  std::shared_ptr<CostModelAbstract> cost;
  Scalar weight;
  bool active;
};

/**
 * @brief Summation of individual cost models
 *
 * This class serves to manage a set of added cost models. The cost functions
 * might active or inactive, with this approach we avoid dynamic allocation of
 * memory. Each cost model is added through `addCost`, where the weight and its
 * status can be defined.
 *
 * The main computations are carring out in `calc()` and `calcDiff()` routines.
 * `calc()` computes the costs (and its residuals) and `calcDiff()` computes the
 * derivatives of the cost functions (and its residuals). Concretely speaking,
 * `calcDiff()` builds a linear-quadratic approximation of the total cost
 * function with the form: \f$\mathbf{\ell_x}\in\mathbb{R}^{ndx}\f$,
 * \f$\mathbf{\ell_u}\in\mathbb{R}^{nu}\f$,
 * \f$\mathbf{\ell_{xx}}\in\mathbb{R}^{ndx\times ndx}\f$,
 * \f$\mathbf{\ell_{xu}}\in\mathbb{R}^{ndx\times nu}\f$,
 * \f$\mathbf{\ell_{uu}}\in\mathbb{R}^{nu\times nu}\f$ are the Jacobians and
 * Hessians, respectively.
 *
 * \sa `CostModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelSumTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataSumTpl<Scalar> CostDataSum;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef CostModelAbstractTpl<Scalar> CostModelAbstract;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef CostItemTpl<Scalar> CostItem;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  typedef std::map<std::string, std::shared_ptr<CostItem> > CostModelContainer;
  typedef std::map<std::string, std::shared_ptr<CostDataAbstract> >
      CostDataContainer;

  /**
   * @brief Initialize the cost-sum model
   *
   * @param[in] state  State description
   * @param[in] nu     Dimension of control vector
   */
  CostModelSumTpl(std::shared_ptr<StateAbstract> state, const std::size_t nu);

  /**
   * @brief Initialize the cost-sum model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State description
   */
  explicit CostModelSumTpl(std::shared_ptr<StateAbstract> state);
  ~CostModelSumTpl();

  /**
   * @brief Add a cost item
   *
   * @param[in] name    Cost name
   * @param[in] cost    Cost model
   * @param[in] weight  Cost weight
   * @param[in] active  True if the cost is activated (default true)
   */
  void addCost(const std::string& name, std::shared_ptr<CostModelAbstract> cost,
               const Scalar weight, const bool active = true);

  /**
   * @brief Add a cost item
   *
   * @param[in] cost_item Cost item
   */
  void addCost(const std::shared_ptr<CostItem>& cost_item);

  /**
   * @brief Remove a cost item
   *
   * @param[in] name    Cost name
   */
  void removeCost(const std::string& name);

  /**
   * @brief Change the cost status
   *
   * @param[in] name    Cost name
   * @param[in] active  Cost status (true for active and false for inactive)
   */
  void changeCostStatus(const std::string& name, const bool active);

  /**
   * @brief Compute the total cost value
   *
   * @param[in] data  Cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  void calc(const std::shared_ptr<CostDataSum>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the total cost value for nodes that depends only on the
   * state
   *
   * It updates the total cost based on the state only. This function is used in
   * the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  void calc(const std::shared_ptr<CostDataSum>& data,
            const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the Jacobian and Hessian of the total cost
   *
   * @param[in] data  Cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  void calcDiff(const std::shared_ptr<CostDataSum>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the Jacobian and Hessian of the total cost for nodes that
   * depends on the state only
   *
   * It updates the Jacobian and Hessian of the total cost based on the state
   * only. This function is used in the terminal nodes of an optimal control
   * problem.
   *
   * @param[in] data  Cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  void calcDiff(const std::shared_ptr<CostDataSum>& data,
                const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the cost data
   *
   * The default data contains objects to store the values of the cost, residual
   * vector and their derivatives (first and second order derivatives). However,
   * it is possible to specialize this function is we need to create additional
   * data, for instance, to avoid dynamic memory allocation.
   *
   * @param data  Data collector
   * @return the cost data
   */
  std::shared_ptr<CostDataSum> createData(DataCollectorAbstract* const data);

  /**
   * @brief Cast the cost-sum model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return CostModelSumTpl<NewScalar> A cost-sum model with the
   * new scalar type.
   */
  template <typename NewScalar>
  CostModelSumTpl<NewScalar> cast() const;

  /**
   * @brief Return the state
   */
  const std::shared_ptr<StateAbstract>& get_state() const;

  /**
   * @brief Return the stack of cost models
   */
  const CostModelContainer& get_costs() const;

  /**
   * @brief Return the dimension of the control input
   */
  std::size_t get_nu() const;

  /**
   * @brief Return the dimension of the active residual vector
   */
  std::size_t get_nr() const;

  /**
   * @brief Return the dimension of the total residual vector
   */
  std::size_t get_nr_total() const;

  /**
   * @brief Return the names of the set of active costs
   */
  const std::set<std::string>& get_active_set() const;

  /**
   * @brief Return the names of the set of inactive costs
   */
  const std::set<std::string>& get_inactive_set() const;

  DEPRECATED(
      "get_active() is deprecated and will be replaced with get_active_set()",
      const std::vector<std::string>& get_active() {
        active_.clear();
        active_.reserve(active_set_.size());
        for (const auto& contact : active_set_) {
          active_.push_back(contact);
        }
        return active_;
      };)

  DEPRECATED(
      "get_inactive() is deprecated and will be replaced with "
      "get_inactive_set()",
      const std::vector<std::string>& get_inactive() {
        inactive_.clear();
        inactive_.reserve(inactive_set_.size());
        for (const auto& contact : inactive_set_) {
          inactive_.push_back(contact);
        }
        return inactive_;
      };)

  /**
   * @brief Return the status of a given cost name
   *
   * @param[in] name  Cost name
   */
  bool getCostStatus(const std::string& name) const;

  /**
   * @brief Print information on the stack of costs
   */
  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os,
                                  const CostModelSumTpl<Scalar>& model);

 private:
  std::shared_ptr<StateAbstract> state_;  //!< State description
  CostModelContainer costs_;              //!< Stack of cost items
  std::size_t nu_;                        //!< Dimension of the control input
  std::size_t nr_;        //!< Dimension of the active residual vector
  std::size_t nr_total_;  //!< Dimension of the total residual vector
  std::set<std::string> active_set_;  //!< Names of the active set of cost items
  std::set<std::string>
      inactive_set_;  //!< Names of the inactive set of cost items

  // Vector variants. These are to maintain the API compatibility for the
  // deprecated syntax. These will be removed in future versions along with
  // get_active() / get_inactive()
  std::vector<std::string> active_;
  std::vector<std::string> inactive_;
};

template <typename _Scalar>
struct CostDataSumTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef CostItemTpl<Scalar> CostItem;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  CostDataSumTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Lx_internal(model->get_state()->get_ndx()),
        Lu_internal(model->get_nu()),
        Lxx_internal(model->get_state()->get_ndx(),
                     model->get_state()->get_ndx()),
        Lxu_internal(model->get_state()->get_ndx(), model->get_nu()),
        Luu_internal(model->get_nu(), model->get_nu()),
        shared(data),
        cost(Scalar(0.)),
        Lx(Lx_internal.data(), model->get_state()->get_ndx()),
        Lu(Lu_internal.data(), model->get_nu()),
        Lxx(Lxx_internal.data(), model->get_state()->get_ndx(),
            model->get_state()->get_ndx()),
        Lxu(Lxu_internal.data(), model->get_state()->get_ndx(),
            model->get_nu()),
        Luu(Luu_internal.data(), model->get_nu(), model->get_nu()) {
    Lx.setZero();
    Lu.setZero();
    Lxx.setZero();
    Lxu.setZero();
    Luu.setZero();
    for (typename CostModelSumTpl<Scalar>::CostModelContainer::const_iterator
             it = model->get_costs().begin();
         it != model->get_costs().end(); ++it) {
      const std::shared_ptr<CostItem>& item = it->second;
      costs.insert(std::make_pair(item->name, item->cost->createData(data)));
    }
  }

  template <class ActionData>
  void shareMemory(ActionData* const data) {
    // Save memory by setting the internal variables with null dimension
    Lx_internal.resize(0);
    Lu_internal.resize(0);
    Lxx_internal.resize(0, 0);
    Lxu_internal.resize(0, 0);
    Luu_internal.resize(0, 0);
    // Share memory with the differential action data
    new (&Lx) Eigen::Map<VectorXs>(data->Lx.data(), data->Lx.size());
    new (&Lu) Eigen::Map<VectorXs>(data->Lu.data(), data->Lu.size());
    new (&Lxx) Eigen::Map<MatrixXs>(data->Lxx.data(), data->Lxx.rows(),
                                    data->Lxx.cols());
    new (&Lxu) Eigen::Map<MatrixXs>(data->Lxu.data(), data->Lxu.rows(),
                                    data->Lxu.cols());
    new (&Luu) Eigen::Map<MatrixXs>(data->Luu.data(), data->Luu.rows(),
                                    data->Luu.cols());
  }

  VectorXs get_Lx() const { return Lx; }
  VectorXs get_Lu() const { return Lu; }
  MatrixXs get_Lxx() const { return Lxx; }
  MatrixXs get_Lxu() const { return Lxu; }
  MatrixXs get_Luu() const { return Luu; }

  void set_Lx(const VectorXs& _Lx) {
    if (Lx.size() != _Lx.size()) {
      throw_pretty(
          "Invalid argument: " << "Lx has wrong dimension (it should be " +
                                      std::to_string(Lx.size()) + ")");
    }
    Lx = _Lx;
  }
  void set_Lu(const VectorXs& _Lu) {
    if (Lu.size() != _Lu.size()) {
      throw_pretty(
          "Invalid argument: " << "Lu has wrong dimension (it should be " +
                                      std::to_string(Lu.size()) + ")");
    }
    Lu = _Lu;
  }
  void set_Lxx(const MatrixXs& _Lxx) {
    if (Lxx.rows() != _Lxx.rows() || Lxx.cols() != _Lxx.cols()) {
      throw_pretty(
          "Invalid argument: " << "Lxx has wrong dimension (it should be " +
                                      std::to_string(Lxx.rows()) + ", " +
                                      std::to_string(Lxx.cols()) + ")");
    }
    Lxx = _Lxx;
  }
  void set_Lxu(const MatrixXs& _Lxu) {
    if (Lxu.rows() != _Lxu.rows() || Lxu.cols() != _Lxu.cols()) {
      throw_pretty(
          "Invalid argument: " << "Lxu has wrong dimension (it should be " +
                                      std::to_string(Lxu.rows()) + ", " +
                                      std::to_string(Lxu.cols()) + ")");
    }
    Lxu = _Lxu;
  }
  void set_Luu(const MatrixXs& _Luu) {
    if (Luu.rows() != _Luu.rows() || Luu.cols() != _Luu.cols()) {
      throw_pretty(
          "Invalid argument: " << "Luu has wrong dimension (it should be " +
                                      std::to_string(Luu.rows()) + ", " +
                                      std::to_string(Luu.cols()) + ")");
    }
    Luu = _Luu;
  }

  // Creates internal data in case we don't share it externally
  VectorXs Lx_internal;
  VectorXs Lu_internal;
  MatrixXs Lxx_internal;
  MatrixXs Lxu_internal;
  MatrixXs Luu_internal;

  typename CostModelSumTpl<Scalar>::CostDataContainer costs;
  DataCollectorAbstract* shared;
  Scalar cost;
  Eigen::Map<VectorXs> Lx;
  Eigen::Map<VectorXs> Lu;
  Eigen::Map<MatrixXs> Lxx;
  Eigen::Map<MatrixXs> Lxu;
  Eigen::Map<MatrixXs> Luu;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/costs/cost-sum.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::CostItemTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::CostModelSumTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::CostDataSumTpl)

#endif  // CROCODDYL_CORE_COSTS_COST_SUM_HPP_
