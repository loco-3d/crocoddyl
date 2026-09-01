///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_PARAMS_PARAMETER_MANAGER_HPP_
#define CROCODDYL_CORE_PARAMS_PARAMETER_MANAGER_HPP_

#include <limits>
#include <map>
#include <set>

#include "crocoddyl/core/dynamics-base.hpp"
#include "crocoddyl/core/params-base.hpp"

namespace crocoddyl {

namespace internal {
template <typename Scalar>
struct ParameterDataManagerAccessTpl;
}  // namespace internal

/**
 * @brief Named parameter model and its activation state
 *
 * The item shares ownership of its mutable model, but its name, model pointer
 * and status are externally read-only. Activation changes are owned by
 * ParameterManagerTpl so its dimensions and active/inactive sets stay
 * synchronized. Copies preserve the identity of the underlying model.
 */
template <typename _Scalar>
struct ParameterItemTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ParamsAbstractTpl<Scalar> ParamsAbstract;

  /**
   * @brief Initialize a parameter item
   *
   * @param[in] name    Parameter name
   * @param[in] param   Non-null parameter model
   * @param[in] active  Activation status
   *
   * @throw crocoddyl::Exception if `param` is null
   */
  ParameterItemTpl(const std::string& name,
                   std::shared_ptr<ParamsAbstract> param,
                   const bool active = true)
      : name_(name), param_(param), active_(active) {
    if (param_ == nullptr) {
      throw_pretty("Invalid argument: parameter model is null");
    }
  }
  ParameterItemTpl(const ParameterItemTpl&) = default;
  ParameterItemTpl& operator=(const ParameterItemTpl&) = delete;

  /** @return Parameter name */
  const std::string& get_name() const { return name_; }

  /** @return Shared parameter model; the pointer itself is read-only */
  const std::shared_ptr<ParamsAbstract>& get_param() const { return param_; }

  /** @return Activation status managed by ParameterManagerTpl */
  bool get_active() const { return active_; }

  friend std::ostream& operator<<(std::ostream& os,
                                  const ParameterItemTpl<Scalar>& item) {
    os << "{" << *item.param_ << "}";
    return os;
  }

 private:
  friend class ParameterManagerTpl<Scalar>;

  std::string name_;                       //!< Parameter name
  std::shared_ptr<ParamsAbstract> param_;  //!< Shared parameter model
  bool active_;                            //!< Activation status
};

/**
 * @brief Manager of action and dynamics parameter models
 *
 * Active parameters are stacked deterministically: action parameters form the
 * first partition and dynamics parameters the second, with lexicographic name
 * order inside each partition. Inactive items remain owned by the manager but
 * consume no offset. Global offsets refer to the complete vector; action and
 * caller-provided derivative matrices use offsets relative to their own
 * partitions.
 *
 * The manager owns its item bookkeeping and shares each parameter model. A
 * copied manager receives independent items, sets and dimensions while
 * preserving model identity. Scalar casts rebuild the same names, ordering and
 * status using cast parameter models.
 *
 * Data must be created after adding or removing models. Activation changes
 * retain the per-item layout and require `ParameterDataManagerTpl::resize()`.
 * Call `update()` before either derivative operation. Parameter data is shared
 * and read-only during derivative evaluation; derivative matrices are supplied
 * by each node. All data, vector and state/control dimensions are checked and
 * reported with Crocoddyl exceptions.
 */
template <typename _Scalar>
class ParameterManagerTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef ParamsAbstractTpl<Scalar> ParamsAbstract;
  typedef ParamsDataAbstractTpl<Scalar> ParamsDataAbstract;
  typedef ActionModelParamsAbstractTpl<Scalar> ActionModelParamsAbstract;
  typedef DynamicsParamsAbstractTpl<Scalar> DynamicsParamsAbstract;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterItemTpl<Scalar> ParameterItem;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  typedef std::map<std::string, std::shared_ptr<ParameterItem> >
      ParameterContainer;
  typedef std::set<std::string> NameSet;

  /**
   * @brief Initialize an empty parameter manager
   *
   * @param[in] state  Non-null state description
   * @throw crocoddyl::Exception if `state` is null
   */
  explicit ParameterManagerTpl(std::shared_ptr<StateAbstract> state);

  /**
   * @brief Copy the manager with independent item bookkeeping
   */
  ParameterManagerTpl(const ParameterManagerTpl& other);
  ParameterManagerTpl& operator=(const ParameterManagerTpl&) = delete;
  ~ParameterManagerTpl() = default;

  /**
   * @brief Add an action-parameter model
   *
   * Duplicate names leave the manager unchanged and emit the established
   * Crocoddyl warning.
   *
   * @throw crocoddyl::Exception if `param` is null or has an incompatible state
   */
  void addParam(const std::string& name,
                std::shared_ptr<ActionModelParamsAbstract> param,
                const bool active = true);

  /**
   * @brief Add a dynamics-parameter model
   *
   * Duplicate names leave the manager unchanged and emit the established
   * Crocoddyl warning.
   *
   * @throw crocoddyl::Exception if `param` is null or has an incompatible state
   */
  void addParam(const std::string& name,
                std::shared_ptr<DynamicsParamsAbstract> param,
                const bool active = true);

  /** @brief Remove an item, warning without mutation when `name` is absent */
  void removeParam(const std::string& name);

  /** @brief Change an item's status, warning without mutation if absent */
  void changeParamStatus(const std::string& name, bool active);

  /**
   * @brief Return an item's status, warning and returning false if absent
   * @return Activation status, or false when `name` is absent
   */
  bool getParamStatus(const std::string& name) const;

  /**
   * @brief Update the aggregate vector and every active item data
   *
   * @param[in] data  Consistent parameter-manager data
   * @param[in] p     Active vector ordered by action then dynamics partitions
   * @throw crocoddyl::Exception for null/stale data or a wrong vector dimension
   */
  void update(const std::shared_ptr<ParameterDataManager>& data,
              const Eigen::Ref<const VectorXs>& p) const;

  /**
   * @brief Compute every active action sensitivity into node-local storage
   *
   * @param[out] dx_dp  Matrix of size ndx by np_action
   * @throw crocoddyl::Exception for null/stale data or wrong state/control
   * sizes
   */
  void calcDiff_action(const std::shared_ptr<ParameterDataManager>& data,
                       const std::shared_ptr<ActionDataAbstract>& action_data,
                       Eigen::Ref<MatrixXs> dx_dp,
                       const Eigen::Ref<const VectorXs>& x,
                       const Eigen::Ref<const VectorXs>& u) const;

  /**
   * @brief Compute and return every active action sensitivity
   *
   * This convenience function allocates the sensitivity matrix and calls
   * calcDiff_action().
   *
   * @param[in] data         Consistent parameter-manager data
   * @param[in] action_data  Action data owned by the caller
   * @param[in] x            State point
   * @param[in] u            Control point
   * @return Matrix of size ndx by np_action
   */
  MatrixXs calcDiff_action_x(
      const std::shared_ptr<ParameterDataManager>& data,
      const std::shared_ptr<ActionDataAbstract>& action_data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) const;

  /**
   * @brief Compute every active dynamics regressor into node-local storage
   *
   * @param[out] dtau_dp  Matrix of size nv by np_dynamics
   * @throw crocoddyl::Exception for null/stale data or wrong state/control
   * sizes
   */
  void calcDiff_dynamics(
      const std::shared_ptr<ParameterDataManager>& data,
      const std::shared_ptr<DynamicsDataAbstract>& dynamics_data,
      Eigen::Ref<MatrixXs> dtau_dp, const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) const;

  /**
   * @brief Compute and return every active dynamics regressor
   *
   * This convenience function allocates the regressor matrix and calls
   * calcDiff_dynamics().
   *
   * @param[in] data           Consistent parameter-manager data
   * @param[in] dynamics_data  Dynamics data owned by the caller
   * @param[in] x              State point
   * @param[in] u              Control point
   * @return Matrix of size nv by np_dynamics
   */
  MatrixXs calcDiff_dynamics_x(
      const std::shared_ptr<ParameterDataManager>& data,
      const std::shared_ptr<DynamicsDataAbstract>& dynamics_data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) const;

  /**
   * @brief Return the stacked zero vector for active items
   * @return Vector in canonical action/dynamics order
   */
  VectorXs zero() const;

  /**
   * @brief Return the stacked random vector for active items
   * @return Vector in canonical action/dynamics order
   */
  VectorXs rand() const;

  /**
   * @brief Create complete manager data for the current model set and status
   * @return New manager data with a valid non-owning self-link
   * @throw crocoddyl::Exception if an item creates inconsistent data
   */
  std::shared_ptr<ParameterDataManager> createData() const;

  /**
   * @brief Cast the complete nonempty manager to another scalar type
   * @return Manager preserving names, ordering, status, dimensions and models
   * @throw crocoddyl::Exception if an item cannot preserve its family on cast
   */
  template <typename NewScalar>
  ParameterManagerTpl<NewScalar> cast() const;

  /** @return Shared state description */
  const std::shared_ptr<StateAbstract>& get_state() const;

  /** @return Total active parameter dimension */
  std::size_t get_np() const;

  /** @return Active action-parameter dimension */
  std::size_t get_np_action() const;

  /** @return Active dynamics-parameter dimension */
  std::size_t get_np_dynamics() const;

  /** @return Lexicographically ordered action items, active and inactive */
  const ParameterContainer& get_action_params() const;

  /** @return Lexicographically ordered dynamics items, active and inactive */
  const ParameterContainer& get_dynamics_params() const;

  /** @return Lexicographically ordered active names across both partitions */
  const NameSet& get_active_set() const;

  /** @return Lexicographically ordered inactive names across both partitions */
  const NameSet& get_inactive_set() const;

  /** @brief Print information about the parameter manager */
  void print(std::ostream& os) const;

  template <class Scalar>
  friend std::ostream& operator<<(std::ostream& os,
                                  const ParameterManagerTpl<Scalar>& model);

 protected:
  void addParamItem(const std::string& name,
                    std::shared_ptr<ParamsAbstract> param,
                    ParameterContainer* container, const bool active);
  void addToSets(const std::string& name, bool active);
  void updateDimensions(const std::shared_ptr<ParameterItem>& item,
                        const int delta, const bool action);
  void assertDataIsConsistent(
      const std::shared_ptr<ParameterDataManager>& data) const;
  std::size_t assertItemDataIsConsistent(
      const std::string& name, const std::shared_ptr<ParameterItem>& item,
      const std::string& data_name,
      const std::shared_ptr<ParamsDataAbstract>& item_data,
      const bool action) const;

  std::shared_ptr<StateAbstract> state_;
  std::size_t np_;
  std::size_t np_action_;
  std::size_t np_dynamics_;
  ParameterContainer action_params_;
  ParameterContainer dynamics_params_;
  NameSet active_set_;
  NameSet inactive_set_;
};

/**
 * @brief Complete data owned by `ParameterManagerTpl`
 *
 * The aggregate parameter vector owns the action-prefix/dynamics-suffix
 * layout, and both maps own one specialized parameter data object for every
 * model, including inactive items. The inherited `parameter_data` pointer is
 * a non-owning self-link and is valid only for this object's lifetime.
 *
 * `resize()` supports activation changes with an unchanged model/name set. An
 * add or remove invalidates the data and requires recreation; stale layouts or
 * names are rejected before access. `setZero()` preserves every dimension and
 * activation flag while zeroing the aggregate and all per-item data. Copies
 * share aggregate and per-item data ownership, matching collector semantics,
 * and rebuild the non-owning self-link to the copied object. Thus, data
 * mutations remain shared while the copied collector cannot dangle through the
 * original manager-data object.
 */
template <typename _Scalar>
struct ParameterDataManagerTpl : public DataCollectorParamsTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef DataCollectorParamsTpl<Scalar> Base;
  typedef ParamsDataAbstractTpl<Scalar> ParamsDataAbstract;
  typedef ParameterManagerTpl<Scalar> ParameterManager;

  typedef std::map<std::string, std::shared_ptr<ParamsDataAbstract> >
      ParameterDataContainer;

  /**
   * @brief Initialize complete data from a non-null manager
   *
   * @throw crocoddyl::Exception for a null model or null/inconsistent item data
   */
  explicit ParameterDataManagerTpl(const ParameterManager* const model);
  ParameterDataManagerTpl(const ParameterDataManagerTpl& other);
  ParameterDataManagerTpl& operator=(const ParameterDataManagerTpl&) = delete;
  virtual ~ParameterDataManagerTpl() = default;

  /**
   * @brief Resize the aggregate layout after status-only changes
   *
   * @throw crocoddyl::Exception for null arguments, an invalid self-link, or
   * stale/inconsistent item names and data
   */
  void resize(const ParameterManager* const model);

  /**
   * @brief Zero the aggregate and every active/inactive item data
   * @throw crocoddyl::Exception if an aggregate or per-item data is null
   */
  void setZero();

  ParameterDataContainer action_params;    //!< All action item data
  ParameterDataContainer dynamics_params;  //!< All dynamics item data

 private:
  typedef std::map<std::string, std::size_t> ActiveOffsetContainer;

  void refreshActiveLayout(const ParameterManager* const model);

  ActiveOffsetContainer active_offsets_;

  friend struct internal::ParameterDataManagerAccessTpl<Scalar>;
};

namespace internal {

/** @brief Internal read access to the frozen active parameter layout. */
template <typename Scalar>
struct ParameterDataManagerAccessTpl {
  static std::size_t getActiveOffset(
      const ParameterDataManagerTpl<Scalar>& data, const std::string& name) {
    typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
    typename ParameterDataManager::ActiveOffsetContainer::const_iterator it =
        data.active_offsets_.find(name);
    if (it == data.active_offsets_.end()) {
      throw_pretty("Invalid argument: parameter data for '"
                   << name << "' does not exist");
    }
    if (it->second == std::numeric_limits<std::size_t>::max()) {
      throw_pretty("Invalid argument: parameter '" << name << "' is inactive");
    }
    return it->second;
  }
};

}  // namespace internal

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/params/parameter-manager.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ParameterItemTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ParameterManagerTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ParameterDataManagerTpl)

#endif  // CROCODDYL_CORE_PARAMS_PARAMETER_MANAGER_HPP_
