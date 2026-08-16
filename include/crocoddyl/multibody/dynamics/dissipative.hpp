///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_DYNAMICS_DISSIPATIVE_HPP_
#define CROCODDYL_MULTIBODY_DYNAMICS_DISSIPATIVE_HPP_

#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/multibody/actuations/multibody.hpp"
#include "crocoddyl/multibody/params/actuation.hpp"

namespace crocoddyl {
namespace internal {

/** @brief Internal continuous-dynamics view used by parameter regressors */
template <typename Scalar>
struct DynamicsDataParameterRegressorTpl
    : public DynamicsDataAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef DynamicsDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;

  template <template <typename S> class Model>
  DynamicsDataParameterRegressorTpl(Model<Scalar>* const model,
                                    const std::size_t nu,
                                    DataCollectorAbstract* const shared)
      : Base(model), u(nu) {
    resize(model, nu, shared);
  }

  DynamicsDataParameterRegressorTpl(
      const DynamicsDataParameterRegressorTpl& other,
      DataCollectorAbstract* const shared)
      : Base(other), u(other.u) {
    Base::shared = shared;
  }

  template <template <typename S> class Model>
  void resize(Model<Scalar>* const model, const std::size_t nu,
              DataCollectorAbstract* const shared) {
    const std::size_t nv = model->get_state()->get_nv();
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t np = model->get_np();
    Base::vdot.resize(nv);
    Base::Fx.resize(nv, ndx);
    Base::Fu.resize(nv, nu);
    Base::Fp.resize(nv, np);
    Base::tmp_ustatic.resize(nu);
    u.resize(nu);
    Base::shared = shared;
    Base::setZero();
    u.setZero();
  }

  VectorXs u;  //!< Command workspace when the native dynamics has no control
};

/** @brief Restore a manager's full-layout mode after active-layout work */
template <typename ConstraintManager>
class ActiveConstraintModeGuardTpl {
 public:
  explicit ActiveConstraintModeGuardTpl(
      const std::shared_ptr<ConstraintManager>& constraints)
      : constraints_(constraints),
        restore_(constraints->getComputeAllConstraints()) {
    if (restore_) {
      constraints_->setComputeAllConstraints(false);
    }
  }

  ~ActiveConstraintModeGuardTpl() { restore(); }

  void restore() {
    if (restore_) {
      constraints_->setComputeAllConstraints(true);
      restore_ = false;
    }
  }

  ActiveConstraintModeGuardTpl(const ActiveConstraintModeGuardTpl&) = delete;
  ActiveConstraintModeGuardTpl& operator=(const ActiveConstraintModeGuardTpl&) =
      delete;

 private:
  std::shared_ptr<ConstraintManager> constraints_;
  bool restore_;
};

template <typename Scalar>
void updateDissipativePowerFromActuation(
    const std::shared_ptr<ActuationDataAbstractTpl<Scalar> >& actuation_data,
    const Eigen::Ref<const typename MathBaseTpl<Scalar>::VectorXs>& v,
    typename MathBaseTpl<Scalar>::VectorXs& dissipative_P,
    typename MathBaseTpl<Scalar>::MatrixXs* const dP_dv = nullptr) {
  dissipative_P.setZero();
  if (dP_dv != nullptr) {
    dP_dv->setZero();
  }

  const ActuationDataMultibodyTpl<Scalar>* multibody_data =
      dynamic_cast<const ActuationDataMultibodyTpl<Scalar>*>(
          actuation_data.get());
  if (multibody_data == nullptr) {
    return;
  }

  dissipative_P[0] = multibody_data->friction.dot(v);
  if (dP_dv != nullptr) {
    dP_dv->row(0).noalias() =
        (-actuation_data->dtau_dx.rightCols(v.size()) * v +
         multibody_data->friction)
            .transpose();
  }
}

template <typename Scalar>
void updateDissipativePowerParams(
    const std::shared_ptr<ParameterManagerTpl<Scalar> >& params,
    const Eigen::Ref<const typename MathBaseTpl<Scalar>::MatrixXs>& dtau_dp,
    const Eigen::Ref<const typename MathBaseTpl<Scalar>::VectorXs>& v,
    typename MathBaseTpl<Scalar>::MatrixXs& dP_dp) {
  dP_dp.setZero();
  if (params == nullptr) {
    return;
  }

  std::size_t offset = params->get_np_action();
  std::size_t dynamics_offset = 0;
  for (typename ParameterManagerTpl<Scalar>::ParameterContainer::const_iterator
           it = params->get_dynamics_params().begin();
       it != params->get_dynamics_params().end(); ++it) {
    const std::shared_ptr<typename ParameterManagerTpl<Scalar>::ParameterItem>&
        item = it->second;
    if (!item->get_active()) {
      continue;
    }

    const std::size_t np_item = item->get_param()->get_np();
    const std::shared_ptr<ActuationMultibodyParamsTpl<Scalar> >
        actuation_param =
            std::dynamic_pointer_cast<ActuationMultibodyParamsTpl<Scalar> >(
                item->get_param());
    if (actuation_param != nullptr) {
      dP_dp.middleCols(offset, np_item).row(0).noalias() =
          v.transpose() * dtau_dp.middleCols(dynamics_offset, np_item);
    }
    offset += np_item;
    dynamics_offset += np_item;
  }
}

}  // namespace internal
}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_DYNAMICS_DISSIPATIVE_HPP_
