///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/params/parameter-phase.hpp"

namespace crocoddyl {

template <typename Scalar>
ProblemAbstractTpl<Scalar>::ProblemAbstractTpl() : nthreads_(1) {
#ifdef CROCODDYL_WITH_MULTITHREADING
  if (enableMultithreading()) {
    nthreads_ = CROCODDYL_WITH_NTHREADS;
  }
#endif
}

template <typename Scalar>
ProblemAbstractTpl<Scalar>::ProblemAbstractTpl(
    const ProblemAbstractTpl<Scalar>& problem)
    : nthreads_(problem.nthreads_), is_updated_(problem.is_updated_) {}

template <typename Scalar>
std::vector<typename ProblemAbstractTpl<Scalar>::VectorXs>
ProblemAbstractTpl<Scalar>::rollout_us(const std::vector<VectorXs>& us) {
  std::vector<VectorXs> xs(get_T() + 1);
  rollout(us, xs);
  return xs;
}

template <typename Scalar>
void ProblemAbstractTpl<Scalar>::updateWarmstart() {
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      get_runningDatas();
  const std::size_t T = get_T();
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(get_nthreads())
#endif
  for (std::size_t i = 0; i < T; ++i) {
    models[i]->updateWarmstart(datas[i]);
  }
  get_terminalModel()->updateWarmstart(get_terminalData());
}

template <typename Scalar>
std::size_t ProblemAbstractTpl<Scalar>::get_nthreads() const {
  return nthreads_;
}

template <typename Scalar>
void ProblemAbstractTpl<Scalar>::set_nthreads(const int nthreads) {
#ifndef CROCODDYL_WITH_MULTITHREADING
  (void)nthreads;
  std::cerr << "Warning: the number of threads won't affect the computational "
               "performance as multithreading support is not enabled."
            << std::endl;
#else
  if (nthreads < 1) {
    nthreads_ = CROCODDYL_WITH_NTHREADS;
  } else {
    nthreads_ = static_cast<std::size_t>(nthreads);
  }
  if (!enableMultithreading()) {
    std::cerr << "Warning: the number of threads won't affect the "
                 "computational performance as multithreading support is not "
                 "enabled."
              << std::endl;
    nthreads_ = 1;
  }
#endif
}

template <typename Scalar>
std::vector<
    std::shared_ptr<typename ProblemAbstractTpl<Scalar>::ActionDataAbstract> >
ProblemAbstractTpl<Scalar>::get_runningPhaseDatas(
    const std::size_t phase_idx) const {
  const std::vector<std::size_t>& starts = get_phase_idxs();
  const std::vector<std::size_t>& ends = get_phase_edxs();
  if (phase_idx >= starts.size() || phase_idx >= ends.size()) {
    throw_pretty("Invalid argument: phase_idx " << phase_idx << " >= n_phases "
                                                << starts.size());
  }
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      get_runningDatas();
  std::vector<std::shared_ptr<ActionDataAbstract> > phase_datas;
  phase_datas.reserve(ends[phase_idx] - starts[phase_idx]);
  for (std::size_t t = starts[phase_idx]; t < ends[phase_idx]; ++t) {
    phase_datas.push_back(datas[t]);
  }
  return phase_datas;
}

template <typename Scalar>
bool ProblemAbstractTpl<Scalar>::is_updated() {
  const bool status = is_updated_;
  is_updated_ = false;
  return status;
}

template <typename Scalar>
void ProblemAbstractTpl<Scalar>::set_is_updated(const bool is_updated) {
  is_updated_ = is_updated;
}

template <typename Scalar>
std::size_t ProblemAbstractTpl<Scalar>::get_n_phases() const {
  return get_paramsModel().size();
}

template <typename Scalar>
void ProblemAbstractTpl<Scalar>::update_p(const Eigen::Ref<const VectorXs>& p,
                                          const std::size_t phase_idx) {
  const std::vector<std::shared_ptr<ParameterPhaseModel> >& models =
      get_paramsModel();
  if (phase_idx >= models.size()) {
    throw_pretty("Invalid argument: phase_idx " << phase_idx << " >= n_phases "
                                                << models.size());
  }
  models[phase_idx]->update(get_paramsData()[phase_idx], p);
}

template <typename Scalar>
const std::vector<std::size_t>& ProblemAbstractTpl<Scalar>::get_phase_idxs()
    const {
  static const std::vector<std::size_t> empty;
  return empty;
}

template <typename Scalar>
const std::vector<std::size_t>& ProblemAbstractTpl<Scalar>::get_phase_edxs()
    const {
  static const std::vector<std::size_t> empty;
  return empty;
}

template <typename Scalar>
const std::vector<
    std::shared_ptr<typename ProblemAbstractTpl<Scalar>::ParameterPhaseModel> >&
ProblemAbstractTpl<Scalar>::get_paramsModel() const {
  static const std::vector<std::shared_ptr<ParameterPhaseModel> > empty;
  return empty;
}

template <typename Scalar>
const std::vector<
    std::shared_ptr<typename ProblemAbstractTpl<Scalar>::ParameterPhaseData> >&
ProblemAbstractTpl<Scalar>::get_paramsData() const {
  static const std::vector<std::shared_ptr<ParameterPhaseData> > empty;
  return empty;
}

template <typename Scalar>
bool ProblemAbstractTpl<Scalar>::has_parameter_constraints() const {
  const std::vector<std::shared_ptr<ParameterPhaseModel> >& models =
      get_paramsModel();
  for (std::size_t i = 0; i < models.size(); ++i) {
    if (models[i]->has_constraints()) {
      return true;
    }
  }
  return false;
}

}  // namespace crocoddyl
