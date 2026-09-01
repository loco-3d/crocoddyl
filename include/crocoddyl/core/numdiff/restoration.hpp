///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_RESTORATION_HPP_
#define CROCODDYL_CORE_NUMDIFF_RESTORATION_HPP_

#include <memory>

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace internal {

template <typename Model>
const std::shared_ptr<Model>& checkNumDiffModel(
    const std::shared_ptr<Model>& model) {
  if (model == nullptr) {
    throw_pretty("Invalid argument: model is null");
  }
  return model;
}

template <typename Model>
Model* checkNumDiffModel(Model* const model) {
  if (model == nullptr) {
    throw_pretty("Invalid argument: model is null");
  }
  return model;
}

/**
 * @brief Restore mutable evaluation state when a numerical perturbation exits
 *
 * Numerical differentiation temporarily changes parameter and collector data.
 * This internal guard restores the nominal state both on success and while an
 * exception propagates. The explicit `restore()` call preserves restoration
 * errors on the normal path; destruction performs best-effort restoration
 * without replacing an active exception.
 */
template <typename Restore>
class NumDiffRestorationTpl {
 public:
  explicit NumDiffRestorationTpl(Restore restore)
      : restore_(restore), active_(true) {}

  NumDiffRestorationTpl(const NumDiffRestorationTpl&) = delete;
  NumDiffRestorationTpl& operator=(const NumDiffRestorationTpl&) = delete;

  ~NumDiffRestorationTpl() {
    if (active_) {
      try {
        restore_();
      } catch (...) {
      }
    }
  }

  void restore() {
    restore_();
    active_ = false;
  }

 private:
  Restore restore_;
  bool active_;
};

}  // namespace internal
}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_NUMDIFF_RESTORATION_HPP_
