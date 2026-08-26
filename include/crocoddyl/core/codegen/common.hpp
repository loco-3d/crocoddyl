///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, INRIA, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CODEGEN_COMMON_HPP_
#define CROCODDYL_CORE_CODEGEN_COMMON_HPP_

#ifdef CROCODDYL_WITH_CODEGEN

#ifdef _OPENMP
#include <omp.h>
#endif

#include "crocoddyl/core/action-base.hpp"

namespace crocoddyl {

struct CodegenEigenThreadGuard {
  explicit CodegenEigenThreadGuard(const int threads)
      : previous_threads(Eigen::nbThreads())
#ifdef _OPENMP
        ,
        previous_omp_threads(omp_get_max_threads())
#endif
  {
    Eigen::setNbThreads(threads);
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
  }

  ~CodegenEigenThreadGuard() {
    Eigen::setNbThreads(previous_threads);
#ifdef _OPENMP
    omp_set_num_threads(previous_omp_threads);
#endif
  }

  int previous_threads;
#ifdef _OPENMP
  int previous_omp_threads;
#endif
};

template <typename Scalar>
std::unique_ptr<CppAD::ADFun<CppAD::cg::CG<Scalar>>> clone_adfun(
    const CppAD::ADFun<CppAD::cg::CG<Scalar>>& original) {
  auto cloned = std::make_unique<CppAD::ADFun<CppAD::cg::CG<Scalar>>>();
  *cloned = original;  // Use assignment operator to copy the function
  return cloned;
}

template <typename FromScalar, typename ToScalar>
std::function<
    void(std::shared_ptr<ActionModelAbstractTpl<ToScalar>>,
         const Eigen::Ref<const typename MathBaseTpl<ToScalar>::VectorXs>&)>
cast_function(
    const std::function<void(
        std::shared_ptr<ActionModelAbstractTpl<FromScalar>>,
        const Eigen::Ref<const typename MathBaseTpl<FromScalar>::VectorXs>&)>&
        fn) {
  return [fn](std::shared_ptr<ActionModelAbstractTpl<ToScalar>> to_base,
              const Eigen::Ref<const typename MathBaseTpl<ToScalar>::VectorXs>&
                  to_vector) {
    const std::shared_ptr<ActionModelAbstractTpl<FromScalar>>& from_base =
        to_base->template cast<FromScalar>();
    const typename MathBaseTpl<FromScalar>::VectorXs from_vector =
        to_vector.template cast<FromScalar>();
    fn(from_base, from_vector);
  };
}

enum CompilerType { GCC = 0, CLANG };

inline constexpr CompilerType defaultCompilerType() {
#if defined(__clang__)
  return CLANG;
#elif defined(__GNUC__)
  return GCC;
#else
  return CLANG;
#endif
}

inline const char* compilerExecutable(CompilerType compiler) {
  switch (compiler) {
    case GCC:
#ifdef CROCODDYL_CODEGEN_GCC_COMPILER_PATH
      return CROCODDYL_CODEGEN_GCC_COMPILER_PATH;
#else
      return "/usr/bin/gcc";
#endif
    case CLANG:
#ifdef CROCODDYL_CODEGEN_CLANG_COMPILER_PATH
      return CROCODDYL_CODEGEN_CLANG_COMPILER_PATH;
#else
      return "/usr/bib/clang";
#endif
  }
  return "cc";
}

}  // namespace crocoddyl

#endif  // CROCODDYL_WITH_CODEGEN

#endif  // CROCODDYL_CORE_CODEGEN_COMMON_HPP_
