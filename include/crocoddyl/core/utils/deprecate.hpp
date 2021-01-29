///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh, CNRS, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_DEPRECATE_HPP_
#define CROCODDYL_CORE_UTILS_DEPRECATE_HPP_

// Helper to deprecate functions and methods
// See
// https://blog.samat.io/2017/02/27/Deprecating-functions-and-methods-in-Cplusplus/
// For C++14
#if __cplusplus >= 201402L
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated)
#define DEPRECATED(msg, func) [[deprecated(msg)]] func
#endif
#endif
// For everyone else
#else
#ifdef __GNUC__
#define DEPRECATED(msg, func) func __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#define DEPRECATED(msg, func) __declspec(deprecated(msg)) func
#endif
#endif

// For more details, visit
// https://stackoverflow.com/questions/171435/portability-of-warning-preprocessor-directive
// (copy paste from pinocchio/macros.hpp)
#if defined(__GNUC__) || defined(__clang__)
#define CROCODDYL_PRAGMA(x) _Pragma(#x)
#define CROCODDYL_PRAGMA_MESSAGE(the_message)                                  \
  CROCODDYL_PRAGMA(GCC message #the_message)
#define CROCODDYL_PRAGMA_WARNING(the_message)                                  \
  CROCODDYL_PRAGMA(GCC warning #the_message)
#define CROCODDYL_PRAGMA_DEPRECATED(the_message)                               \
  CROCODDYL_PRAGMA_WARNING(Deprecated : #the_message)
#define CROCODDYL_PRAGMA_DEPRECATED_HEADER(old_header, new_header)             \
  CROCODDYL_PRAGMA_WARNING(                                                    \
      Deprecated header file                                                   \
      : old_header has been replaced by                                        \
            new_header.\n Please use new_header instead of old_header.)
#endif

#endif // CROCODDYL_CORE_UTILS_DEPRECATE_HPP_
