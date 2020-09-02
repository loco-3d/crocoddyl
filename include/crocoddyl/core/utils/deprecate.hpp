///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_DEPRECATE_HPP_
#define CROCODDYL_CORE_UTILS_DEPRECATE_HPP_

// Helper to deprecate functions and methods
// See https://blog.samat.io/2017/02/27/Deprecating-functions-and-methods-in-Cplusplus/
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

#endif  // CROCODDYL_CORE_UTILS_DEPRECATE_HPP_