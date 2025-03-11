///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh, CNRS, INRIA
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

/**
 * @brief Macros for generating compiler warnings via pragmas (GCC/Clang only)
 *
 * These macros allow the generation of custom compiler warnings and messages
 * using `#pragma` in a portable and readable way. They are useful for alerting
 * users about deprecated headers, features that will be removed, or general
 * warnings during compilation.
 *
 * Supported compilers: GCC and Clang
 *
 * References:
 *   https://stackoverflow.com/questions/171435/portability-of-warning-preprocessor-directive
 *
 * Usage:
 *   CROCODDYL_PRAGMA_WARNING("This is a warning")
 *   CROCODDYL_PRAGMA_DEPRECATED("This feature is deprecated")
 *   CROCODDYL_PRAGMA_DEPRECATED_HEADER("old.hpp", "new.hpp")
 *   CROCODDYL_PRAGMA_TO_BE_REMOVED_HEADER("soon_removed.hpp")
 */
#if defined(__GNUC__) || defined(__clang__)
#define CROCODDYL_PRAGMA(x) _Pragma(#x)

/**
 * @brief Emit a custom compiler message
 *
 * @param[in] the_message  The message to display during compilation
 */
#define CROCODDYL_PRAGMA_MESSAGE(the_message) \
  CROCODDYL_PRAGMA(GCC message #the_message)

/**
 * @brief Emit a custom compiler warning
 *
 * @param[in] the_message  The warning message to display
 */
#define CROCODDYL_PRAGMA_WARNING(the_message) \
  CROCODDYL_PRAGMA(GCC warning #the_message)

/**
 * @brief Emit a deprecation warning with a custom message
 *
 * @param[in] the_message  The deprecation message
 */
#define CROCODDYL_PRAGMA_DEPRECATED(the_message) \
  CROCODDYL_PRAGMA_WARNING(Deprecated : #the_message)

#ifndef CROCODDYL_IGNORE_DEPRECATED_HEADER
/**
 * @brief Emit a warning for deprecated headers that have replacements
 *
 * @param[in] old_header  The deprecated header
 * @param[in] new_header  The recommended replacement
 */
#define CROCODDYL_PRAGMA_DEPRECATED_HEADER(old_header, new_header) \
  CROCODDYL_PRAGMA_WARNING(                                        \
      Deprecated header file : old_header has been replaced by     \
          new_header.\n Please use new_header instead of old_header.)
#else
#define CROCODDYL_PRAGMA_DEPRECATED_HEADER(old_header, new_header)
#endif  // CROCODDYL_IGNORE_DEPRECATED_HEADER

#ifndef CROCODDYL_IGNORE_DEPRECATED_HEADER
/**
 * @brief Emit a warning for headers that will be removed in the future
 *
 * @param[in] remove_header  The header to be deprecated and removed
 */
#define CROCODDYL_PRAGMA_TO_BE_REMOVED_HEADER(remove_header) \
  CROCODDYL_PRAGMA_WARNING(                                  \
      Deprecated header file : remove_header has now been    \
          deprecated.\n It would be removed in the upcoming releases.)
#else
#define CROCODDYL_PRAGMA_TO_BE_REMOVED_HEADER(remove_header)
#endif  // CROCODDYL_IGNORE_TO_BE_REMOVED_HEADER
#endif  // defined(__GNUC__) || defined(__clang__)

/**
 * @brief Macros to temporarily disable and re-enable deprecated warnings
 *
 * These macros allow you to suppress deprecation warnings (e.g., from
 * using [[deprecated]] functions) in a cross-platform way. This is
 * particularly useful when maintaining compatibility while deprecating APIs.
 *
 * Usage:
 *   CROCODDYL_DISABLE_WARNING_DEPRECATED
 *   // Code that may trigger deprecation warnings
 *   CROCODDYL_ENABLE_WARNING_DEPRECATED
 *
 * For compilers that support warning pragmas (MSVC, Clang, GCC), the macros
 * use push/pop to ensure the warning state is restored after use.
 * For unknown compilers, these macros expand to nothing.
 */
#if defined(_MSC_VER)  // Microsoft Visual C++
#define CROCODDYL_DISABLE_WARNING_DEPRECATED \
  __pragma(warning(push)) __pragma(warning(disable : 4996))
#define CROCODDYL_ENABLE_WARNING_DEPRECATED __pragma(warning(pop))
#elif defined(__clang__) || defined(__GNUC__)  // Clang or GCC
#define DO_PRAGMA(X) _Pragma(#X)
#define CROCODDYL_DISABLE_WARNING_DEPRECATED \
  DO_PRAGMA(GCC diagnostic push)             \
  DO_PRAGMA(GCC diagnostic ignored "-Wdeprecated-declarations")
#define CROCODDYL_ENABLE_WARNING_DEPRECATED DO_PRAGMA(GCC diagnostic pop)
#else  // Unknown compilers — macros do nothing
#define CROCODDYL_DISABLE_WARNING_DEPRECATED
#define CROCODDYL_ENABLE_WARNING_DEPRECATED
#endif

#endif  // CROCODDYL_CORE_UTILS_DEPRECATE_HPP_
