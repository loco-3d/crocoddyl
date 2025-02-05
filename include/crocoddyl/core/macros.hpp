///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2025, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MACROS_HPP_
#define CROCODDYL_MACROS_HPP_

#include <iostream>
#include <memory>
#include <vector>

#define DEFAULT_SCALAR double

#if __cplusplus <= 201103L
namespace std {
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
}  // namespace std
#endif

#define CROCODDYL_BASE_CAST(base_class, class)                              \
  template <typename Scalar>                                                \
  std::shared_ptr<class<Scalar>> cast() const {                             \
    return std::static_pointer_cast<class<Scalar>>(cast_impl<Scalar>());    \
  } template <typename Scalar>                                              \
  std::shared_ptr<base_class> cast_impl() const {                           \
    if (typeid(Scalar) == typeid(double)) {                                 \
      return cloneAsDouble();                                               \
    } else if (typeid(Scalar) == typeid(float)) {                           \
      return cloneAsFloat();                                                \
    } else {                                                                \
      std::cout << "Unsupported casting: casting to double as default"      \
                << std::endl;                                               \
      return cloneAsDouble();                                               \
    }                                                                       \
  }                                                                         \
  /* Pure virtual method that derived classes must implement for casting */ \
  virtual std::unique_ptr<base_class> cloneAsDouble() const = 0;            \
  virtual std::unique_ptr<base_class> cloneAsFloat() const = 0;

/**
 * @brief Macro to declare the code for casting a Crocoddyl class
 * When including this macro within a class, it is necessary to define the
 * specialized cast operator, e.g.,
 *  template <typename NewScalar>
 *  derived_class<NewScalar> cast() const {
 *      typedef derived_class<NewScalar> ReturnType;
 *      ReturnType ret("pass arguments from 'this'");
 *      return ret;
 *  }
 */
#define CROCODDYL_DERIVED_CAST(base_class, derived_class)                 \
  template <typename NewScalar>                                           \
  explicit derived_class(const derived_class<NewScalar>& other) {         \
    *this = other.template cast<Scalar>(); /* This needs to define a cast \
                                              operator */                 \
  }                                                                       \
  /* Implements casting by overriding `cloneAsFloat` */                   \
  std::unique_ptr<base_class> cloneAsDouble() const override {            \
    return std::make_unique<derived_class<double>>(*this);                \
  }                                                                       \
  std::unique_ptr<base_class> cloneAsFloat() const override {             \
    return std::make_unique<derived_class<float>>(*this);                 \
  }

#define CROCODDYL_INNER_DERIVED_CAST(base_class, inner_class, derived_class) \
  template <typename NewScalar>                                              \
  explicit derived_class(                                                    \
      const typename inner_class<NewScalar>::derived_class& other) {         \
    *this = other.template cast<Scalar>(); /* This needs to define a cast    \
                                              operator */                    \
  }                                                                          \
  /* Implements casting by overriding `cloneAsFloat` */                      \
  std::unique_ptr<base_class> cloneAsDouble() const override {               \
    return std::make_unique<typename inner_class<double>::derived_class>(    \
        this->template cast<double>());                                      \
  }                                                                          \
  std::unique_ptr<base_class> cloneAsFloat() const override {                \
    return std::make_unique<typename inner_class<float>::derived_class>(     \
        this->template cast<float>());                                       \
  }

#endif  // CROCODDYL_MACROS_HPP_
