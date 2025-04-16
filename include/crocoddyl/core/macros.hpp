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

#ifdef CROCODDYL_WITH_CODEGEN
#include <cppad/cg/support/cppadcg_eigen.hpp>
#include <cppad/cppad.hpp>
#endif

#define DEFAULT_SCALAR double
typedef double Float64;
typedef float Float32;
#ifdef CROCODDYL_WITH_CODEGEN
typedef CppAD::cg::CG<Float64> CGFloat64;
typedef CppAD::AD<CGFloat64> ADFloat64;
#endif

#if __cplusplus <= 201103L
namespace std {
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
}  // namespace std
#endif

#ifdef CROCODDYL_WITH_CODEGEN
#define CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(class_name)                 \
  extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<double>;                                                   \
  extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<float>;                                                    \
  extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<ADFloat64>;

#define CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(class_name)                 \
  extern template struct CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<double>;                                                    \
  extern template struct CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<float>;                                                     \
  extern template struct CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<ADFloat64>;

#define CROCODDYL_DECLARE_FLOATINGPOINT_EXTERN_TEMPLATE_CLASS(class_name)   \
  extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<double>;                                                   \
  extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<float>;

#define CROCODDYL_DECLARE_FLOATINGPOINT_EXTERN_TEMPLATE_STRUCT(class_name)   \
  extern template struct CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<double>;                                                    \
  extern template struct CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<float>;
#else
#define CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(class_name)                 \
  extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<double>;                                                   \
  extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<float>;

#define CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(class_name)                 \
  extern template struct CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<double>;                                                    \
  extern template struct CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<float>;

#define CROCODDYL_DECLARE_FLOATINGPOINT_EXTERN_TEMPLATE_CLASS(class_name)   \
  extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<double>;                                                   \
  extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<float>;

#define CROCODDYL_DECLARE_FLOATINGPOINT_EXTERN_TEMPLATE_STRUCT(class_name)   \
  extern template struct CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<double>;                                                    \
  extern template struct CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI \
      class_name<float>;
#endif

#ifdef CROCODDYL_WITH_CODEGEN
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
    } else if (typeid(Scalar) == typeid(ADFloat64)) {                       \
      return cloneAsADDouble();                                             \
    } else {                                                                \
      std::cout << "Unsupported casting: casting to double as default"      \
                << std::endl;                                               \
      return cloneAsDouble();                                               \
    }                                                                       \
  }                                                                         \
  /* Pure virtual method that derived classes must implement for casting */ \
  virtual std::shared_ptr<base_class> cloneAsDouble() const = 0;            \
  virtual std::shared_ptr<base_class> cloneAsFloat() const = 0;             \
  virtual std::shared_ptr<base_class> cloneAsADDouble() const = 0;

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
#define CROCODDYL_DERIVED_CAST(base_class, derived_class)          \
  template <typename NewScalar>                                    \
  explicit derived_class(const derived_class<NewScalar>& other)    \
      : derived_class(std::move(other.template cast<Scalar>())) {} \
  /* Implements casting by overriding `cloneAsFloat` */            \
  std::shared_ptr<base_class> cloneAsDouble() const override {     \
    return std::make_shared<derived_class<double>>(*this);         \
  }                                                                \
  std::shared_ptr<base_class> cloneAsFloat() const override {      \
    return std::make_shared<derived_class<float>>(*this);          \
  }                                                                \
  std::shared_ptr<base_class> cloneAsADDouble() const override {   \
    return std::make_shared<derived_class<ADFloat64>>(*this);      \
  }

#define CROCODDYL_DERIVED_FLOATINGPOINT_CAST(base_class, derived_class) \
  template <typename NewScalar>                                         \
  explicit derived_class(const derived_class<NewScalar>& other)         \
      : derived_class(std::move(other.template cast<Scalar>())) {}      \
  /* Implements casting by overriding `cloneAsFloat` */                 \
  std::shared_ptr<base_class> cloneAsDouble() const override {          \
    return std::make_shared<derived_class<double>>(*this);              \
  }                                                                     \
  std::shared_ptr<base_class> cloneAsFloat() const override {           \
    return std::make_shared<derived_class<float>>(*this);               \
  }                                                                     \
  std::shared_ptr<base_class> cloneAsADDouble() const override {        \
    std::cout << "Unsupported casting: retuning to double as default"   \
              << std::endl;                                             \
    return cloneAsDouble();                                             \
  }

#define CROCODDYL_BASE_DERIVED_CAST(base_class, derived_class)   \
  /* Implements casting by overriding `cloneAsFloat` */          \
  std::shared_ptr<base_class> cloneAsDouble() const override {   \
    return std::shared_ptr<base_class>(nullptr);                 \
  }                                                              \
  std::shared_ptr<base_class> cloneAsFloat() const override {    \
    return std::shared_ptr<base_class>(nullptr);                 \
  }                                                              \
  std::shared_ptr<base_class> cloneAsADDouble() const override { \
    return std::shared_ptr<base_class>(nullptr);                 \
  }

#define CROCODDYL_INNER_DERIVED_CAST(base_class, inner_class, derived_class) \
  template <typename NewScalar>                                              \
  explicit derived_class(                                                    \
      const typename inner_class<NewScalar>::derived_class& other) {         \
    *this = other.template cast<Scalar>(); /* This needs to define a cast    \
                                              operator */                    \
  }                                                                          \
  /* Implements casting by overriding `cloneAsFloat` */                      \
  std::shared_ptr<base_class> cloneAsDouble() const override {               \
    return std::make_shared<typename inner_class<double>::derived_class>(    \
        this->template cast<double>());                                      \
  }                                                                          \
  std::shared_ptr<base_class> cloneAsFloat() const override {                \
    return std::make_shared<typename inner_class<float>::derived_class>(     \
        this->template cast<float>());                                       \
  }                                                                          \
  std::shared_ptr<base_class> cloneAsADDouble() const override {             \
    return std::make_shared<typename inner_class<ADFloat64>::derived_class>( \
        this->template cast<ADFloat64>());                                   \
  }
#else
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
  virtual std::shared_ptr<base_class> cloneAsDouble() const = 0;            \
  virtual std::shared_ptr<base_class> cloneAsFloat() const = 0;

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
#define CROCODDYL_DERIVED_CAST(base_class, derived_class)          \
  template <typename NewScalar>                                    \
  explicit derived_class(const derived_class<NewScalar>& other)    \
      : derived_class(std::move(other.template cast<Scalar>())) {} \
  /* Implements casting by overriding `cloneAsFloat` */            \
  std::shared_ptr<base_class> cloneAsDouble() const override {     \
    return std::make_shared<derived_class<double>>(*this);         \
  }                                                                \
  std::shared_ptr<base_class> cloneAsFloat() const override {      \
    return std::make_shared<derived_class<float>>(*this);          \
  }

#define CROCODDYL_DERIVED_FLOATINGPOINT_CAST(base_class, derived_class) \
  template <typename NewScalar>                                         \
  explicit derived_class(const derived_class<NewScalar>& other)         \
      : derived_class(std::move(other.template cast<Scalar>())) {}      \
  /* Implements casting by overriding `cloneAsFloat` */                 \
  std::shared_ptr<base_class> cloneAsDouble() const override {          \
    return std::make_shared<derived_class<double>>(*this);              \
  }                                                                     \
  std::shared_ptr<base_class> cloneAsFloat() const override {           \
    return std::make_shared<derived_class<float>>(*this);               \
  }

#define CROCODDYL_BASE_DERIVED_CAST(base_class, derived_class) \
  /* Implements casting by overriding `cloneAsFloat` */        \
  std::shared_ptr<base_class> cloneAsDouble() const override { \
    return std::shared_ptr<base_class>(nullptr);               \
  }                                                            \
  std::shared_ptr<base_class> cloneAsFloat() const override {  \
    return std::shared_ptr<base_class>(nullptr);               \
  }

#define CROCODDYL_INNER_DERIVED_CAST(base_class, inner_class, derived_class) \
  template <typename NewScalar>                                              \
  explicit derived_class(                                                    \
      const typename inner_class<NewScalar>::derived_class& other) {         \
    *this = other.template cast<Scalar>(); /* This needs to define a cast    \
                                              operator */                    \
  }                                                                          \
  /* Implements casting by overriding `cloneAsFloat` */                      \
  std::shared_ptr<base_class> cloneAsDouble() const override {               \
    return std::make_shared<typename inner_class<double>::derived_class>(    \
        this->template cast<double>());                                      \
  }                                                                          \
  std::shared_ptr<base_class> cloneAsFloat() const override {                \
    return std::make_shared<typename inner_class<float>::derived_class>(     \
        this->template cast<float>());                                       \
  }
#endif
#endif  // CROCODDYL_MACROS_HPP_
