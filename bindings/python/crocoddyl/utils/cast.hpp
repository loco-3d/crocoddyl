

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2025, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_UTILS_CAST_HPP_
#define BINDINGS_PYTHON_CROCODDYL_UTILS_CAST_HPP_

#include <boost/python.hpp>

#include "crocoddyl/core/utils/conversions.hpp"
#include "python/crocoddyl/utils/scalar.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

/**
 * @brief Add the Python method cast to allow casting of this by predefined cast
 * types.
 */
template <typename Model, bool FTypesOnly = false>
struct CastVisitor : public bp::def_visitor<CastVisitor<Model, FTypesOnly>> {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def("cast", &cast_instance<FTypesOnly>, bp::arg("dtype"),
           "Returns a copy of *this.");
  }

 private:
  // Helper function for casting, using std::enable_if to handle specific types
  template <typename ScalarType>
  static bp::object cast_instance_impl(const Model& self, std::true_type) {
    // No cast needed if ScalarType is the correct type
    return bp::object(self);
  }

  template <typename ScalarType>
  static bp::object cast_instance_impl(const Model& self, std::false_type) {
    // Otherwise, perform the cast to the requested type
    return bp::object(self.template cast<ScalarType>());
  }

  // Main cast instance function that uses SFINAE
  template <bool IsFTypesOnly = FTypesOnly>
  static typename std::enable_if<!IsFTypesOnly, bp::object>::type cast_instance(
      const Model& self, DType dtype) {
    switch (dtype) {
      case DType::Float64:
        return cast_instance_impl<Float64>(
            self, std::is_same<typename Model::Scalar, Float64>());
      case DType::Float32:
        return cast_instance_impl<Float32>(
            self, std::is_same<typename Model::Scalar, Float32>());
#ifdef CROCODDYL_WITH_CODEGEN
      case DType::ADFloat64:
        return cast_instance_impl<ADFloat64>(
            self, std::is_same<typename Model::Scalar, ADFloat64>());
#endif
      default:
        PyErr_SetString(PyExc_TypeError, "Unsupported dtype.");
        bp::throw_error_already_set();
        return bp::object();
    }
  }

  template <bool IsFTypesOnly = FTypesOnly>
  static typename std::enable_if<IsFTypesOnly, bp::object>::type cast_instance(
      const Model& self, DType dtype) {
    switch (dtype) {
      case DType::Float64:
        return cast_instance_impl<Float64>(
            self, std::is_same<typename Model::Scalar, Float64>());
      case DType::Float32:
        return cast_instance_impl<Float32>(
            self, std::is_same<typename Model::Scalar, Float32>());
      default:
        PyErr_SetString(PyExc_TypeError, "Unsupported dtype.");
        bp::throw_error_already_set();
        return bp::object();
    }
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_UTILS_CAST_HPP_
