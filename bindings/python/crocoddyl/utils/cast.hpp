

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
template <typename Model>
struct CastVisitor : public bp::def_visitor<CastVisitor<Model>> {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def("cast", &cast_instance, bp::arg("dtype"),
           "Returns a copy of *this.");
  }

 private:
  static bp::object cast_instance(const Model& self, DType dtype) {
    switch (dtype) {
      case DType::Float64:
        return bp::object(self.template cast<Float64>());
      case DType::Float32:
        return bp::object(self.template cast<Float32>());
#ifdef CROCODDYL_WITH_CODEGEN_DISABLE
      case DType::ADFloat64:
        return bp::object(self.template cast<ADFloat64>());
#endif
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
