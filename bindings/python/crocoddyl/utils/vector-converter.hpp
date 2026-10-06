///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_UTILS_VECTOR_CONVERTER_HPP_
#define BINDINGS_PYTHON_CROCODDYL_UTILS_VECTOR_CONVERTER_HPP_

#include <eigenpy/std-vector.hpp>

namespace crocoddyl {
namespace python {

using eigenpy::StdVectorPythonVisitor;

/**
 * @brief Keep a vector alive while one of its elements is referenced by Python
 *
 * This visitor replaces the vector's element getter with one that makes the
 * returned element the custodian of its containing vector. It is useful for
 * vectors of data objects whose validity depends on other storage owned by the
 * parent data object.
 */
template <typename Container>
struct StdVectorElementOwnerPythonVisitor
    : public boost::python::def_visitor<
          StdVectorElementOwnerPythonVisitor<Container> > {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def("__getitem__", &getItem,
           boost::python::return_value_policy<
               boost::python::return_by_value,
               boost::python::with_custodian_and_ward_postcall<0, 1> >());
  }

 private:
  static typename Container::value_type getItem(Container& container,
                                                boost::python::ssize_t index) {
    const boost::python::ssize_t size =
        static_cast<boost::python::ssize_t>(container.size());
    if (index < 0) {
      index += size;
    }
    if (index < 0 || index >= size) {
      PyErr_SetString(PyExc_IndexError, "Index out of range");
      boost::python::throw_error_already_set();
    }
    return container[static_cast<std::size_t>(index)];
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_UTILS_VECTOR_CONVERTER_HPP_
