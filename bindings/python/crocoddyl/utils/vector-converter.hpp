///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_UTILS_VECTOR_CONVERTER_HPP_
#define BINDINGS_PYTHON_CROCODDYL_UTILS_VECTOR_CONVERTER_HPP_

#include <Eigen/Dense>
#include <vector>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/to_python_converter.hpp>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

/// @note Registers converter from a provided type to the python
///       iterable type to the.
template <class T, bool NoProxy = true>
struct vector_to_list {
  static PyObject* convert(const std::vector<T>& vec) {
    typedef typename std::vector<T>::const_iterator const_iter;
    bp::list* l = new boost::python::list();
    for (const_iter it = vec.begin(); it != vec.end(); ++it) {
      if (NoProxy) {
        l->append(boost::ref(*it));
      } else {
        l->append(*it);
      }
    }
    return l->ptr();
  }
  static PyTypeObject const* get_pytype() { return &PyList_Type; }
};

/// @brief Type that allows for registration of conversions from
///        python iterable types.
struct list_to_vector {
  /// @note Registers converter from a python iterable type to the
  ///       provided type.
  template <typename Container>
  list_to_vector& from_python() {
    boost::python::converter::registry::push_back(&list_to_vector::convertible, &list_to_vector::construct<Container>,
                                                  boost::python::type_id<Container>());

    // Support chaining.
    return *this;
  }

  /// @brief Check if PyObject is iterable.
  static void* convertible(PyObject* object) { return PyObject_GetIter(object) ? object : NULL; }

  /// @brief Convert iterable PyObject to C++ container type.
  ///
  /// Container Concept requirements:
  ///
  ///   * Container::value_type is CopyConstructable.
  ///   * Container can be constructed and populated with two iterators.
  ///     I.e. Container(begin, end)
  template <typename Container>
  static void construct(PyObject* object, boost::python::converter::rvalue_from_python_stage1_data* data) {
    namespace python = boost::python;
    // Object is a borrowed reference, so create a handle indicting it is
    // borrowed for proper reference counting.
    python::handle<> handle(python::borrowed(object));

    // Obtain a handle to the memory block that the converter has allocated
    // for the C++ type.
    typedef python::converter::rvalue_from_python_storage<Container> storage_type;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

    typedef python::stl_input_iterator<typename Container::value_type> iterator;

    // Allocate the C++ type into the converter's memory block, and assign
    // its handle to the converter's convertible variable.  The C++
    // container is populated by passing the begin and end iterators of
    // the python object to the container's constructor.
    new (storage) Container(iterator(python::object(handle)),  // begin
                            iterator());                       // end
    data->convertible = storage;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_UTILS_VECTOR_CONVERTER_HPP_
