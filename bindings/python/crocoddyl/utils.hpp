///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_UTILS_HPP_
#define PYTHON_CROCODDYL_UTILS_HPP_

#include <boost/python/stl_iterator.hpp>
#include <boost/python/to_python_converter.hpp>
#include <Eigen/Dense>
#include <vector>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

template <typename T>
struct is_pointer {
  static const bool value = false;
};

template <typename T>
struct is_pointer<T*> {
  static const bool value = true;
};

template <class T>
bp::list std_vector_to_python_list(const std::vector<T>& vec) {
  const long unsigned int& n = vec.size();
  bp::list list;
  for (unsigned int i = 0; i < n; ++i) {
    if (is_pointer<T>::value) {
      list.append(boost::ref(vec[i]));
    } else {
      list.append(vec[i]);
    }
  }
  return list;
}

template <class T>
std::vector<T> python_list_to_std_vector(const bp::list& list) {
  const long int& n = len(list);
  std::vector<T> vec;
  vec.resize(n);
  for (int i = 0; i < n; ++i) {
    vec[i] = bp::extract<T>(list[i]);
  }
  return vec;
}

/// @note Registers converter from a provided type to the python
///       iterable type to the.
template <class T>
struct vector_to_list {
  static PyObject* convert(const std::vector<T>& vec) {
    bp::list* l = new boost::python::list();
    for (size_t i = 0; i < vec.size(); i++) {
      if (is_pointer<T>::value) {
        l->append(boost::ref(vec[i]));
      } else {
        l->append(vec[i]);
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

#endif  // PYTHON_CROCODDYL_UTILS_HPP_