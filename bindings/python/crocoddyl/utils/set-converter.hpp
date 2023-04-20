///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh, INRIA
//                          University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_UTILS_SET_CONVERTER_HPP_
#define BINDINGS_PYTHON_CROCODDYL_UTILS_SET_CONVERTER_HPP_

#include <boost/python/stl_iterator.hpp>
#include <boost/python/to_python_converter.hpp>
#include <set>

#include "set_indexing_suite.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

/**
 * @brief Create a pickle interface for the std::set
 *
 * @param[in] Container  Set type to be pickled
 * \sa Pickle
 */
template <typename Container>
struct PickleSet : bp::pickle_suite {
  static bp::tuple getinitargs(const Container&) { return bp::make_tuple(); }

  static bp::tuple getstate(bp::object op) {
    bp::list list;
    const Container& ret = bp::extract<const Container&>(op);
    for (const auto& it : ret) {
      list.append(it);
    }
    return bp::make_tuple(list);
  }

  static void setstate(bp::object op, bp::tuple tup) {
    Container& o = bp::extract<Container&>(op)();
    bp::stl_input_iterator<typename Container::value_type> begin(tup[0]), end;
    o.insert(begin, end);
  }
};

/** @brief Type that allows for registration of conversions from python iterable
 * types. */
template <typename Container>
struct set_to_set {
  /** @note Registers converter from a python iterable type to the provided
   * type. */
  static void register_converter() {
    bp::converter::registry::push_back(&set_to_set::convertible,
                                       &set_to_set::construct,
                                       bp::type_id<Container>());
  }

  /** @brief Check if PyObject is iterable. */
  static void* convertible(PyObject* object) {
    // Check if it is a set
    if (!PySet_Check(object)) return 0;

    PyObject* iter = PyObject_GetIter(object);
    PyObject* item;
    while ((item = PyIter_Next(iter))) {
      bp::extract<typename Container::value_type> elt(iter);
      // Py_DECREF(item);
      if (!elt.check()) return 0;
    }
    Py_DECREF(iter);

    return object;
  }

  /** @brief Convert iterable PyObject to C++ container type.
   *
   * Container Concept requirements:
   *    * Container::value_type is CopyConstructable.
   */
  static void construct(PyObject* object,
                        bp::converter::rvalue_from_python_stage1_data* data) {
    // Object is a borrowed reference, so create a handle indicting it is
    // borrowed for proper reference counting.
    bp::handle<> handle(bp::borrowed(object));

    // Obtain a handle to the memory block that the converter has allocated
    // for the C++ type.
    typedef bp::converter::rvalue_from_python_storage<Container> storage_type;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

    typedef bp::stl_input_iterator<typename Container::value_type> iterator;

    // Allocate the C++ type into the converter's memory block, and assign
    // its handle to the converter's convertible variable.  The C++
    // container is populated by passing the begin and end iterators of
    // the python object to the container's constructor.
    new (storage) Container(iterator(bp::object(handle)),  // begin
                            iterator());                   // end
    data->convertible = storage;
  }

  static bp::object toset(Container& self) {
    PyObject* set = PySet_New(NULL);
    for (auto it = self.begin(); it != self.end(); ++it) {
      PySet_Add(set, bp::object(*it).ptr());
    }
    return bp::object(bp::handle<>(set));
  }
};

/**
 * @brief Expose an std::set from a type given as template argument.
 *
 * @param[in] T          Type to expose as std::set<T>.
 * @param[in] Compare    Type for the comparison function in
 * std::set<T,Compare,Allocator>.
 * @param[in] Allocator  Type for the Allocator in
 * std::set<T,Compare,Allocator>.
 * @param[in] NoProxy    When set to false, the elements will be copied when
 * returned to Python.
 */
template <class T, class Compare = std::less<T>,
          class Allocator = std::allocator<T>, bool NoProxy = false>
struct StdSetPythonVisitor
    : public set_indexing_suite<typename std::set<T, Compare, Allocator>,
                                NoProxy>,
      set_to_set<std::set<T, Compare, Allocator>> {
  typedef std::set<T, Compare, Allocator> Container;
  typedef set_to_set<Container> FromPythonSetConverter;

  static void expose(const std::string& class_name,
                     const std::string& doc_string = "") {
    bp::class_<Container>(class_name.c_str(), doc_string.c_str())
        .def(StdSetPythonVisitor())
        .def("toset", &FromPythonSetConverter::toset, bp::arg("self"),
             "Returns the std::set as a Python set.")
        .def_pickle(PickleSet<Container>());
    // Register conversion
    FromPythonSetConverter::register_converter();
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_UTILS_SET_CONVERTER_HPP_
