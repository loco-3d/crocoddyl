///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_UTILS_MAP_CONVERTER_HPP_
#define BINDINGS_PYTHON_CROCODDYL_UTILS_MAP_CONVERTER_HPP_

#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/to_python_converter.hpp>

#include "python/crocoddyl/utils/vector-converter.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

/**
 * @brief Create a pickle interface for the std::map
 *
 * @param[in] Container  Map type to be pickled
 * \sa Pickle
 */
template <typename Container>
struct PickleMap : public eigenpy::PickleVector<Container> {
  static void setstate(bp::object op, bp::tuple tup) {
    Container& o = bp::extract<Container&>(op)();
    bp::stl_input_iterator<typename Container::value_type> begin(tup[0]), end;
    o.insert(begin, end);
  }
};

/// Conversion from dict to map solution proposed in
/// https://stackoverflow.com/questions/6116345/boostpython-possible-to-automatically-convert-from-dict-stdmap
/// This template encapsulates the conversion machinery.
template <typename Container>
struct dict_to_map {
  static void register_converter() {
    bp::converter::registry::push_back(&dict_to_map::convertible,
                                       &dict_to_map::construct,
                                       bp::type_id<Container>());
  }

  /// Check if conversion is possible
  static void* convertible(PyObject* object) {
    // Check if it is a list
    if (!PyObject_GetIter(object)) return 0;
    return object;
  }

  /// Perform the conversion
  static void construct(PyObject* object,
                        bp::converter::rvalue_from_python_stage1_data* data) {
    // convert the PyObject pointed to by `object` to a bp::dict
    bp::handle<> handle(bp::borrowed(object));  // "smart ptr"
    bp::dict dict(handle);

    // get a pointer to memory into which we construct the map
    // this is provided by the Python runtime
    typedef bp::converter::rvalue_from_python_storage<Container> storage_type;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

    // placement-new allocate the result
    new (storage) Container();

    // iterate over the dictionary `dict`, fill up the map `map`
    Container& map(*(static_cast<Container*>(storage)));
    bp::list keys(dict.keys());
    int keycount(static_cast<int>(bp::len(keys)));
    for (int i = 0; i < keycount; ++i) {
      // get the key
      bp::object keyobj(keys[i]);
      bp::extract<typename Container::key_type> keyproxy(keyobj);
      if (!keyproxy.check()) {
        PyErr_SetString(PyExc_KeyError, "Bad key type");
        bp::throw_error_already_set();
      }
      typename Container::key_type key = keyproxy();

      // get the corresponding value
      bp::object valobj(dict[keyobj]);
      bp::extract<typename Container::mapped_type> valproxy(valobj);
      if (!valproxy.check()) {
        PyErr_SetString(PyExc_ValueError, "Bad value type");
        bp::throw_error_already_set();
      }
      typename Container::mapped_type val = valproxy();
      map[key] = val;
    }

    // remember the location for later
    data->convertible = storage;
  }

  static bp::dict todict(Container& self) {
    bp::dict dict;
    typename Container::const_iterator it;
    for (it = self.begin(); it != self.end(); ++it) {
      dict.setdefault(it->first, it->second);
    }
    return dict;
  }
};

/**
 * @brief Expose an std::map from a type given as template argument.
 *
 * @param[in] T          Type to expose as std::map<T>.
 * @param[in] Compare    Type for the Compare in std::map<T,Compare,Allocator>.
 * @param[in] Allocator  Type for the Allocator in
 * std::map<T,Compare,Allocator>.
 * @param[in] NoProxy    When set to false, the elements will be copied when
 * returned to Python.
 */
template <class Key, class T, class Compare = std::less<Key>,
          class Allocator = std::allocator<std::pair<const Key, T> >,
          bool NoProxy = false>
struct StdMapPythonVisitor
    : public bp::map_indexing_suite<
          typename std::map<Key, T, Compare, Allocator>, NoProxy>,
      public dict_to_map<std::map<Key, T, Compare, Allocator> > {
  typedef std::map<Key, T, Compare, Allocator> Container;
  typedef dict_to_map<Container> FromPythonDictConverter;

  static void expose(const std::string& class_name,
                     const std::string& doc_string = "") {
    namespace bp = bp;

    bp::class_<Container>(class_name.c_str(), doc_string.c_str())
        .def(StdMapPythonVisitor())
        .def("todict", &FromPythonDictConverter::todict, bp::arg("self"),
             "Returns the std::map as a Python dictionary.")
        .def_pickle(PickleMap<Container>());
    // Register conversion
    FromPythonDictConverter::register_converter();
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_UTILS_MAP_CONVERTER_HPP_
