///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_UTILS_MAP_CONVERTER_HPP_
#define BINDINGS_PYTHON_CROCODDYL_UTILS_MAP_CONVERTER_HPP_

#include <Eigen/Dense>
#include <map>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/to_python_converter.hpp>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

/// @note Registers converter from a provided type to the python
///       iterable type to the.
template <class K, class V, bool NoProxy = true>
struct map_to_dict {
  static PyObject* convert(const std::map<K, V>& map) {
    bp::dict* dict = new boost::python::dict();
    typename std::map<K, V>::const_iterator it;
    for (it = map.begin(); it != map.end(); ++it) {
      if (NoProxy) {
        dict->setdefault(it->first, boost::ref(it->second));
      } else {
        dict->setdefault(it->first, it->second);
      }
    }
    return dict->ptr();
  }
  static PyTypeObject const* get_pytype() { return &PyDict_Type; }
};

/// Conversion from dict to map solution proposed in
/// https://stackoverflow.com/questions/6116345/boostpython-possible-to-automatically-convert-from-dict-stdmap
/// This template encapsulates the conversion machinery.
template <typename key_t, typename val_t>
struct dict_to_map {
  /// The type of the map we convert the Python dict into
  typedef std::map<key_t, val_t> map_t;

  dict_to_map& from_python() {
    boost::python::converter::registry::push_back(&dict_to_map::convertible, &dict_to_map::construct,
                                                  boost::python::type_id<map_t>());
    // Support chaining.
    return *this;
  }

  /// Check if conversion is possible
  static void* convertible(PyObject* object) { return PyObject_GetIter(object) ? object : NULL; }

  /// Perform the conversion
  static void construct(PyObject* objptr, boost::python::converter::rvalue_from_python_stage1_data* data) {
    // convert the PyObject pointed to by `objptr` to a boost::python::dict
    boost::python::handle<> objhandle(boost::python::borrowed(objptr));  // "smart ptr"
    boost::python::dict d(objhandle);

    // get a pointer to memory into which we construct the map
    // this is provided by the Python runtime
    void* storage =
        reinterpret_cast<boost::python::converter::rvalue_from_python_storage<map_t>*>(data)->storage.bytes;

    // placement-new allocate the result
    new (storage) map_t();

    // iterate over the dictionary `d`, fill up the map `m`
    map_t& m(*(static_cast<map_t*>(storage)));
    boost::python::list keys(d.keys());
    int keycount(static_cast<int>(boost::python::len(keys)));
    for (int i = 0; i < keycount; ++i) {
      // get the key
      boost::python::object keyobj(keys[i]);
      boost::python::extract<key_t> keyproxy(keyobj);
      if (!keyproxy.check()) {
        PyErr_SetString(PyExc_KeyError, "Bad key type");
        boost::python::throw_error_already_set();
      }
      key_t key = keyproxy();

      // get the corresponding value
      boost::python::object valobj(d[keyobj]);
      boost::python::extract<val_t> valproxy(valobj);
      if (!valproxy.check()) {
        PyErr_SetString(PyExc_ValueError, "Bad value type");
        boost::python::throw_error_already_set();
      }
      val_t val = valproxy();
      m[key] = val;
    }

    // remember the location for later
    data->convertible = storage;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_UTILS_MAP_CONVERTER_HPP_
