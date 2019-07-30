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
#include <map>

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

/// @note Registers converter from a provided type to the python
///       iterable type to the.
template <class K, class V>
struct map_to_dict {
  static PyObject* convert(const std::map<K, V>& map) {
    bp::dict* dict = new boost::python::dict();
    typename std::map<K, V>::const_iterator it;
    for (it = map.begin(); it != map.end(); ++it) {
      dict->setdefault(it->first, boost::ref(it->second));
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

#endif  // PYTHON_CROCODDYL_UTILS_HPP_