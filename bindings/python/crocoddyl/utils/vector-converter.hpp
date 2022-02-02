///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_UTILS_VECTOR_CONVERTER_HPP_
#define BINDINGS_PYTHON_CROCODDYL_UTILS_VECTOR_CONVERTER_HPP_

#include <vector>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/to_python_converter.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

/**
 * @brief Create a pickle interface for the std::vector
 *
 * @param[in] Container  Vector type to be pickled
 * \sa Pickle
 */
template <typename Container>
struct PickleVector : bp::pickle_suite {
  static bp::tuple getinitargs(const Container&) { return bp::make_tuple(); }
  static bp::tuple getstate(bp::object op) { return bp::make_tuple(bp::list(bp::extract<const Container&>(op)())); }
  static void setstate(bp::object op, bp::tuple tup) {
    Container& o = bp::extract<Container&>(op)();
    bp::stl_input_iterator<typename Container::value_type> begin(tup[0]), end;
    o.insert(o.begin(), begin, end);
  }
};

/** @brief Type that allows for registration of conversions from python iterable types. */
template <typename Container>
struct list_to_vector {
  /** @note Registers converter from a python iterable type to the provided type. */
  static void register_converter() {
    bp::converter::registry::push_back(&list_to_vector::convertible, &list_to_vector::construct,
                                       bp::type_id<Container>());
  }

  /** @brief Check if PyObject is iterable. */
  static void* convertible(PyObject* object) {
    // Check if it is a list
    if (!PyList_Check(object)) return 0;

    // Retrieve the underlying list
    bp::object bp_obj(bp::handle<>(bp::borrowed(object)));
    bp::list bp_list(bp_obj);
    bp::ssize_t list_size = bp::len(bp_list);

    // Check if all the elements contained in the current vector is of type T
    for (bp::ssize_t k = 0; k < list_size; ++k) {
      bp::extract<typename Container::value_type> elt(bp_list[k]);
      if (!elt.check()) return 0;
    }
    return object;
  }

  /** @brief Convert iterable PyObject to C++ container type.
   *
   * Container Concept requirements:
   *    * Container::value_type is CopyConstructable.
   *    * Container can be constructed and populated with two iterators.
   * i.e. Container(begin, end)
   */
  static void construct(PyObject* object, bp::converter::rvalue_from_python_stage1_data* data) {
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

  static bp::list tolist(Container& self) {
    typedef bp::iterator<Container> iterator;
    bp::list list(iterator()(self));
    return list;
  }
};

/**
 * @brief Expose an std::vector from a type given as template argument.
 *
 * @param[in] T          Type to expose as std::vector<T>.
 * @param[in] Allocator  Type for the Allocator in std::vector<T,Allocator>.
 * @param[in] NoProxy    When set to false, the elements will be copied when returned to Python.
 */
template <class T, class Allocator = std::allocator<T>, bool NoProxy = false>
struct StdVectorPythonVisitor : public bp::vector_indexing_suite<typename std::vector<T, Allocator>, NoProxy>,
                                public list_to_vector<std::vector<T, Allocator> > {
  typedef std::vector<T, Allocator> Container;
  typedef list_to_vector<Container> FromPythonListConverter;

  static void expose(const std::string& class_name, const std::string& doc_string = "") {
    namespace bp = bp;

    bp::class_<Container>(class_name.c_str(), doc_string.c_str())
        .def(StdVectorPythonVisitor())
        .def("tolist", &FromPythonListConverter::tolist, bp::arg("self"), "Returns the std::vector as a Python list.")
        .def_pickle(PickleVector<Container>());
    // Register conversion
    FromPythonListConverter::register_converter();
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_UTILS_VECTOR_CONVERTER_HPP_
