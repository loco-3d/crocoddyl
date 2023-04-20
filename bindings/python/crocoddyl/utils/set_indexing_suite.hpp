///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_UTILS_SET_INDEXING_SUITE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_UTILS_SET_INDEXING_SUITE_HPP_

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace crocoddyl {
namespace python {

template <typename Container, bool NoProxy, typename DerivedPolicies>
class set_indexing_suite;

namespace detail {
template <typename Container, bool NoProxy>
class final_set_derived_policies
    : public crocoddyl::python::set_indexing_suite<
          Container, NoProxy, final_set_derived_policies<Container, NoProxy> > {
};
}  // namespace detail

// The set_indexing_suite class is a predefined indexing_suite derived
// class for wrapping std::set (and std::set like) classes. It provides
// all the policies required by the indexing_suite (see indexing_suite).
// Example usage:
//
//  class X {...};
//
//  ...
//
//      class_<std::set<X> >("XSet")
//          .def(set_indexing_suite<std::set<X> >())
//      ;
//
// By default indexed elements are returned by proxy. This can be
// disabled by supplying *true* in the NoProxy template parameter.
//
template <typename Container, bool NoProxy = false,
          typename DerivedPolicies =
              detail::final_set_derived_policies<Container, NoProxy> >
class set_indexing_suite
    : public boost::python::vector_indexing_suite<Container, NoProxy,
                                                  DerivedPolicies> {
  typedef boost::python::vector_indexing_suite<Container, NoProxy,
                                               DerivedPolicies>
      base;

 public:
  typedef typename base::data_type data_type;
  typedef typename base::index_type index_type;
  typedef typename base::key_type key_type;

  template <class Class>
  static void extension_def(Class& class_) {
    class_.def("add", &function<DerivedPolicies::add>)
        .def("remove", &function<DerivedPolicies::remove>)
        .def("discard", &function<DerivedPolicies::discard>)
        .def("clear", &DerivedPolicies::clear);
  }

  static bool contains(Container& container, key_type const& key) {
    return container.find(key) != container.end();
  }

  static void add(Container& container, data_type const& v) {
    container.insert(v);
  }

  static void discard(Container& container, data_type const& v) {
    container.erase(v);
  }

  static void remove(Container& container, data_type const& v) {
    if (!container.erase(v)) {
      PyErr_SetString(PyExc_KeyError, "Element doesn't exist");
      boost::python::throw_error_already_set();
    }
  }

  static void clear(Container& container) { container.clear(); }

  static data_type get_item(Container& container, index_type i) {
    return *std::next(container.begin(), i);
  }

  static void set_item(Container&, index_type, data_type const&) {
    not_supported();
  }

  static void delete_item(Container& container, index_type i) {
    container.erase(advance(container.begin(), i));
  }

  static boost::python::object get_slice(Container& container, index_type from,
                                         index_type to) {
    if (from > to) return boost::python::object(Container());

    auto s = slice(container, from, to);
    return boost::python::object(Container(s.first, s.second));
  }

  static void set_slice(Container&, index_type, index_type, data_type const&) {
    not_supported();
  }

  template <typename Iter>
  static void set_slice(Container&, index_type, index_type, Iter, Iter) {
    not_supported();
  }

  static void delete_slice(Container& container, index_type from,
                           index_type to) {
    if (to >= from) {
      auto s = slice(container, from, to);
      container.erase(s.first, s.second);
    }
  }

 private:
  static typename Container::iterator advance(
      typename Container::iterator it, typename Container::difference_type i) {
    return std::advance(it, i), it;
  }

  static std::pair<typename Container::iterator, typename Container::iterator>
  slice(Container& container, index_type from, index_type to) {
    BOOST_ASSERT(to >= from);
    std::pair<typename Container::iterator, typename Container::iterator> s;
    s.first = container.begin();
    std::advance(s.first, from);
    s.second = s.first;
    std::advance(s.second, to - from);
    return s;
  }

  template <void (*fn)(Container&, data_type const&)>
  static void function(Container& container, const boost::python::object v) {
    using namespace boost::python;
    extract<data_type&> elemRef(v);

    if (elemRef.check()) {
      fn(container, elemRef());
    } else {
      extract<data_type> elem(v);
      if (elem.check()) {
        fn(container, elem());
      } else {
        PyErr_SetString(PyExc_TypeError, "Invalid type");
        throw_error_already_set();
      }
    }
  }

  static void not_supported() {
    PyErr_SetString(PyExc_TypeError,
                    "__setitem__ not supported for set object");
    boost::python::throw_error_already_set();
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_UTILS_SET_INDEXING_SUITE_HPP_
