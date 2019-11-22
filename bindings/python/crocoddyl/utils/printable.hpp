///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_UTILS_PRINTABLE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_UTILS_PRINTABLE_HPP_

#include <boost/python.hpp>

namespace crocoddyl {
namespace python {
namespace bp = boost::python;

///
/// \brief Set the Python method __str__ and __repr__ to use the overloading operator<<.
///
template <class C>
struct PrintableVisitor : public bp::def_visitor<PrintableVisitor<C> > {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::self_ns::str(bp::self_ns::self)).def(bp::self_ns::repr(bp::self_ns::self));
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_UTILS_PRINTABLE_HPP_