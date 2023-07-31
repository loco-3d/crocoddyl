///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2023-2023, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_UTILS_COPYABLE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_UTILS_COPYABLE_HPP_

#include <boost/python.hpp>

namespace crocoddyl {
namespace python {
namespace bp = boost::python;

///
/// \brief Add the Python method copy to allow a copy of this by calling the
/// copy constructor.
///
template <class C>
struct CopyableVisitor : public bp::def_visitor<CopyableVisitor<C> > {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def("copy", &copy, bp::arg("self"), "Returns a copy of *this.");
    cl.def("__copy__", &copy, bp::arg("self"), "Returns a copy of *this.");
    cl.def("__deepcopy__", &deepcopy, bp::args("self", "memo"),
           "Returns a deep copy of *this.");
  }

 private:
  static C copy(const C& self) { return C(self); }
  static C deepcopy(const C& self, bp::dict) { return C(self); }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_UTILS_COPYABLE_HPP_
