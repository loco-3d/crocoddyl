#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_FWD_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_FWD_HPP_

#define PYTHON_BINDINGS

#include <eigenpy/eigenpy.hpp>
#include <boost/python.hpp>
#include <boost/python/enum.hpp>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeCore();
void exposeMultibody();

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_FWD_HPP_