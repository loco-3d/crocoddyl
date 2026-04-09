#include "colmpc/free-fwddyn.hpp"

#include "colmpc/python.hpp"

namespace colmpc {

namespace python {

namespace bp = boost::python;

void exposeDynamicFreeFwd() {
  bp::register_ptr_to_python<
      std::shared_ptr<DifferentialActionModelFreeFwdDynamics>>();

  bp::class_<DifferentialActionModelFreeFwdDynamics,
             bp::bases<crocoddyl::DifferentialActionModelFreeFwdDynamics>>(
      "DifferentialActionModelFreeFwdDynamics",
      "DifferentialActionModelFreeFwdDynamics base class with collision "
      "calculations.",
      bp::init<
          std::shared_ptr<StateMultibody>,
          std::shared_ptr<crocoddyl::ActuationModelAbstract>,
          std::shared_ptr<crocoddyl::CostModelSum>,
          bp::optional<std::shared_ptr<crocoddyl::ConstraintModelManager>>>(
          bp::args("self", "state", "actuation", "costs", "constraints"),
          "Initialize the free forward-dynamics action model.\n\n"
          ":param state: multibody state\n"
          ":param actuation: abstract actuation model\n"
          ":param costs: stack of cost functions\n"
          ":param constraints: stack of constraint functions"))
      //   .add_property("geometry",
      //   bp::make_function(&DifferentialActionModelFreeFwdDynamics::get_geometry()))
      ;
}

}  // namespace python
}  // namespace colmpc
