#include "colmpc/multibody.hpp"
#include "colmpc/python.hpp"

namespace colmpc {
template class StateMultibodyTpl<double>;

namespace python {

namespace bp = boost::python;

void exposeStateMultibody() {
  bp::register_ptr_to_python<std::shared_ptr<StateMultibody>>();

  bp::class_<StateMultibody, bp::bases<crocoddyl::StateMultibody>>(
      "StateMultibody",
      bp::init<std::shared_ptr<pinocchio::Model>,
               std::shared_ptr<pinocchio::GeometryModel>>(
          bp::args("self", "pinocchioModel", "geometryModel"),
          "Initialize the multibody state given a Pinocchio model.\n\n"
          ":param pinocchioModel: pinocchio model (i.e. multibody model)")
          [bp::with_custodian_and_ward<1, 2>()])
      .add_property(
          "geometry",
          bp::make_function(&StateMultibody::get_geometry,
                            bp::return_value_policy<bp::return_by_value>()));
}

}  // namespace python
}  // namespace colmpc
