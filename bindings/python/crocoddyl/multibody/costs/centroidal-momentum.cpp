///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/centroidal-momentum.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostCentroidalMomentum() {  // TODO: Remove once the deprecated update call has been removed in a future
                                       // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  typedef Eigen::Matrix<double, 6, 1> Vector6d;

  bp::register_ptr_to_python<boost::shared_ptr<CostModelCentroidalMomentum> >();

  bp::class_<CostModelCentroidalMomentum, bp::bases<CostModelResidual> >(
      "CostModelCentroidalMomentum",
      "This cost function defines a residual vector as r = h - href, with h and href as the current and reference "
      "centroidal momenta, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, Vector6d, int>(
          bp::args("self", "state", "activation", "href", "nu"),
          "Initialize the centroidal momentum cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param href: reference centroidal momentum\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, Vector6d>(
          bp::args("self", "state", "activation", "href"),
          "Initialize the centroidal momentum cost model.\n\n"
          "The default nu is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param href: reference centroidal momentum"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Vector6d, int>(
          bp::args("self", "state", "href", "nu"),
          "Initialize the centroidal momentum cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          ":param state: state of the multibody system\n"
          ":param href: reference centroidal momentum\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, Vector6d>(
          bp::args("self", "state", "href"),
          "Initialize the centroidal momentum cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param href: reference centroidal momentum"))
      .add_property("reference", &CostModelCentroidalMomentum::get_reference<MathBaseTpl<double>::Vector6s>,
                    &CostModelCentroidalMomentum::set_reference<MathBaseTpl<double>::Vector6s>,
                    "reference centroidal momentum")
      .add_property("href",
                    bp::make_function(&CostModelCentroidalMomentum::get_reference<MathBaseTpl<double>::Vector6s>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelCentroidalMomentum::set_reference<MathBaseTpl<double>::Vector6s>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference centroidal momentum");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl
