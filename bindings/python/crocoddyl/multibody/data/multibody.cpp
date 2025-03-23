///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"

#include "crocoddyl/multibody/data/multibody.hpp"

namespace crocoddyl {
namespace python {

template <typename Data>
struct DataCollectorMultibodyVisitor
    : public bp::def_visitor<DataCollectorMultibodyVisitor<Data>> {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.add_property(
        "pinocchio",
        bp::make_getter(&Data::pinocchio, bp::return_internal_reference<>()),
        "pinocchio data");
  }
};

#define CROCODDYL_DATA_COLLECTOR_MULTIBODY_PYTHON_BINDINGS(Scalar)             \
  typedef DataCollectorMultibodyTpl<Scalar> Data;                              \
  typedef DataCollectorAbstractTpl<Scalar> DataBase;                           \
  typedef pinocchio::DataTpl<Scalar> PinocchioData;                            \
  bp::register_ptr_to_python<std::shared_ptr<Data>>();                         \
  bp::class_<Data, bp::bases<DataBase>>(                                       \
      "DataCollectorMultibody", "Data collector for multibody systems.\n\n",   \
      bp::init<PinocchioData*>(                                                \
          bp::args("self", "pinocchio"),                                       \
          "Create multibody data collection.\n\n"                              \
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 2>()]) \
      .def(DataCollectorMultibodyVisitor<Data>())                              \
      .def(CopyableVisitor<Data>());

#define CROCODDYL_DATA_COLLECTOR_ACTMULTIBODY_PYTHON_BINDINGS(Scalar) \
  typedef DataCollectorActMultibodyTpl<Scalar> Data;                  \
  typedef DataCollectorMultibodyTpl<Scalar> DataBase1;                \
  typedef DataCollectorActuationTpl<Scalar> DataBase2;                \
  typedef pinocchio::DataTpl<Scalar> PinocchioData;                   \
  typedef ActuationDataAbstractTpl<Scalar> ActuationData;             \
  bp::register_ptr_to_python<std::shared_ptr<Data>>();                \
  bp::class_<Data, bp::bases<DataBase1, DataBase2>>(                  \
      "DataCollectorActMultibody",                                    \
      "Data collector for actuated multibody systems.\n\n",           \
      bp::init<PinocchioData*, std::shared_ptr<ActuationData>>(       \
          bp::args("self", "pinocchio", "actuation"),                 \
          "Create multibody data collection.\n\n"                     \
          ":param pinocchio: Pinocchio data\n"                        \
          ":param actuation: actuation data")                         \
          [bp::with_custodian_and_ward<1, 2>()])                      \
      .def(CopyableVisitor<Data>());

#define CROCODDYL_DATA_COLLECTOR_JOINT_ACTMULTIBODY_PYTHON_BINDINGS(Scalar) \
  typedef DataCollectorJointActMultibodyTpl<Scalar> Data;                   \
  typedef DataCollectorActMultibodyTpl<Scalar> DataBase1;                   \
  typedef DataCollectorJointTpl<Scalar> DataBase2;                          \
  typedef pinocchio::DataTpl<Scalar> PinocchioData;                         \
  typedef ActuationDataAbstractTpl<Scalar> ActuationData;                   \
  typedef JointDataAbstractTpl<Scalar> JointData;                           \
  bp::register_ptr_to_python<std::shared_ptr<Data>>();                      \
  bp::class_<Data, bp::bases<DataBase1, DataBase2>>(                        \
      "DataCollectorJointActMultibody",                                     \
      "Data collector for actuated-joint multibody systems.\n\n",           \
      bp::init<PinocchioData*, std::shared_ptr<ActuationData>,              \
               std::shared_ptr<JointData>>(                                 \
          bp::args("self", "pinocchio", "actuation", "joint"),              \
          "Create multibody data collection.\n\n"                           \
          ":param pinocchio: Pinocchio data\n"                              \
          ":param actuation: actuation data\n"                              \
          ":param joint: joint data")[bp::with_custodian_and_ward<1, 2>()]) \
      .def(CopyableVisitor<Data>());

void exposeDataCollectorMultibody() {
  CROCODDYL_PYTHON_SCALARS(CROCODDYL_DATA_COLLECTOR_MULTIBODY_PYTHON_BINDINGS)
  CROCODDYL_PYTHON_SCALARS(
      CROCODDYL_DATA_COLLECTOR_ACTMULTIBODY_PYTHON_BINDINGS)
  CROCODDYL_PYTHON_SCALARS(
      CROCODDYL_DATA_COLLECTOR_JOINT_ACTMULTIBODY_PYTHON_BINDINGS)
}

}  // namespace python
}  // namespace crocoddyl
