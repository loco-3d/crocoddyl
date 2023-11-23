///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2023-2023, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

void exposeException() {
  bp::class_<Exception> ExceptionClass(
      "Exception", bp::init<std::string, const char *, const char *, int>(
                       bp::args("self", "msg", "file", "func", "line"),
                       "Initialize the Crocoddyl's exception."));
  ExceptionClass.add_property("message", &Exception::getMessage)
      .add_property("extra_data", &Exception::getExtraData);

  ExceptionType = createExceptionClass("Exception");
  bp::register_exception_translator<Exception>(&translateException);
}

}  // namespace python
}  // namespace crocoddyl
