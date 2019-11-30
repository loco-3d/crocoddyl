///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

CrocoddylException::CrocoddylException(std::string message) { this->message = message; }

CrocoddylException::~CrocoddylException() throw() {}

const char* CrocoddylException::what() const throw() { return this->message.c_str(); }

std::string CrocoddylException::getMessage() { return this->message; }

}  // namespace crocoddyl