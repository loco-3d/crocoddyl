///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <crocoddyl/core/action-base.hpp>
#include <pinocchio/codegen/cppadcg.hpp>

namespace crocoddyl {
  
template <typename _Scalar>
struct CGActionAbstract {
public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionModelAbstractTpl<Scalar> BaseModel;
  typedef ActionDataAbstractTpl<Scalar> BaseData;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  
  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef MathBaseTpl<ADScalar> ADMathBase;
  typedef ActionModelAbstractTpl<ADScalar> ADBaseModel;
  typedef ActionDataAbstractTpl<ADScalar> ADBaseData;
  typedef typename ADMathBase::VectorXs ADVectorXs;
  typedef typename ADMathBase::MatrixXs ADMatrixXs;
  
  CGActionModelAbstract(boost::shared_ptr<BaseModel> base_model,
                        boost::shared_ptr<BaseData> base_data)
    : ad_base_model(boost::make_shared<ADBaseModel> (base_model->template cast<ADScalar>()))
    , ad_base_data(ad_base_model->createData())
    , build_forward(true)
    , build_jacobian(false)
  {
    ad_X = ADVectorXs(base_model->get_state()->get_nx()+base_model->get_nu()); //input dim
    ad_Y = ADVectorXs(base_model->get_state()->get_nx()); //output dim
  }
  
  
  void buildCalc()
  {
    const std::size_t& nx = base_model->get_state()->get_nx();
    const std::size_t& nu = base_model->get_nu();
    assert(ad_X.size() == nx+nu);

    CppAD::Independent(ad_X);
    ad_model->calc(ad_data, ad_X.head(nx), ad_X.tail(nu));
    ad_fun.Dependent(ad_X, ad_data->xnext);
    ad_fun.optimize("no_compare_op");
  }
  
  void initLib()
  {
    buildCalc();
    //buildCalcDiff();
    // generates source code
    cgen_ptr = std::unique_ptr<CppAD::cg::ModelCSourceGen<Scalar> >(new CppAD::cg::ModelCSourceGen<Scalar>(ad_fun, function_name));
    cgen_ptr->setCreateForwardZero(build_forward);
    cgen_ptr->setCreateJacobian(build_jacobian);
    libcgen_ptr = std::unique_ptr<CppAD::cg::ModelLibraryCSourceGen<Scalar> >(new CppAD::cg::ModelLibraryCSourceGen<Scalar>(*cgen_ptr));
    
    dynamicLibManager_ptr
      = std::unique_ptr<CppAD::cg::DynamicModelLibraryProcessor<Scalar> >(new CppAD::cg::DynamicModelLibraryProcessor<Scalar>(*libcgen_ptr,library_name));
  }

  CppAD::cg::ModelCSourceGen<Scalar> & codeGenerator()
  { return *cgen_ptr; }

  void compileLib()
  {
    CppAD::cg::GccCompiler<Scalar> compiler;
    std::vector<std::string> compile_options = compiler.getCompileFlags();
    compile_options[0] = "-Ofast";
    compiler.setCompileFlags(compile_options);
    dynamicLibManager_ptr->createDynamicLibrary(compiler,false);
  }

  bool existLib() const
  {
    const std::string filename = dynamicLibManager_ptr->getLibraryName() + CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION;
    std::ifstream file(filename.c_str());
    return file.good();
  }

  void loadLib(const bool generate_if_not_exist = true)
  {
    if(not existLib() && generate_if_not_exist)
      compileLib();
    
    const auto it = dynamicLibManager_ptr->getOptions().find("dlOpenMode");
    if (it == dynamicLibManager_ptr->getOptions().end())
      {
        dynamicLib_ptr.reset(new CppAD::cg::LinuxDynamicLib<Scalar>(dynamicLibManager_ptr->getLibraryName() +  CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION));
      }
    else
      {
        int dlOpenMode = std::stoi(it->second);
        dynamicLib_ptr.reset(new CppAD::cg::LinuxDynamicLib<Scalar>(dynamicLibManager_ptr->getLibraryName() +  CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION, dlOpenMode));
      }
    
    generatedFun_ptr = dynamicLib_ptr->model(function_name.c_str());
  }
  
  void evalCalc(const VectorXs& x, const VectorXs& u)
  {
    assert(build_forward);
    const std::size_t& nx = base_model->get_state()->get_nx();
    const std::size_t& nu = base_model->get_nu();
    assert(x.size() == nx);
    assert(u.size() == nu);

    X_in.head(nx) = x;
    X_in.tail(nu) = u;
    generatedFun_ptr->ForwardZero(X_in,base_data->xnext);
  }
  

  /// \brief Dimension of the input vector
  Eigen::DenseIndex getInputDimension() const { return ad_X.size(); }
  /// \brief Dimension of the output vector
  Eigen::DenseIndex getOutputDimension() const { return ad_Y.size(); }
};

} //namespace crocoddyl
