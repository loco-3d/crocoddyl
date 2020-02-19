

namespace crocoddyl {
  //Actuation
  template<typename Scalar> class ActuationModelFloatingBaseTpl;
  template<typename Scalar> class ActuationModelFullTpl;
  //Contacts
  template<typename Scalar> class ContactModelAbstractTpl;
  template<typename Scalar> class ContactDataAbstractTpl;
  //Actions
  template<typename Scalar> class ActionModelImpulseFwdDynamicsTpl;
  template<typename Scalar> class ActionDataImpulseFwdDynamicsTpl;
  //Diffs
  template<typename Scalar> class DifferentialActionModelFreeFwdDynamicsTpl;  
  template<typename Scalar> class DifferentialActionDataFreeFwdDynamicsTpl;
  template<typename Scalar> class DifferentialActionModelContactFwdDynamicsTpl;
  template<typename Scalar> class DifferentialActionDataContactFwdDynamicsTpl;  
  //Frames
  template<typename Scalar> class FrameTranslationTpl;
  template<typename Scalar> class FrameRotationTpl;
  template<typename Scalar> class FramePlacementTpl;
  template<typename Scalar> class FrameMotionTpl;
  template<typename Scalar> class FrameForceTpl;
  //Costs
  template<typename Scalar> class CostModelAbstractTpl;
  template<typename Scalar> class CostDataAbstractTpl;
  template<typename Scalar> class CostModelFrameTranslationTpl;
  template<typename Scalar> class CostDataFrameTranslationTpl;
  template<typename Scalar> class CostItemTpl;
  template<typename Scalar> class CostModelSumTpl;
  template<typename Scalar> class CostDataSumTpl;
  template<typename Scalar> class CostModelCentroidalMomentumTpl;
  template<typename Scalar> class CostDataCentroidalMomentumTpl;
  template<typename Scalar> class CostModelCoMPositionTpl;
  template<typename Scalar> class CostDataCoMPositionTpl;
  template<typename Scalar> class CostModelFramePlacementTpl;
  template<typename Scalar> class CostDataFramePlacementTpl;
  template<typename Scalar> class CostModelImpulseCoMTpl;
  template<typename Scalar> class CostDataImpulseCoMTpl;  
  template<typename Scalar> class CostModelStateTpl;
  template<typename Scalar> class CostDataStateTpl;
  template<typename Scalar> class CostModelFrameVelocityTpl;
  template<typename Scalar> class CostDataFrameVelocityTpl;
  template<typename Scalar> class CostModelContactFrictionConeTpl;
  template<typename Scalar> class CostDataContactFrictionConeTpl;
  template<typename Scalar> class CostModelContactForceTpl;  
  template<typename Scalar> class CostDataContactForceTpl;
  template<typename Scalar> class CostModelControlTpl;
  template<typename Scalar> class CostModelFrameRotationTpl;
  template<typename Scalar> class CostDataFrameRotationTpl;
  //Impulses
  template<typename Scalar> class ImpulseModelAbstractTpl;
  template<typename Scalar> class ImpulseDataAbstractTpl;
  //Contacts
  template<typename Scalar> class ContactItemTpl;
  template<typename Scalar> class ContactModelMultipleTpl;
  template<typename Scalar> class ContactDataMultipleTpl;
  template<typename Scalar> class ContactModel3DTpl;
  template<typename Scalar> class ContactData3DTpl;
  template<typename Scalar> class ContactModel6DTpl;
  template<typename Scalar> class ContactData6DTpl;
  //Friction
  template<typename Scalar> class FrictionConeTpl;
  //States
  template<typename Scalar> class StateMultibodyTpl;
  //DataCollector
  template<typename Scalar> class DataCollectorMultibodyTpl;
  template<typename Scalar> class DataCollectorActMultibodyTpl;
  template<typename Scalar> class DataCollectorContactTpl;
  template<typename Scalar> class DataCollectorMultibodyInContactTpl;
  template<typename Scalar> class DataCollectorActMultibodyInContactTpl;
  template<typename Scalar> class DataCollectorImpulseTpl;
  template<typename Scalar> class DataCollectorMultibodyInImpulseTpl;
  //Impulses
  template<typename Scalar> class ImpulseModel6DTpl;
  template<typename Scalar> class ImpulseData6DTpl;
  template<typename Scalar> class ImpulseModel3DTpl;
  template<typename Scalar> class ImpulseData3DTpl;
  template<typename Scalar> class ImpulseItemTpl;
  template<typename Scalar> class ImpulseModelMultipleTpl;
  template<typename Scalar> class ImpulseDataMultipleTpl;



  /*******************************Template Instantiation**************************/
  
  typedef ActuationModelFloatingBaseTpl<double> ActuationModelFloatingBase;
  typedef ActuationModelFullTpl<double> ActuationModelFull;

  typedef ContactModelAbstractTpl<double> ContactModelAbstract;
  typedef ContactDataAbstractTpl<double> ContactDataAbstract;

  typedef ActionModelImpulseFwdDynamicsTpl<double> ActionModelImpulseFwdDynamics;
  typedef ActionDataImpulseFwdDynamicsTpl<double> ActionDataImpulseFwdDynamics;

  typedef DifferentialActionModelFreeFwdDynamicsTpl<double> DifferentialActionModelFreeFwdDynamics;  
  typedef DifferentialActionDataFreeFwdDynamicsTpl<double> DifferentialActionDataFreeFwdDynamics;
  typedef DifferentialActionModelContactFwdDynamicsTpl<double> DifferentialActionModelContactFwdDynamics;
  typedef DifferentialActionDataContactFwdDynamicsTpl<double> DifferentialActionDataContactFwdDynamics;  

  typedef FrameTranslationTpl<double> FrameTranslation;
  typedef FrameRotationTpl<double> FrameRotation;
  typedef FramePlacementTpl<double> FramePlacement;
  typedef FrameMotionTpl<double> FrameMotion;
  typedef FrameForceTpl<double> FrameForce;

  typedef CostModelAbstractTpl<double> CostModelAbstract;
  typedef CostDataAbstractTpl<double> CostDataAbstract;
  typedef CostModelFrameTranslationTpl<double> CostModelFrameTranslation;
  typedef CostDataFrameTranslationTpl<double> CostDataFrameTranslation;
  typedef CostItemTpl<double> CostItem;
  typedef CostModelSumTpl<double> CostModelSum;
  typedef CostDataSumTpl<double> CostDataSum;
  typedef CostModelCentroidalMomentumTpl<double> CostModelCentroidalMomentum;
  typedef CostDataCentroidalMomentumTpl<double> CostDataCentroidalMomentum;
  typedef CostModelCoMPositionTpl<double> CostModelCoMPosition;
  typedef CostDataCoMPositionTpl<double> CostDataCoMPosition;
  typedef CostModelFramePlacementTpl<double> CostModelFramePlacement;
  typedef CostDataFramePlacementTpl<double> CostDataFramePlacement;
  typedef CostModelImpulseCoMTpl<double> CostModelImpulseCoM;
  typedef CostDataImpulseCoMTpl<double> CostDataImpulseCoM;  
  typedef CostModelStateTpl<double> CostModelState;
  typedef CostDataStateTpl<double> CostDataState;
  typedef CostModelFrameVelocityTpl<double> CostModelFrameVelocity;
  typedef CostDataFrameVelocityTpl<double> CostDataFrameVelocity;
  typedef CostModelContactFrictionConeTpl<double> CostModelContactFrictionCone;
  typedef CostDataContactFrictionConeTpl<double> CostDataContactFrictionCone;
  typedef CostModelContactForceTpl<double> CostModelContactForce;  
  typedef CostDataContactForceTpl<double> CostDataContactForce;
  typedef CostModelControlTpl<double> CostModelControl;
  typedef CostModelFrameRotationTpl<double> CostModelFrameRotation;

  typedef CostDataFrameRotationTpl<double> CostDataFrameRotation;

  typedef ImpulseModelAbstractTpl<double> ImpulseModelAbstract;
  typedef ImpulseDataAbstractTpl<double> ImpulseDataAbstract;

  typedef ContactItemTpl<double> ContactItem;
  typedef ContactModelMultipleTpl<double> ContactModelMultiple;
  typedef ContactDataMultipleTpl<double> ContactDataMultiple;
  typedef ContactModel3DTpl<double> ContactModel3D;
  typedef ContactData3DTpl<double> ContactData3D;
  typedef ContactModel6DTpl<double> ContactModel6D;
  typedef ContactData6DTpl<double> ContactData6D;

  typedef FrictionConeTpl<double> FrictionCone;

  typedef StateMultibodyTpl<double> StateMultibody;

  typedef DataCollectorMultibodyTpl<double> DataCollectorMultibody;
  typedef DataCollectorActMultibodyTpl<double> DataCollectorActMultibody;
  typedef DataCollectorContactTpl<double> DataCollectorContact;
  typedef DataCollectorMultibodyInContactTpl<double> DataCollectorMultibodyInContact;
  typedef DataCollectorActMultibodyInContactTpl<double> DataCollectorActMultibodyInContact;
  typedef DataCollectorImpulseTpl<double> DataCollectorImpulse;
  typedef DataCollectorMultibodyInImpulseTpl<double> DataCollectorMultibodyInImpulse;

  typedef ImpulseModel6DTpl<double> ImpulseModel6D;
  typedef ImpulseData6DTpl<double> ImpulseData6D;
  typedef ImpulseModel3DTpl<double> ImpulseModel3D;
  typedef ImpulseData3DTpl<double> ImpulseData3D;
  typedef ImpulseItemTpl<double> ImpulseItem;
  typedef ImpulseModelMultipleTpl<double> ImpulseModelMultiple;
  typedef ImpulseDataMultipleTpl<double> ImpulseDataMultiple;

}
