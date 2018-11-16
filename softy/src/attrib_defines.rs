use crate::geo::math::Matrix3;

pub(crate) const FIXED_ATTRIB: &'static str = "fixed";
pub(crate) const DISPLACEMENT_ATTRIB: &'static str = "displacement";
pub(crate) const REFERENCE_POSITION_ATTRIB: &'static str = "ref";
pub(crate) const REFERENCE_VOLUME_ATTRIB: &'static str = "ref_volume";
pub(crate) const REFERENCE_SHAPE_MATRIX_INV_ATTRIB: &'static str = "ref_shape_mtx_inv";
pub(crate) const STRAIN_ENERGY_ATTRIB: &'static str = "strain_energy";
pub(crate) const ELASTIC_FORCE_ATTRIB: &'static str = "elastic_force";

pub(crate) type FixedIntType = i8;
pub(crate) type DispType = [f64;3];
pub(crate) type RefPosType = [f64;3];
pub(crate) type RefVolType = f64;
pub(crate) type RefShapeMtxInvType= Matrix3<f64>;
pub(crate) type StrainEnergyType = f64;
pub(crate) type ElasticForceType = [f64;3];
