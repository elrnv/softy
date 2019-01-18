use crate::geo::math::Matrix3;

pub(crate) const FIXED_ATTRIB: &str = "fixed";
pub(crate) const DISPLACEMENT_ATTRIB: &str = "displacement";
pub(crate) const REFERENCE_POSITION_ATTRIB: &str = "ref";
pub(crate) const REFERENCE_VOLUME_ATTRIB: &str = "ref_volume";
pub(crate) const REFERENCE_SHAPE_MATRIX_INV_ATTRIB: &str = "ref_shape_mtx_inv";
pub(crate) const STRAIN_ENERGY_ATTRIB: &str = "strain_energy";
pub(crate) const ELASTIC_FORCE_ATTRIB: &str = "elastic_force";

pub(crate) type FixedIntType = i8;
pub(crate) type DispType = [f64; 3];
pub(crate) type RefPosType = [f64; 3];
pub(crate) type RefVolType = f64;
pub(crate) type RefShapeMtxInvType = Matrix3<f64>;
pub(crate) type StrainEnergyType = f64;
pub(crate) type ElasticForceType = [f64; 3];
