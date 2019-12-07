use utils::soap::{Matrix3, Matrix2};

pub const SOURCE_INDEX_ATTRIB: &str = "src_idx";
pub const FIXED_ATTRIB: &str = "fixed";
pub(crate) const TETMESH_VERTEX_INDEX_ATTRIB: &str = "_i";
pub(crate) const MASS_ATTRIB: &str = "mass";
pub const DENSITY_ATTRIB: &str = "density";
pub const MU_ATTRIB: &str = "mu";
pub const LAMBDA_ATTRIB: &str = "lambda";
pub(crate) const VELOCITY_ATTRIB: &str = "vel";
pub const REFERENCE_POSITION_ATTRIB: &str = "ref";
pub(crate) const REFERENCE_VOLUME_ATTRIB: &str = "ref_volume";
pub(crate) const REFERENCE_AREA_ATTRIB: &str = "ref_area";
pub(crate) const REFERENCE_SHAPE_MATRIX_INV_ATTRIB: &str = "ref_shape_mtx_inv";
pub(crate) const STRAIN_ENERGY_ATTRIB: &str = "strain_energy";
pub(crate) const ELASTIC_FORCE_ATTRIB: &str = "elastic_force";
pub(crate) const POTENTIAL_ATTRIB: &str = "potential";
pub(crate) const PRESSURE_ATTRIB: &str = "pressure";
pub const FRICTION_ATTRIB: &str = "friction";
pub const CONTACT_ATTRIB: &str = "contact";

pub type SourceIndexType = usize;
pub type FixedIntType = i8;
pub(crate) type TetMeshVertexIndexType = usize;
pub(crate) type MassType = f64;
pub type DensityType = f32;
pub type MuType = f32;
pub type LambdaType = f32;
pub(crate) type VelType = [f64; 3];
pub type RefPosType = [f32; 3];
pub(crate) type RefVolType = f64;
pub(crate) type RefAreaType = f64;
pub(crate) type RefTetShapeMtxInvType = Matrix3<f64>;
pub(crate) type RefTriShapeMtxInvType = Matrix2<f64>;
pub(crate) type StrainEnergyType = f64;
pub(crate) type ElasticForceType = [f64; 3];
pub(crate) type PotentialType = f64;
pub(crate) type PressureType = f64;
pub type FrictionImpulseType = [f64; 3];
pub type ContactImpulseType = [f64; 3];
