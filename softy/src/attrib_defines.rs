use geo::math::Matrix3;

pub const SOURCE_INDEX_ATTRIB : &str = "src_idx";
pub(crate) const TETMESH_VERTEX_INDEX_ATTRIB : &str = "_i";
pub(crate) const DENSITY_ATTRIB : &str = "density";
pub(crate) const MU_ATTRIB : &str = "mu";
pub(crate) const LAMBDA_ATTRIB : &str = "lambda";
pub(crate) const FIXED_ATTRIB: &str = "fixed";
pub(crate) const VELOCITY_ATTRIB: &str = "vel";
pub(crate) const REFERENCE_POSITION_ATTRIB: &str = "ref";
pub(crate) const REFERENCE_VOLUME_ATTRIB: &str = "ref_volume";
pub(crate) const REFERENCE_SHAPE_MATRIX_INV_ATTRIB: &str = "ref_shape_mtx_inv";
pub(crate) const STRAIN_ENERGY_ATTRIB: &str = "strain_energy";
pub(crate) const ELASTIC_FORCE_ATTRIB: &str = "elastic_force";
pub(crate) const FRICTION_ATTRIB: &str = "friction";
pub(crate) const CONTACT_ATTRIB: &str = "contact";

pub type SourceIndexType = u32;
pub(crate) type TetMeshVertexIndexType = usize;
pub(crate) type DensityType = f64;
pub(crate) type MuType = f64;
pub(crate) type LambdaType = f64;
pub(crate) type FixedIntType = i8;
pub(crate) type VelType = [f64; 3];
pub(crate) type RefPosType = [f64; 3];
pub(crate) type RefVolType = f64;
pub(crate) type RefShapeMtxInvType = Matrix3<f64>;
pub(crate) type StrainEnergyType = f64;
pub(crate) type ElasticForceType = [f64; 3];
pub(crate) type FrictionImpulseType = [f64; 3];
pub(crate) type ContactImpulseType = [f64; 3];
