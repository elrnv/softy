pub const SOURCE_INDEX_ATTRIB: &str = "src_idx";
pub const FIXED_ATTRIB: &str = "fixed";
pub const MATERIAL_ID_ATTRIB: &str = "mtl_id";
pub const OBJECT_ID_ATTRIB: &str = "obj_id";
pub const VOLUME_ZONE_ID_ATTRIB: &str = "vol_zone_id";
pub const DENSITY_ATTRIB: &str = "density";
pub const DAMPING_ATTRIB: &str = "damping";
pub const MU_ATTRIB: &str = "mu";
pub const LAMBDA_ATTRIB: &str = "lambda";
pub const BENDING_STIFFNESS_ATTRIB: &str = "bend";
pub const VELOCITY_ATTRIB: &str = "vel";
pub const REFERENCE_VERTEX_POS_ATTRIB: &str = "ref";
pub const REFERENCE_CELL_VERTEX_POS_ATTRIB: &str = "ref_cv";
pub const REFERENCE_FACE_VERTEX_POS_ATTRIB: &str = "ref_fv";
pub const REFERENCE_ANGLE_ATTRIB: &str = "ref_angle";
pub(crate) const ELASTIC_FORCE_ATTRIB: &str = "elastic_force";
pub(crate) const CONSTRAINT_FORCE_ATTRIB: &str = "constraint_force";
pub const FRICTION_ATTRIB: &str = "friction";
pub const CONTACT_ATTRIB: &str = "contact";
pub(crate) const VERTEX_TYPE_ATTRIB: &str = "__softy_internal_vertex_type";
pub(crate) const STRAIN_ENERGY_ATTRIB: &str = "strain_energy";
pub(crate) const MASS_ATTRIB: &str = "mass";
pub(crate) const MASS_INV_ATTRIB: &str = "__softy_internal_mass_inv";
pub(crate) const ORIGINAL_VERTEX_INDEX_ATTRIB: &str = "__softy_internal_orig_vertex_index";
pub const RESIDUAL_ATTRIB: &str = "residual";
pub(crate) const POTENTIAL_ATTRIB: &str = "potential";

#[cfg(feature = "optsolver")]
mod optsolver_defines {
    use tensr::{Matrix2, Matrix3};
    pub(crate) const REFERENCE_VOLUME_ATTRIB: &str = "ref_volume";
    pub(crate) const REFERENCE_AREA_ATTRIB: &str = "ref_area";
    pub(crate) const REFERENCE_SHAPE_MATRIX_INV_ATTRIB: &str = "ref_shape_mtx_inv";
    pub(crate) const PRESSURE_ATTRIB: &str = "pressure";

    pub(crate) const TETMESH_VERTEX_INDEX_ATTRIB: &str = "_i";

    pub(crate) type RigidRefPosType = [f64; 3];
    pub(crate) type RefVolType = f64;
    pub(crate) type RefAreaType = f64;
    pub(crate) type RefTetShapeMtxInvType = Matrix3<f64>;
    pub(crate) type RefTriShapeMtxInvType = Matrix2<f64>;
    pub(crate) type PressureType = f64;

    pub(crate) type TetMeshVertexIndexType = usize;
}

#[cfg(feature = "optsolver")]
use optsolver_defines::*;

pub(crate) type StrainEnergyType = f64;
pub type SourceIndexType = usize;
pub type OriginalVertexIndexType = usize;
pub type FixedIntType = i8;
pub type MaterialIdType = i32;
pub type ObjectIdType = i32;
pub type VolumeZoneIdType = i32;
pub(crate) type MassType = f64;
pub(crate) type MassInvType = f64;
pub type DensityType = f32;
pub type DampingType = f32;
pub type MuType = f32;
pub type LambdaType = f32;
pub type BendingStiffnessType = f32;
pub type VelType = [f64; 3];
pub type RefPosType = [f32; 3];
pub type RefAngleType = f32;
pub(crate) type ElasticForceType = [f64; 3];
pub type FrictionImpulseType = [f64; 3];
pub type ContactImpulseType = [f64; 3];
pub type ResidualType = [f64; 3];
pub(crate) type PotentialType = f64;
