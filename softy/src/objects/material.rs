//! This module defines the uniform material assigned to every physical object.
//! Any material property set to `None` typically indicates that this property
//! is defined on the mesh itself. This allows us to define variable material
//! properties.

use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialBase<P> {
    /// Material unique identifier.
    /// It is the user's responsibility to ensure that this value is used correctly:
    ///
    /// `material1 == material2` if and only if `material1.id == material2.id`.
    pub id: usize,
    /// Material properties specific to the type of material.
    pub properties: P,
}

/// Common material properties shared among all deformable objects.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeformableProperties {
    /// Parameters determining the elastic behaviour of a simulated solid. If `None`, we will look
    /// for elasticity parameters in the mesh.
    pub elasticity: Option<Elasticity>,
    /// The density of the material. If `None`, we will look for a density attribute in the mesh.
    pub density: Option<f32>,
    /// Coefficient measuring the amount of artificial viscosity as dictated by the Rayleigh
    /// damping model. This value should be premultiplied by the timestep reciprocal to save
    /// passing the time step around to elastic energy models which are otherwise independent of
    /// time step.
    pub damping: f32,
}

impl Default for DeformableProperties {
    fn default() -> Self {
        DeformableProperties {
            elasticity: None, // Assuming variable elasticity
            density: None,    // Assuming variable density
            damping: 0.0,
        }
    }
}

/// Fixed material properties.
///
/// A fixed material is not subject to external physics and so has no physical properties.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FixedProperties;

impl Default for FixedProperties {
    fn default() -> Self {
        FixedProperties
    }
}

/// Rigid material properties.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigidProperties {
    pub density: f32,
}

/// Soft shell material properties.
///
/// This struct describes the physical properties of a deformable shell object.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftShellProperties {
    /// Bending stiffness sets the resistance of the material to bending.
    ///
    /// Bending stiffness is measured in kg/s^2 which are the units of surface tension.
    /// When set to `None`, bending stiffness will be inferred from the underlying mesh or
    /// otherwise assumed to be zero.
    pub bending_stiffness: Option<f32>,
    /// Common material properties shared among all deformable materials.
    pub deformable: DeformableProperties,
}

impl Default for SoftShellProperties {
    fn default() -> Self {
        SoftShellProperties {
            bending_stiffness: None,
            deformable: DeformableProperties::default(),
        }
    }
}

/// Solids are always elastically deformable. For rigid solids, use rigid shells,
/// because rigid objects don't require interior properties.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolidProperties {
    /// Volume preservation sets the material to be globally incompressible.
    ///
    /// In contrast to Bulk Modulus, this parameter affects *global* incompressibility,
    /// while Bulk Modulus affects *local* incompressibility (on a per element level).
    pub volume_preservation: bool,

    /// Common material properties shared among all deformable materials.
    pub deformable: DeformableProperties,
}

/// A trait for materials that can can be moved by external forces.
pub trait DynamicMaterial {
    /// The density parameter.
    fn density(&self) -> Option<f32>;
}

/// A trait for materials capable of deforming. This helps generalize over Solid and Shell materials.
/// If the output is `Option`al, it means that the object doesn't have that
/// material set. If it is set to be some default value the option should not be
/// `None`. This helps distinguish between variable and uniform material properties.
pub trait DeformableMaterial: DynamicMaterial {
    /// The elasticity parameters.
    fn elasticity(&self) -> Option<Elasticity>;
    /// The damping parameter.
    fn damping(&self) -> f32;
}

impl Default for SolidProperties {
    fn default() -> Self {
        SolidProperties {
            volume_preservation: false,
            deformable: DeformableProperties::default(),
        }
    }
}

/// A generic material that can be assigned to a `TriMeshShell`.
#[derive(Copy, Clone, Debug)]
pub enum ShellMaterial {
    Fixed(FixedMaterial),
    Rigid(RigidMaterial),
    Soft(SoftShellMaterial),
}

/// A generic material.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Material {
    Fixed(FixedMaterial),
    Rigid(RigidMaterial),
    SoftShell(SoftShellMaterial),
    Solid(SolidMaterial),
}

pub type FixedMaterial = MaterialBase<FixedProperties>;
pub type RigidMaterial = MaterialBase<RigidProperties>;
pub type SoftShellMaterial = MaterialBase<SoftShellProperties>;
pub type SolidMaterial = MaterialBase<SolidProperties>;

impl From<FixedMaterial> for Material {
    fn from(mat: FixedMaterial) -> Self {
        Material::Fixed(mat)
    }
}
impl From<RigidMaterial> for Material {
    fn from(mat: RigidMaterial) -> Self {
        Material::Rigid(mat)
    }
}
impl From<SoftShellMaterial> for Material {
    fn from(mat: SoftShellMaterial) -> Self {
        Material::SoftShell(mat)
    }
}

impl From<SolidMaterial> for Material {
    fn from(mat: SolidMaterial) -> Self {
        Material::Solid(mat)
    }
}
impl RigidMaterial {
    /// Construct a rigid material with the given identifier and density.
    ///
    /// Creating a rigid material requires the density parameter because there is no sensible
    /// common default for a rigid material.
    pub fn new(id: usize, density: f32) -> RigidMaterial {
        MaterialBase {
            id,
            properties: RigidProperties { density },
        }
    }
}
/*
 * Important: When making changes to materials, make sure that the following conversions are up to
 * date. The rest of the material constructors use this code for building new materials from old
 * ones.
 *
 * The exception is RigidMaterial which cannot be created from a fixed material without knowing
 * density in advance.
 */

impl From<FixedMaterial> for SoftShellMaterial {
    fn from(mtl: FixedMaterial) -> Self {
        mtl.with_properties(SoftShellProperties::default())
    }
}

impl From<RigidMaterial> for SoftShellMaterial {
    fn from(mtl: RigidMaterial) -> Self {
        mtl.with_properties(SoftShellProperties::default())
            .with_density(mtl.properties.density)
    }
}

impl From<FixedMaterial> for ShellMaterial {
    fn from(mtl: FixedMaterial) -> Self {
        ShellMaterial::Fixed(mtl)
    }
}

impl From<RigidMaterial> for ShellMaterial {
    fn from(mtl: RigidMaterial) -> Self {
        ShellMaterial::Rigid(mtl)
    }
}

impl From<SoftShellMaterial> for ShellMaterial {
    fn from(mtl: SoftShellMaterial) -> Self {
        ShellMaterial::Soft(mtl)
    }
}

/*
 * Material assembly
 */

impl<P: Default> MaterialBase<P> {
    /// Build a default material.
    ///
    /// This function may require specifying the generic properties parameter `P` explicitly.
    /// Consider using one of `fixed`, `rigid`, `soft_shell` or `soft_solid` constructors instead.
    pub fn new(id: usize) -> MaterialBase<P> {
        MaterialBase {
            id,
            properties: Default::default(),
        }
    }

    /// Overrides the preset `id`.
    ///
    /// This is useful for incrementing the material id
    /// when building on top of an existing material.
    pub fn with_id(mut self, id: usize) -> MaterialBase<P> {
        self.id = id;
        self
    }
}

impl<P> MaterialBase<P> {
    /// Set the properties of this material.
    ///
    /// This function will convert this material into a `Material<Q>` type where
    /// `Q` is the type of the given properties.
    pub fn with_properties<Q>(self, properties: Q) -> MaterialBase<Q> {
        MaterialBase {
            id: self.id,
            properties,
        }
    }
}

impl ShellMaterial {
    /// Build a default material.
    pub fn new(id: usize) -> Self {
        Self::Fixed(FixedMaterial::new(id))
    }

    /// Add elasticity parameters to this material.
    ///
    /// This function converts this material to a `SoftShellMaterial`.
    pub fn with_elasticity(self, elasticity: Elasticity) -> Self {
        match self {
            ShellMaterial::Fixed(m) => m.with_elasticity(elasticity),
            ShellMaterial::Rigid(m) => m.with_elasticity(elasticity),
            ShellMaterial::Soft(m) => m.with_elasticity(elasticity),
        }
        .into()
    }

    /// Add damping to this material.
    ///
    /// This function converts this material to a `SoftShellMaterial`.
    pub fn with_damping(self, damping: f32, time_step: f32) -> Self {
        match self {
            ShellMaterial::Fixed(m) => m.with_damping(damping, time_step),
            ShellMaterial::Rigid(m) => m.with_damping(damping, time_step),
            ShellMaterial::Soft(m) => m.with_damping(damping, time_step),
        }
        .into()
    }

    /// Add density to this material.
    ///
    /// This function converts this material to a `RigidMaterial`.
    pub fn with_density(self, density: f32) -> Self {
        match self {
            ShellMaterial::Fixed(m) => m.with_density(density).into(),
            ShellMaterial::Rigid(m) => m.with_density(density).into(),
            ShellMaterial::Soft(m) => m.with_density(density).into(),
        }
    }

    /// Add bending stiffness to this material.
    ///
    /// This function converts this material to a `SoftShellMaterial`.
    pub fn with_bending_stiffness(self, stiffness: f32) -> Self {
        match self {
            ShellMaterial::Fixed(m) => m.with_bending_stiffness(stiffness),
            ShellMaterial::Rigid(m) => m.with_bending_stiffness(stiffness),
            ShellMaterial::Soft(m) => m.with_bending_stiffness(stiffness),
        }
        .into()
    }
}

impl FixedMaterial {
    /// Add elasticity parameters to this material.
    ///
    /// This function converts this material to a `SoftShellMaterial`.
    pub fn with_elasticity(self, elasticity: Elasticity) -> SoftShellMaterial {
        SoftShellMaterial::from(self).with_elasticity(elasticity)
    }

    /// Add damping to this material.
    ///
    /// This function converts this material to a `SoftShellMaterial`.
    pub fn with_damping(self, damping: f32, time_step: f32) -> SoftShellMaterial {
        SoftShellMaterial::from(self).with_damping(damping, time_step)
    }

    /// Add density to this material.
    ///
    /// This function converts this material to a `RigidMaterial`.
    pub fn with_density(self, density: f32) -> RigidMaterial {
        RigidMaterial::new(self.id, density)
    }

    /// Add bending stiffness to this material.
    ///
    /// This function converts this material to a `SoftShellMaterial`.
    pub fn with_bending_stiffness(self, stiffness: f32) -> SoftShellMaterial {
        SoftShellMaterial::from(self).with_bending_stiffness(stiffness)
    }
}

impl RigidMaterial {
    /// Add elasticity parameters to this material.
    ///
    /// This function converts this material to a `SoftShellMaterial`.
    pub fn with_elasticity(self, elasticity: Elasticity) -> SoftShellMaterial {
        SoftShellMaterial::from(self).with_elasticity(elasticity)
    }

    /// Add damping to this material.
    ///
    /// This function converts this material to a `SoftShellMaterial`.
    pub fn with_damping(self, damping: f32, time_step: f32) -> SoftShellMaterial {
        SoftShellMaterial::from(self).with_damping(damping, time_step)
    }

    /// Construct a new `RigidMaterial` for the current one with the given density parameter.
    pub fn with_density(mut self, density: f32) -> RigidMaterial {
        self.properties.density = density;
        self
    }

    /// Add bending stiffness to this material.
    ///
    /// This function converts this material to a `SoftShellMaterial`.
    pub fn with_bending_stiffness(self, stiffness: f32) -> SoftShellMaterial {
        SoftShellMaterial::from(self).with_bending_stiffness(stiffness)
    }
}

impl SoftShellMaterial {
    /// Construct a new `SoftShellMaterial` from this one with the given elasticity parameter.
    pub fn with_elasticity(mut self, elasticity: Elasticity) -> SoftShellMaterial {
        self.properties.deformable.elasticity = Some(elasticity);
        self
    }

    /// Construct a new `SoftShellMaterial` from this one with the given damping parameter.
    pub fn with_damping(mut self, damping: f32, time_step: f32) -> SoftShellMaterial {
        self.properties.deformable = self.properties.deformable.with_damping(damping, time_step);
        self
    }

    /// Construct a new `SoftShellMaterial` from this one with the given density parameter.
    pub fn with_density(mut self, density: f32) -> SoftShellMaterial {
        self.properties.deformable.density = Some(density);
        self
    }

    /// Construct a new `SoftShellMaterial` from this one with the given bending stiffness
    /// parameter.
    pub fn with_bending_stiffness(mut self, stiffness: f32) -> SoftShellMaterial {
        self.properties.bending_stiffness = Some(stiffness);
        self
    }

    /// Get the bending stiffness parameter.
    pub fn bending_stiffness(&self) -> Option<f32> {
        self.properties.bending_stiffness
    }
}

impl SolidMaterial {
    pub fn with_elasticity(mut self, elasticity: Elasticity) -> SolidMaterial {
        self.properties.deformable.elasticity = Some(elasticity);
        self
    }
    pub fn with_density(mut self, density: f32) -> SolidMaterial {
        self.properties.deformable.density = Some(density);
        self
    }
    pub fn with_damping(mut self, damping: f32, time_step: f32) -> SolidMaterial {
        self.properties.deformable = self.properties.deformable.with_damping(damping, time_step);
        self
    }
    pub fn with_volume_preservation(mut self, volume_preservation: bool) -> SolidMaterial {
        self.properties.volume_preservation = volume_preservation;
        self
    }
    pub fn volume_preservation(&self) -> bool {
        self.properties.volume_preservation
    }
    pub fn model(&self) -> ElasticityModel {
        self.properties
            .deformable
            .elasticity
            .map_or(ElasticityModel::NeoHookean, |x| x.model)
    }
}

impl DeformableProperties {
    pub fn with_elasticity(self, elasticity: Elasticity) -> DeformableProperties {
        DeformableProperties {
            elasticity: Some(elasticity),
            ..self
        }
    }
    pub fn with_density(self, density: f32) -> DeformableProperties {
        DeformableProperties {
            density: Some(density),
            ..self
        }
    }
    pub fn with_damping(self, mut damping: f32, time_step: f32) -> DeformableProperties {
        damping *= if time_step != 0.0 {
            1.0 / time_step
        } else {
            0.0
        };

        DeformableProperties { damping, ..self }
    }

    fn elasticity(&self) -> Option<Elasticity> {
        self.elasticity
    }
    fn damping(&self) -> f32 {
        self.damping
    }
    fn density(&self) -> Option<f32> {
        self.density
    }
}

/*
 * Elasticity parameters
 */

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ElasticityModel {
    StableNeoHookean,
    NeoHookean,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Elasticity {
    pub model: ElasticityModel,
    pub parameters: ElasticityParameters,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ElasticityParameters {
    /// Lame parameters
    Lame {
        /// First Lame parameter. Measured in Pa = N/m² = kg/(ms²).
        lambda: f32,
        /// Second Lame parameter. Measured in Pa = N/m² = kg/(ms²).
        mu: f32,
    },
    /// Young's modulus and Poisson's ratio.
    YoungPoisson {
        /// Young's modulus.
        ///
        /// Measured in Pa = N/m² = kg/(ms²).
        young: f32,
        /// Poisson's ratio.
        ///
        /// This is in the range `[0. 0.5]` for most realistic materials.
        poisson: f32,
    },
    /// Bulk and shear moduli.
    BulkShear {
        /// Bulk modulus measures the material's resistance to expansion and compression, i.e. its
        /// incompressibility. The larger the value, the more incompressible the material is.
        /// Think of this as "Volume Stiffness".
        bulk: f32,
        /// Shear modulus measures the material's resistance to shear deformation. The larger the
        /// value, the more it resists changes in shape. Think of this as "Shape Stiffness".
        shear: f32,
    },
}

impl ElasticityParameters {
    /// Retrieves the first Lamé parameter.
    pub fn lambda(self) -> f32 {
        match self {
            ElasticityParameters::Lame { lambda, .. } => lambda,
            ElasticityParameters::YoungPoisson { young, poisson } => {
                young * poisson / (1.0 + poisson) * (1.0 - 2.0 * poisson)
            }
            ElasticityParameters::BulkShear { bulk, shear } => bulk - 2.0 * shear / 3.0,
        }
    }
    /// Retrieve raw first and second Lamé parameters as a pair.
    pub fn mu(self) -> f32 {
        match self {
            ElasticityParameters::Lame { mu, .. } => mu,
            ElasticityParameters::YoungPoisson { young, poisson } => {
                young / (2.0 * (1.0 + poisson))
            }
            ElasticityParameters::BulkShear { shear, .. } => shear,
        }
    }
    /// Retrieve raw first and second Lame parameters as a pair.
    pub fn lame(self) -> (f32, f32) {
        (self.lambda(), self.mu())
    }
    pub fn to_lame(self) -> ElasticityParameters {
        let (lambda, mu) = self.lame();
        ElasticityParameters::Lame { lambda, mu }
    }
}

impl Elasticity {
    /// Construct elasticity parameters from standard Lame parameters.
    ///
    /// In case of Stable Neo-Hookean, the given parameters are converted to be compatible with
    /// that model.
    pub fn new(model: ElasticityModel, parameters: ElasticityParameters) -> Self {
        Self { model, parameters }
    }
    /// Construct elasticity parameters from standard Lame parameters.
    pub fn from_lame(lambda: f32, mu: f32, model: ElasticityModel) -> Self {
        Self {
            model,
            parameters: ElasticityParameters::Lame { lambda, mu },
        }
    }

    /// Bulk modulus measures the material's resistance to expansion and compression, i.e. its
    /// incompressibility. The larger the value, the more incompressible the material is.
    /// Think of this as "Volume Stiffness".
    /// Shear modulus measures the material's resistance to shear deformation. The larger the
    /// value, the more it resists changes in shape. Think of this as "Shape Stiffness".
    pub fn from_bulk_shear(bulk: f32, shear: f32) -> Self {
        Self::from_bulk_shear_with_model(bulk, shear, ElasticityModel::NeoHookean)
    }
    pub fn from_young_poisson(young: f32, poisson: f32) -> Self {
        Self::from_young_poisson_with_model(young, poisson, ElasticityModel::NeoHookean)
    }

    pub fn from_bulk_shear_with_model(bulk: f32, shear: f32, model: ElasticityModel) -> Self {
        Self::new(model, ElasticityParameters::BulkShear { bulk, shear })
    }
    pub fn from_young_poisson_with_model(young: f32, poisson: f32, model: ElasticityModel) -> Self {
        Self::new(model, ElasticityParameters::YoungPoisson { young, poisson })
    }

    /// Return the model effective Lame parameters to be used in simulation.
    pub(crate) fn lame(&self) -> (f32, f32) {
        let (lambda, mu) = self.parameters.lame();
        match self.model {
            ElasticityModel::NeoHookean => (lambda, mu),
            ElasticityModel::StableNeoHookean => (lambda + 5.0 * mu / 6.0, 4.0 * mu / 3.0),
        }
    }
    pub(crate) fn lambda(&self) -> f32 {
        self.lame().0
    }
    pub(crate) fn mu(&self) -> f32 {
        self.lame().1
    }
}

impl DynamicMaterial for RigidMaterial {
    fn density(&self) -> Option<f32> {
        Some(self.properties.density)
    }
}

impl DynamicMaterial for SoftShellMaterial {
    fn density(&self) -> Option<f32> {
        self.properties.deformable.density()
    }
}

impl DynamicMaterial for SolidMaterial {
    fn density(&self) -> Option<f32> {
        self.properties.deformable.density()
    }
}

impl DeformableMaterial for SoftShellMaterial {
    fn elasticity(&self) -> Option<Elasticity> {
        self.properties.deformable.elasticity()
    }
    fn damping(&self) -> f32 {
        self.properties.deformable.damping()
    }
}

impl DeformableMaterial for SolidMaterial {
    fn elasticity(&self) -> Option<Elasticity> {
        self.properties.deformable.elasticity()
    }
    fn damping(&self) -> f32 {
        self.properties.deformable.damping()
    }
}
