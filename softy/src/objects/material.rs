//! This module defines the uniform material assigned to every physical object.
//! Any material property set to `None` typically indicates that this property
//! is defined on the mesh itself. This allows us to define variable material
//! properties.
//!
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Material<P> {
    /// Material unique identifier.
    /// It is the user's responsibility to ensure that this value is used correctly:
    ///
    /// `material1 == material2` if and only if `material1.id == material2.id`.
    pub id: usize,
    /// Material properties specific to the type of material.
    pub properties: P,
}

/// Common material properties shared among all deformable objects.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DeformableProperties {
    /// Parameters determining the elastic behaviour of a simulated solid. If `None`, we will look
    /// for elasticity parameters in the mesh.
    pub elasticity: Option<ElasticityParameters>,
    /// The density of the material. If `None`, we will look for a density attribute in the mesh.
    pub density: Option<f64>,
    /// Coefficient measuring the amount of artificial viscosity as dictated by the Rayleigh
    /// damping model. This value should be premultiplied by the timestep reciprocal to save
    /// passing the time step around to elastic energy models which are otherwise independent of
    /// time step.
    pub damping: f64,
    /// Scaling factor used to adjust the magnitudes of the parameters to be closer to 1.0.
    pub scale: f64,
}

impl Default for DeformableProperties {
    fn default() -> Self {
        DeformableProperties {
            elasticity: None, // Assuming variable elasticity
            density: None,    // Assuming variable density
            damping: 0.0,
            scale: 1.0,
        }
    }
}

/// Shells can be deformable or completely rigid. Rigid shells are not to be
/// confused with solids, which are in fact deformable and are fundamentally
/// different because they contain properties of the interior material.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ShellProperties {
    /// A static shell is an infinite mass kinematic object.
    Fixed,
    /// A rigid shell has 6 degrees of freedom: 3 for translation and 3 for rotation.
    Rigid { density: f64 },
    /// A deformable shell has a 3 degrees of freedom for every vertex.
    Deformable { deformable: DeformableProperties },
}

impl Default for ShellProperties {
    fn default() -> Self {
        ShellProperties::Fixed
    }
}

/// Solids are always elastically deformable. For rigid solids, use shells,
/// because rigid solids don't require interior properties.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SolidProperties {
    /// Volume preservation sets the material to be globally incompressible, if set to `true`. In
    /// contrast to Bulk Modulus, this parameter affects global incompressibility,
    /// while Bulk Modulus affects *local* incompressibility (on a per element level).
    pub volume_preservation: bool,

    /// Common material properties shared among all deformable materials.
    pub deformable: DeformableProperties,
}

/// A trait for materials capable of deforming. This helps generalize over Solid and Shell materials.
/// If the output is `Option`al, it means that the object doesn't have that
/// material set. If it is set to be some default value the option should not be
/// `None`. This helps distinguish between variable and uniform material properties.
pub trait Deformable {
    /// Scale used to adjust internal material properties to be closer to 1.0.
    fn scale(&self) -> f64;
    /// The exact elasticity parameters used by solver. These are typically
    /// scaled to be closer to 1.0.
    fn scaled_elasticity(&self) -> Option<ElasticityParameters>;
    /// The exact damping parameter used by solver. This is typically scaled to
    /// be closer to 1.0.
    fn scaled_damping(&self) -> f64;
    /// The exact density parameter used by solver. This is typically scaled to
    /// be closer to 1.0.
    fn scaled_density(&self) -> Option<f64>;
    /// The elasticity parameters provided in the input.
    fn elasticity(&self) -> Option<ElasticityParameters>;
    /// The damping parameter provided in the input.
    fn damping(&self) -> f64;
    /// The density parameter provided in the input.
    fn density(&self) -> Option<f64>;
}

impl Default for SolidProperties {
    fn default() -> Self {
        SolidProperties {
            volume_preservation: false,
            deformable: DeformableProperties::default(),
        }
    }
}

pub type ShellMaterial = Material<ShellProperties>;
pub type SolidMaterial = Material<SolidProperties>;

impl ShellMaterial {
    pub fn fixed(id: usize) -> Self {
        Material {
            id,
            properties: ShellProperties::Fixed,
        }
    }
    pub fn rigid(id: usize, density: f64) -> Self {
        Material {
            id,
            properties: ShellProperties::Rigid { density },
        }
    }
    pub fn deformable(id: usize, deformable: DeformableProperties) -> Self {
        Material {
            id,
            properties: ShellProperties::Deformable { deformable },
        }
    }
    pub fn with_elasticity(mut self, elasticity: ElasticityParameters) -> ShellMaterial {
        match &mut self.properties {
            ShellProperties::Deformable { deformable } => {
                deformable.elasticity = Some(elasticity);
                self
            }
            ShellProperties::Rigid { density } => {
                Self::deformable(self.id,
                                 DeformableProperties::default()
                                 .with_elasticity(elasticity)
                                 .with_density(*density))
            }
            ShellProperties::Fixed => {
                Self::deformable(self.id,
                                 DeformableProperties::default()
                                 .with_elasticity(elasticity))
            }
        }
    }
    pub fn with_damping(mut self, damping: f64, time_step: f64) -> ShellMaterial {
        match &mut self.properties {
            ShellProperties::Deformable { deformable } => {
                *deformable = deformable.with_damping(damping, time_step);
                self
            }
            ShellProperties::Rigid { density } => {
                Self::deformable(self.id,
                                 DeformableProperties::default()
                                 .with_damping(damping, time_step)
                                 .with_density(*density))
            }
            ShellProperties::Fixed => {
                Self::deformable(self.id,
                                 DeformableProperties::default()
                                 .with_damping(damping, time_step))
            }
        }
    }
    pub fn with_density(mut self, density: f64) -> ShellMaterial {
        match &mut self.properties {
            ShellProperties::Deformable { deformable } => {
                deformable.density = Some(density);
                self
            }
            ShellProperties::Rigid { density: self_density } => {
                *self_density = density;
                self
            }
            ShellProperties::Fixed => {
                Self::rigid(self.id, density)
            }
        }
    }

    pub fn normalized(mut self) -> ShellMaterial {
        match &mut self.properties {
            ShellProperties::Deformable { ref mut deformable } => {
                *deformable = deformable.normalized();
                self
            }
            _ => self,
        }
    }
}

impl SolidMaterial {
    pub fn with_elasticity(mut self, elasticity: ElasticityParameters) -> SolidMaterial {
        self.properties.deformable.elasticity = Some(elasticity);
        self
    }
    pub fn with_density(mut self, density: f64) -> SolidMaterial {
        self.properties.deformable.density = Some(density);
        self
    }
    pub fn with_damping(mut self, damping: f64, time_step: f64) -> SolidMaterial {
        self.properties.deformable = self.properties.deformable.with_damping(damping, time_step);
        self
    }
    pub fn with_volume_preservation(mut self, volume_preservation: bool) -> SolidMaterial {
        self.properties.volume_preservation = volume_preservation;
        self
    }
    pub fn normalized(mut self) -> SolidMaterial {
        self.properties.deformable = self.properties.deformable.normalized();
        self
    }
    pub fn volume_preservation(&self) -> bool {
        self.properties.volume_preservation
    }
    pub fn scaled_damping(&self) -> f64 {
        self.properties.deformable.damping
    }
}

impl<P: Default> Material<P> {
    pub fn new(id: usize) -> Material<P> {
        Material {
            id,
            properties: Default::default(),
        }
    }
    /// Overrides the preset id. This is useful for incrementing the material id
    /// when building on top of an existing material.
    pub fn with_id(mut self, id: usize) -> Material<P> {
        self.id = id;
        self
    }
}

impl DeformableProperties {
    pub fn with_elasticity(self, elasticity: ElasticityParameters) -> DeformableProperties {
        DeformableProperties {
            elasticity: Some(elasticity),
            ..self
        }
    }
    pub fn with_density(self, density: f64) -> DeformableProperties {
        DeformableProperties {
            density: Some(density),
            ..self
        }
    }
    pub fn with_damping(self, mut damping: f64, time_step: f64) -> DeformableProperties {
        damping *= if time_step != 0.0 {
            1.0 / time_step
        } else {
            0.0
        };

        DeformableProperties { damping, ..self }
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Rescale parameters uniformly to be closer to 1.0.
    pub fn normalized(self) -> DeformableProperties {
        let DeformableProperties {
            mut elasticity,
            mut density,
            mut damping,
            ..
        } = self.unnormalized();

        let scale;

        if let Some(ref mut elasticity) = elasticity {
            scale = elasticity.normalize();
            density = density.map(|d| d * scale);
            damping *= scale;
        } else if let Some(ref mut density) = density {
            scale = if *density > 0.0 {
                1.0 / *density
            } else if damping > 0.0 {
                1.0 / damping
            } else {
                1.0
            };
            *density *= scale;
            damping *= scale;
        } else {
            scale = if damping > 0.0 { 1.0 / damping } else { 1.0 };
            damping *= scale;
        }

        DeformableProperties {
            elasticity,
            density,
            damping,
            scale,
        }
    }

    /// Undo normalization.
    pub fn unnormalized(self) -> DeformableProperties {
        let DeformableProperties {
            mut elasticity,
            mut density,
            mut damping,
            scale,
        } = self;

        if let Some(ref mut elasticity) = elasticity {
            *elasticity = elasticity.scaled(1.0 / scale);
        }

        if let Some(ref mut density) = density {
            *density /= scale;
        }

        damping /= scale;

        DeformableProperties {
            elasticity,
            density,
            damping,
            scale: 1.0,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ElasticityParameters {
    /// First Lame parameter. Measured in Pa = N/m² = kg/(ms²).
    pub lambda: f64,
    /// Second Lame parameter. Measured in Pa = N/m² = kg/(ms²).
    pub mu: f64,
}

impl ElasticityParameters {
    pub fn scaled(self, scale: f64) -> ElasticityParameters {
        ElasticityParameters {
            lambda: self.lambda * scale,
            mu: self.mu * scale,
        }
    }

    /// Rescale parameters uniformly to be closer to 1.0.
    pub fn normalize(&mut self) -> f64 {
        let scale = if self.mu > 0.0 {
            1.0 / self.mu
        } else if self.lambda > 0.0 {
            1.0 / self.lambda
        } else {
            1.0
        };
        *self = self.scaled(scale);
        scale
    }

    /// Bulk modulus measures the material's resistance to expansion and compression, i.e. its
    /// incompressibility. The larger the value, the more incompressible the material is.
    /// Think of this as "Volume Stiffness".
    /// Shear modulus measures the material's resistance to shear deformation. The larger the
    /// value, the more it resists changes in shape. Think of this as "Shape Stiffness".
    pub fn from_bulk_shear(bulk: f64, shear: f64) -> Self {
        ElasticityParameters {
            lambda: bulk - 2.0 * shear / 3.0,
            mu: shear,
        }
    }
    pub fn from_young_poisson(young: f64, poisson: f64) -> Self {
        ElasticityParameters {
            lambda: young * poisson / (1.0 + poisson) * (1.0 - 2.0 * poisson),
            mu: young / (2.0 * (1.0 + poisson)),
        }
    }
}

/*
 * Deformable implementations
 */

impl Deformable for DeformableProperties {
    fn scale(&self) -> f64 {
        DeformableProperties::scale(self)
    }
    fn scaled_elasticity(&self) -> Option<ElasticityParameters> {
        self.elasticity
    }
    fn scaled_damping(&self) -> f64 {
        self.damping
    }
    fn scaled_density(&self) -> Option<f64> {
        self.density
    }
    fn elasticity(&self) -> Option<ElasticityParameters> {
        self.unnormalized().elasticity
    }
    fn damping(&self) -> f64 {
        self.unnormalized().damping
    }
    fn density(&self) -> Option<f64> {
        self.unnormalized().density
    }
}

impl Deformable for ShellMaterial {
    fn scale(&self) -> f64 {
        match self.properties {
            ShellProperties::Deformable { deformable } => deformable.scale(),
            _ => 1.0,
        }
    }
    fn scaled_elasticity(&self) -> Option<ElasticityParameters> {
        match self.properties {
            ShellProperties::Deformable { deformable } => deformable.scaled_elasticity(),
            _ => None,
        }
    }
    fn scaled_damping(&self) -> f64 {
        match self.properties {
            ShellProperties::Deformable { deformable } => deformable.scaled_damping(),
            _ => 0.0,
        }
    }
    fn scaled_density(&self) -> Option<f64> {
        match self.properties {
            ShellProperties::Rigid { density } => Some(density),
            ShellProperties::Deformable { deformable } => deformable.scaled_density(),
            ShellProperties::Fixed => None,
        }
    }
    fn elasticity(&self) -> Option<ElasticityParameters> {
        match self.properties {
            ShellProperties::Deformable { deformable } => deformable.elasticity(),
            _ => None,
        }
    }
    fn damping(&self) -> f64 {
        match self.properties {
            ShellProperties::Deformable { deformable } => deformable.damping(),
            _ => 0.0,
        }
    }
    fn density(&self) -> Option<f64> {
        match self.properties {
            ShellProperties::Rigid { density } => Some(density),
            ShellProperties::Deformable { deformable } => deformable.density(),
            ShellProperties::Fixed => None,
        }
    }
}
impl Deformable for SolidMaterial {
    fn scale(&self) -> f64 {
        self.properties.deformable.scale()
    }
    fn scaled_elasticity(&self) -> Option<ElasticityParameters> {
        self.properties.deformable.scaled_elasticity()
    }
    fn scaled_damping(&self) -> f64 {
        self.properties.deformable.scaled_damping()
    }
    fn scaled_density(&self) -> Option<f64> {
        self.properties.deformable.scaled_density()
    }
    fn elasticity(&self) -> Option<ElasticityParameters> {
        self.properties.deformable.elasticity()
    }
    fn damping(&self) -> f64 {
        self.properties.deformable.damping()
    }
    fn density(&self) -> Option<f64> {
        self.properties.deformable.density()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Verify that that a normalized material can be unnormalized into its
    /// original state. In other words verify that `unnormalized` is the inverse
    /// of `normalized`.
    #[test]
    fn deformable_material_normalization() {
        let mat = DeformableProperties::default()
            .with_elasticity(ElasticityParameters {
                lambda: 123.0,
                mu: 0.01,
            })
            .with_density(100.0)
            .with_damping(0.125, 0.0725);

        let normalized_mat = mat.normalized();
        let unnormalized_mat = normalized_mat.unnormalized();

        assert_eq!(unnormalized_mat, mat);

        let renormalized_mat = unnormalized_mat.normalized();

        assert_eq!(renormalized_mat, normalized_mat);
    }
}
