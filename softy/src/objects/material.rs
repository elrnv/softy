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

    pub fn normalized(mut self) -> ShellMaterial {
        match &mut self.properties {
            ShellProperties::Deformable { ref mut deformable } => {
                *deformable = deformable.normalized();
                self
            }
            _ => self
        }
    }

    pub fn damping(&self) -> Option<f64> {
        match self.properties {
            ShellProperties::Rigid { .. } => None,
            ShellProperties::Deformable { deformable } => Some(deformable.damping),
            ShellProperties::Fixed => None,
        }
    }
    pub fn density(&self) -> Option<f64> {
        match self.properties {
            ShellProperties::Rigid { density } => Some(density),
            ShellProperties::Deformable { deformable } => deformable.density,
            ShellProperties::Fixed => None,
        }
    }
}

impl SolidMaterial {
    pub fn solid(id: usize, deformable: DeformableProperties, volume_preservation: bool) -> Self {
        Material {
            id,
            properties: SolidProperties {
                deformable,
                volume_preservation,
            },
        }
    }

    pub fn with_id(mut self, id: usize) -> SolidMaterial {
        self.id = id;
        self
    }
    pub fn with_elasticity(mut self, elasticity: ElasticityParameters) -> SolidMaterial {
        self.properties.deformable.elasticity = Some(elasticity);
        self
    }
    pub fn with_density(mut self, density: f64) -> SolidMaterial {
        self.properties.deformable.density = Some(density);
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

    /// Scale used to adjust internal material properties to be closer to 1.0.
    pub fn scale(&self) -> f64 {
        self.properties.deformable.scale()
    }
    /// The exact elasticity parameters used by solver. These are typically
    /// scaled to be closer to 1.0.
    pub fn scaled_elasticity(&self) -> Option<ElasticityParameters> {
        self.properties.deformable.elasticity
    }
    /// The exact damping parameter used by solver. This is typically scaled to
    /// be closer to 1.0.
    pub fn scaled_damping(&self) -> f64 {
        self.properties.deformable.damping
    }
    /// The exact density parameter used by solver. This is typically scaled to
    /// be closer to 1.0.
    pub fn scaled_density(&self) -> Option<f64> {
        self.properties.deformable.density
    }
    /// The elasticity parameters provided in the input.
    pub fn elasticity(&self) -> Option<ElasticityParameters> {
        self.properties.deformable.unnormalized().elasticity
    }
    /// The damping parameter provided in the input.
    pub fn damping(&self) -> f64 {
        self.properties.deformable.unnormalized().damping
    }
    /// The density parameter provided in the input.
    pub fn density(&self) -> Option<f64> {
        self.properties.deformable.unnormalized().density
    }
    pub fn volume_preservation(&self) -> bool {
        self.properties.volume_preservation
    }
}

impl<P: Default> Material<P> {
    pub fn new(id: usize) -> Material<P> {
        Material { id, properties: Default::default() }
    }
}

impl SolidProperties {
    pub fn deformable(self, deformable: DeformableProperties) -> SolidProperties {
        SolidProperties { deformable, ..self }
    }
    pub fn volume_preservation(self, volume_preservation: bool) -> SolidProperties {
        SolidProperties {
            volume_preservation,
            ..self
        }
    }
}

impl ShellProperties {
    pub fn fixed() -> ShellProperties {
        ShellProperties::Fixed
    }
    pub fn rigid(density: f64) -> ShellProperties {
        ShellProperties::Rigid { density }
    }
    pub fn deformable(deformable: DeformableProperties) -> ShellProperties {
        ShellProperties::Deformable { deformable }
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
    pub fn with_damping(self, mut damping: f64, time_step: Option<f64>) -> DeformableProperties {
        damping *= if let Some(dt) = time_step {
            if dt != 0.0 {
                1.0 / dt
            } else {
                0.0
            }
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
        } else {
            if let Some(ref mut density) = density {
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
                scale = if damping > 0.0 {
                    1.0 / damping
                } else {
                    1.0
                };
                damping *= scale;
            }
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
            *elasticity = elasticity.scaled(1.0/scale);
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
            .with_damping(0.125, Some(0.0725));

        let normalized_mat = mat.normalized();
        let unnormalized_mat = normalized_mat.unnormalized();

        assert_eq!(unnormalized_mat, mat);

        let renormalized_mat = unnormalized_mat.normalized();

        assert_eq!(renormalized_mat, normalized_mat);
    }
}
