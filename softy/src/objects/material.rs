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
    elasticity: Option<ElasticityParameters>,
    /// The density of the material. If `None`, we will look for a density attribute in the mesh.
    density: Option<f64>,
    /// Coefficient measuring the amount of artificial viscosity as dictated by the Rayleigh
    /// damping model. This value should be premultiplied by the timestep reciprocal to save
    /// passing the time step around to elastic energy models which are otherwise independent of
    /// time step.
    damping: f64,
    /// Scaling factor used to adjust the magnitudes of the parameters to be closer to 1.0.
    scale: f64,
}

impl Default for DeformableProperties {
    fn default() -> Self {
        DeformableProperties {
            elasticity: None, // Assuming variable elasticity
            density: None, // Assuming variable density
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
    Static,
    /// A rigid shell has 6 degrees of freedom: 3 for translation and 3 for rotation.
    Rigid { density: f64 } ,
    /// A deformable shell has a 3 degrees of freedom for every vertex.
    Deformable {
        deformable: DeformableProperties,
    }
}

impl Default for ShellProperties {
    fn default() -> Self {
        ShellProperties::Rigid
    }
}

/// Solids are always elastically deformable. For rigid solids, use shells,
/// because rigid solids don't require interior properties.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SolidProperties {
    /// Volume preservation sets the material to be globally incompressible, if set to `true`. In
    /// contrast to Bulk Modulus, this parameter affects global incompressibility,
    /// while Bulk Modulus affects *local* incompressibility (on a per element level).
    volume_preservation: bool,

    /// Common material properties shared among all deformable materials.
    deformable: DeformableProperties,
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

impl<P: Default> Material<P> {
    pub fn new() -> Material {
        Material {
            id: 0,
            properties: P::default(),
        }
    }
}

impl SolidProperties {
    pub fn deformable(self, deformable: DeformableProperties) -> SolidProperties {
        SolidProperties {
            deformable,
            ..self
        }
    }
    pub fn volume_preservation(self, volume_preservation: bool) -> SolidProperties {
        SolidProperties {
            volume_preservation,
            ..self
        }
    }
}

impl ShellProperties {
    pub fn rigid(self) -> ShellProperties {
        ShellProperties::Rigid
    }
    pub fn deformable(self, deformable: DeformableProperties) -> ShellProperties {
        ShellProperties::Deformable {
            deformable,
        }
    }
}

impl DeformableProperties {
    pub fn elasticity(self, elasticity: ElasticityParameters) -> DeformableProperties {
        DeformableProperties {
            elasticity: Some(elasticity),
            ..self
        }
    }
    pub fn density(self, density: f64) -> DeformableProperties {
        DeformableProperties {
            density: Some(density),
            ..self
        }
    }
    pub fn damping(self, mut damping: f64, time_step: Option<f64>) -> DeformableProperties {
        damping *= if let Some(dt) = time_step {
            if dt != 0.0 {
                1.0 / dt;
            } else {
                0.0;
            }
        } else {
            0.0
        };

        DeformableProperties {
            damping,
            ..self
        }
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Rescale parameters uniformly to be closer to 1.0.
    pub fn normalized(self) -> DeformableProperties {
        DeformableProperties {
            mut elasticity,
            mut density,
            mut damping,
            mut scale,
        } = self;

        if let Some(ref mut elasticity) = elasticity {
            scale = elasticity.normalize();
            density /= scale;
            damping /= scale;
        } else {
            if let Some(ref mut density) = density {
                scale = *density;
                *density = 1.0;
                damping /= scale;
            } else {
                scale = damping;
                damping = 1.0;
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
        DeformableProperties {
            mut elasticity,
            mut density,
            mut damping,
            mut scale,
        } = self;

        if let Some(ref mut elasticity) = elasticity {
            *elasticity = elasticity.scaled(scale);
        }

        if let Some(ref mut density) = density {
            density *= scale;
        }

        damping *= scale;

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
        let scale = self.shear;
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
            lambda: bulk - 2.0*shear/3.0
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
    /// Verify that that a normalized material can be unnormalized into its
    /// original state. In other words verify that `unnormalized` is the inverse
    /// of `normalized`.
    #[test]
    fn deformable_material_normalization() {
        let mat = DeformableMaterial::default()
            .elasticity(ElasticityParameters { lambda: 123.0, mu: 0.01 })
            .density(100.0)
            .damping(0.125, 0.0725);

        let normalized_mat = mat.normalized();
        let unnormalized_mat = normalized_mat.unnormalized();

        assert_eq!(unnormalized_mat, mat);

        let renormalized_mat = unnormalized_mat.normalized();

        assert_eq!(renormalized_mat, normalized_mat);
    }
}
