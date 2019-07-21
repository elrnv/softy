#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Material {
    /// Material unique identifier.
    id: usize,
    /// Parameters determining the elastic behaviour of a simulated solid. If `None`, we will look
    /// for elasticity parameters in the mesh.
    elasticity: Option<ElasticityParameters>,
    /// The density of the material. If `None`, we will look for a density attribute in the mesh.
    density: Option<f64>,
    /// Volume preservation sets the material to be globally incompressible, if set to `true`. In
    /// contrast to Bulk Modulus, this parameter affects global incompressibility,
    /// while Bulk Modulus affects *local* incompressibility (on a per element level).
    volume_preservation: bool,
    /// Coefficient measuring the amount of artificial viscosity as dictated by the Rayleigh
    /// damping model. This value should be premultiplied by the timestep reciprocal to save
    /// passing the time step around to elastic energy models which are otherwise independent of
    /// time step.
    damping: f64,
    /// Scaling factor used to adjust the magnitudes of the parameters to be closer to 1.0.
    scale: f64,
}

impl Material {
    pub fn new() -> Material {
        Material::default()
    }
    pub fn elasticity(self, elasticity: ElasticityParameters) -> Material {
        Material {
            elasticity: Some(elasticity),
            ..self
        }
    }
    pub fn density(self, density: f64) -> Material {
        Material {
            density: Some(density),
            ..self
        }
    }
    pub fn volume_preservation(self, volume_preservation: bool) -> Material {
        Material {
            volume_preservation,
            ..self
        }
    }
    pub fn damping(self, mut damping: f64, time_step: Option<f64>) -> Material {
        damping *= if let Some(dt) = time_step {
            if dt != 0.0 {
                1.0 / dt;
            } else {
                0.0;
            }
        } else {
            0.0
        };

        Material {
            damping,
            ..self
        }
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Rescale parameters uniformly to be closer to 1.0.
    pub fn normalized(self) -> Material {
        Material {
            mut elasticity,
            mut density,
            mut damping,
            mut scale,
            volume_preservation,
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

        Material {
            elasticity,
            density,
            damping,
            scale,
            volume_preservation,
        }
    }

    /// Undo normalization.
    pub fn unnormalized(self) -> Material {
        Material {
            mut elasticity,
            mut density,
            mut damping,
            mut scale,
            volume_preservation,
        } = self;

        if let Some(ref mut elasticity) = elasticity {
            *elasticity = elasticity.scaled(scale);
        }

        if let Some(ref mut density) = density {
            density *= scale;
        }

        damping *= scale;

        Material {
            elasticity,
            density,
            damping,
            scale: 1.0,
            volume_preservation,
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Material {
            elasticity: None, // Assuming variable elasticity
            density: None, // Assuming variable density
            volume_preservation: false,
            damping: 0.0,
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
