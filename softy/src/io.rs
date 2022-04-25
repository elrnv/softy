use std::fs::File;
use std::path::Path;

use thiserror::Error;

use crate::fem::nl::SimParams as NLParams;
use crate::Material;

#[derive(Error, Debug)]
pub enum LoadConfigError {
    #[error("IO")]
    IO(#[from] std::io::Error),
    #[error("Parse")]
    Parse(#[from] ron::error::SpannedError),
}

pub fn load_nl_params(path: impl AsRef<Path>) -> std::result::Result<NLParams, LoadConfigError> {
    let f = File::open(path.as_ref())?;
    Ok(ron::de::from_reader(f)?)
}

pub fn load_material(path: impl AsRef<Path>) -> std::result::Result<Material, LoadConfigError> {
    let f = File::open(path.as_ref())?;
    Ok(ron::de::from_reader(f)?)
}
