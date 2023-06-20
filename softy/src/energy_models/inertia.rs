//use rayon::prelude::*;

mod rigid_shell;
mod soft_shell;
mod soft_solid;

pub(crate) use soft_shell::*;
pub(crate) use soft_solid::*;

pub trait Inertia<'a, E> {
    fn inertia(&'a self) -> E;
}
