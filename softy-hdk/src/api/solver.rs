use geo::algo::split::TypedMesh;
use geo::attrib::*;
use geo::topology::*;
use softy::{fem, Error, Mesh, PointCloud, SimResult, TetMesh, TriMesh};

// NOTE: We avoid using associated types here because of a compiler bug:
// https://github.com/rust-lang/rust/issues/23856
// This bug makes using this trait very verbose with associated types.

pub trait Solver: Send {
    fn solve(&mut self) -> SimResult;
    fn mesh(&self) -> Mesh;
    fn solid_mesh(&self) -> TetMesh;
    fn shell_mesh(&self) -> TriMesh;
    fn num_vertices(&self) -> usize;
    fn update_vertices(&mut self, pts: &PointCloud) -> Result<(), Error>;
    fn set_interrupter(&mut self, interrupter: Box<dyn FnMut() -> bool + Send>);
}

impl<S> Solver for fem::nl::Solver<S, f64>
where
    S: Send + fem::nl::NLSolver<fem::nl::NLProblem<f64>, f64>,
{
    #[inline]
    fn solve(&mut self) -> SimResult {
        let result = self.step();
        if let Ok(result) = result.as_ref() {
            if let Err(err) = self.log_result(result) {
                log::warn!("Failed to log result: {}", err);
            }
        }
        result.into()
    }
    #[inline]
    fn mesh(&self) -> Mesh {
        fem::nl::Solver::mesh(self)
    }
    #[inline]
    fn solid_mesh(&self) -> TetMesh {
        let mesh = self.mesh();
        let meshes = mesh.split_into_typed_meshes();
        for unimesh in meshes.into_iter() {
            if let TypedMesh::Tet(mut tetmesh) = unimesh {
                tetmesh
                    .reset_attrib_to_default::<i32, CellIndex>("object_type", 0)
                    .unwrap();
                return tetmesh;
            }
        }
        return TetMesh::default();
    }
    #[inline]
    fn shell_mesh(&self) -> TriMesh {
        let mesh = self.mesh();
        let meshes = mesh.split_into_typed_meshes();
        for unimesh in meshes.into_iter() {
            if let TypedMesh::Tri(mut trimesh) = unimesh {
                trimesh
                    .reset_attrib_to_default::<i32, FaceIndex>("object_type", 1)
                    .unwrap();
                return trimesh;
            }
        }
        return TriMesh::default();
    }
    #[inline]
    fn num_vertices(&self) -> usize {
        self.mesh().num_vertices()
    }
    #[inline]
    fn update_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        self.update_vertices(pts)
    }
    #[inline]
    fn set_interrupter(&mut self, interrupter: Box<dyn FnMut() -> bool + Send>) {
        self.set_coarse_interrupter(interrupter);
    }
}
