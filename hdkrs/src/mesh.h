#pragma once

#include <hdkrs/hdkrs.h>
#include <hdkrs/pointer.h>

class GU_Detail;

namespace hdkrs {
namespace mesh {

/**
 * Add the given meshes into the given detail
 */
static void add_polymesh(GU_Detail* detail, OwnedPtr<HR_PolyMesh> polymesh);
static void add_tetmesh(GU_Detail* detail, OwnedPtr<HR_TetMesh> tetmesh);
static void add_pointcloud(GU_Detail* detail, OwnedPtr<HR_PointCloud> ptcloud);
static void update_points(GU_Detail* detail, OwnedPtr<HR_PointCloud> ptcloud);

static OwnedPtr<HR_TetMesh> build_tetmesh(const GU_Detail *detail);

static OwnedPtr<HR_PolyMesh> build_polymesh(const GU_Detail* detail);

static OwnedPtr<HR_PointCloud> build_pointcloud(const GU_Detail* detail);

} // namespace mesh
} // namespace hdkrs

#include "mesh-inl.h"
