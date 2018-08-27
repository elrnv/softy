#pragma once

#include <hdkrs/hdkrs.h>
#include <hdkrs/pointer.h>

class GU_Detail;

namespace hdkrs {
namespace mesh {

/**
 * Add the given meshes into the given detail
 */
static void add_polymesh(GU_Detail* detail, OwnedPtr<PolyMesh> polymesh);
static void add_tetmesh(GU_Detail* detail, OwnedPtr<TetMesh> tetmesh);

static OwnedPtr<TetMesh> build_tetmesh(const GU_Detail *detail);

static OwnedPtr<PolyMesh> build_polymesh(const GU_Detail* detail);

} // namespace mesh
} // namespace hdkrs

#include "mesh-inl.h"
