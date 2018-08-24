#pragma once

#include <hdkrs/hdkrs.h>

class GU_Detail;

namespace mesh {

/**
 * Add the given meshes into the given detail
 */
void add_polymesh(GU_Detail* detail, hdkrs::PolyMesh *polymesh);
void add_tetmesh(GU_Detail* detail, hdkrs::TetMesh *tetmesh);

hdkrs::TetMesh *build_tetmesh(const GU_Detail *detail);

hdkrs::PolyMesh *build_polymesh(const GU_Detail* detail);

} // namespace mesh

#include "mesh-inl.h"
