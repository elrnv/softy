#pragma once

#include <testhdk.h>

class GU_Detail;

namespace mesh {

/**
 * Add the given meshes into the given detail
 */
void add_polymesh(GU_Detail* detail, test::PolyMesh *polymesh);
void add_tetmesh(GU_Detail* detail, test::TetMesh *tetmesh);

test::TetMesh *build_tetmesh(const GU_Detail *detail);

test::PolyMesh *build_polymesh(const GU_Detail* detail);

} // namespace mesh
