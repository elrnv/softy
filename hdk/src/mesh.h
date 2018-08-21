#pragma once

#include <implicitshdk.h>

class GU_Detail;

namespace mesh {

/**
 * Add the given meshes into the given detail
 */
void add_polymesh(GU_Detail* detail, implicits::PolyMesh *polymesh);
void add_tetmesh(GU_Detail* detail, implicits::TetMesh *tetmesh);

implicits::TetMesh *build_tetmesh(const GU_Detail *detail);

implicits::PolyMesh *build_polymesh(const GU_Detail* detail);

} // namespace mesh
