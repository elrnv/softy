#pragma once

namespace hdkrs {
namespace io {

ByteBuffer ByteBuffer::write_vtk_mesh(OwnedPtr<PolyMesh> polymesh) {
    return ByteBuffer(0,0);
}

ByteBuffer ByteBuffer::write_vtk_mesh(OwnedPtr<TetMesh> tetmesh) {
    return ByteBuffer(0,0);
}

MeshVariant parse_vtk_mesh(const char * data, std::size_t size) {
    return MeshVariant(OwnedPtr<TetMesh>(nullptr));
}

} // namespace io
} // namespace hdkrs
