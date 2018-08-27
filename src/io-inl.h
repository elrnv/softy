#pragma once

#include <boost/variant.hpp>

namespace hdkrs {
namespace io {

inline ByteBuffer ByteBuffer::write_vtk_mesh(OwnedPtr<PolyMesh> polymesh) {
    return hdkrs::make_polymesh_vtk_buffer(polymesh.get());
}

inline ByteBuffer ByteBuffer::write_vtk_mesh(OwnedPtr<TetMesh> tetmesh) {
    return hdkrs::make_tetmesh_vtk_buffer(tetmesh.get());
}

MeshVariant parse_vtk_mesh(const char * data, std::size_t size) {
    MeshVariant ret((boost::blank()));
    Mesh mesh = hdkrs::parse_vtk_mesh(data, size);
    switch (mesh.tag) {
        case Mesh::Tag::Tet:
            ret = OwnedPtr<TetMesh>(mesh.tet._0);
            break;
        case Mesh::Tag::Poly:
            ret = OwnedPtr<PolyMesh>(mesh.poly._0);
            break;
    }
    return ret;
}

} // namespace io
} // namespace hdkrs
