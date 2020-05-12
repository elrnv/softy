#pragma once

#include <boost/variant.hpp>

namespace hdkrs {
namespace io {

/*
 * VTK
 */

inline ByteBuffer ByteBuffer::write_vtk_mesh(OwnedPtr<HR_PolyMesh> polymesh) {
    return hr_make_polymesh_vtk_buffer(polymesh.get());
}

inline ByteBuffer ByteBuffer::write_vtk_mesh(OwnedPtr<HR_TetMesh> tetmesh) {
    return hr_make_tetmesh_vtk_buffer(tetmesh.get());
}

inline ByteBuffer ByteBuffer::write_vtk_mesh(OwnedPtr<HR_PointCloud> ptcloud) {
    return hr_make_pointcloud_vtk_buffer(ptcloud.get());
}

MeshVariant parse_vtk_mesh(const char * data, std::size_t size) {
    MeshVariant ret((boost::blank()));
    HR_Mesh mesh = hr_parse_vtk_mesh(data, size);
    switch (mesh.tag) {
        case HRMeshType::HR_TETMESH:
            ret = OwnedPtr<HR_TetMesh>(mesh.tetmesh);
            break;
        case HRMeshType::HR_POLYMESH:
            ret = OwnedPtr<HR_PolyMesh>(mesh.polymesh);
            break;
        default: break;
    }
    return ret;
}

/*
 * Obj
 */

inline ByteBuffer ByteBuffer::write_obj_mesh(OwnedPtr<HR_PolyMesh> polymesh) {
    return hr_make_polymesh_obj_buffer(polymesh.get());
}

inline ByteBuffer ByteBuffer::write_obj_mesh(OwnedPtr<HR_PointCloud> ptcloud) {
    return hr_make_pointcloud_obj_buffer(ptcloud.get());
}

MeshVariant parse_obj_mesh(const char * data, std::size_t size) {
    MeshVariant ret((boost::blank()));
    HR_Mesh mesh = hr_parse_obj_mesh(data, size);
    switch (mesh.tag) {
        case HRMeshType::HR_POLYMESH:
            ret = OwnedPtr<HR_PolyMesh>(mesh.polymesh);
            break;
        default: break;
    }
    return ret;
}

} // namespace io
} // namespace hdkrs
