#pragma once

#include <hdkrs/mesh.h>
#include <hdkrs/hdkrs.h>
#include <boost/variant.hpp>

namespace hdkrs {
namespace io {

class ByteBuffer {
public:
    ~ByteBuffer() {
        hdkrs::ByteBuffer buf;
        buf.data = _data;
        buf.size = _size;
        free_byte_buffer(buf);
    }

    // Get pointer to the allocated buffer data.
    const char * data() { return _data; }

    // Get size of the allocated buffer (number of bytes).
    std::size_t size() { return _size; }

    /**
     * Read the given meshes into an owned buffer.
     */
    static ByteBuffer write_vtk_mesh(OwnedPtr<PolyMesh> polymesh);
    static ByteBuffer write_vtk_mesh(OwnedPtr<TetMesh> tetmesh);

    ByteBuffer(hdkrs::ByteBuffer buf) : _data(buf.data), _size(buf.size) {}
private:
    ByteBuffer(const ByteBuffer&) = delete;
    ByteBuffer(ByteBuffer&&) = delete;

    const char * _data;
    std::size_t  _size;
};

using MeshVariant = boost::variant<OwnedPtr<PolyMesh>, OwnedPtr<TetMesh>, boost::blank>;

MeshVariant parse_vtk_mesh(const char * data, std::size_t size);

} // namespace io
} // namespace hdkrs

#include "io-inl.h"
