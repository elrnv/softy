#pragma once

#include <GU/GU_Detail.h>
#include <GEO/GEO_PrimTetrahedron.h>
#include <GEO/GEO_PrimPoly.h>
#include <GEO/GEO_PolyCounts.h>
#include <GA/GA_PageHandle.h>

#include <vector>
#include <cassert>

namespace hdkrs {

// Implement OwnedPtr specializations

template<>
inline OwnedPtr<PolyMesh>::~OwnedPtr() {
    free_polymesh(_ptr);
}

template<>
inline OwnedPtr<TetMesh>::~OwnedPtr() {
    free_tetmesh(_ptr);
}

template<>
inline OwnedPtr<PointCloud>::~OwnedPtr() {
    free_tetmesh(_ptr);
}

namespace mesh {

inline std::ostream& operator<<(std::ostream& out, AttribLocation where) {
    switch (where) {
        case AttribLocation::Vertex: out << "Vertex"; break;
        case AttribLocation::Face: out << "Face"; break;
        case AttribLocation::Cell: out << "Cell"; break;
        case AttribLocation::FaceVertex: out << "FaceVertex"; break;
        case AttribLocation::CellVertex: out << "CellVertex"; break;
        default: break;
    }
    return out;
}

namespace { // Implementation details

#define ADD_NUM_ATTRIB_IMPL(MTYPE, TYPE, FN) \
    void add_attrib( \
            MTYPE *m, \
            AttribLocation where, \
            const char *name, \
            std::size_t tuple_size, \
            const std::vector<TYPE> &data) \
    { \
        FN( m, where, name, tuple_size, data.size(), data.data() ); \
    }

ADD_NUM_ATTRIB_IMPL(PointCloud, int8, add_pointcloud_attrib_i8)
ADD_NUM_ATTRIB_IMPL(PointCloud, int32, add_pointcloud_attrib_i32)
ADD_NUM_ATTRIB_IMPL(PointCloud, int64_t, add_pointcloud_attrib_i64)
ADD_NUM_ATTRIB_IMPL(PointCloud, fpreal32, add_pointcloud_attrib_f32)
ADD_NUM_ATTRIB_IMPL(PointCloud, fpreal64, add_pointcloud_attrib_f64)

ADD_NUM_ATTRIB_IMPL(PolyMesh, int8, add_polymesh_attrib_i8)
ADD_NUM_ATTRIB_IMPL(PolyMesh, int32, add_polymesh_attrib_i32)
ADD_NUM_ATTRIB_IMPL(PolyMesh, int64_t, add_polymesh_attrib_i64)
ADD_NUM_ATTRIB_IMPL(PolyMesh, fpreal32, add_polymesh_attrib_f32)
ADD_NUM_ATTRIB_IMPL(PolyMesh, fpreal64, add_polymesh_attrib_f64)

ADD_NUM_ATTRIB_IMPL(TetMesh, int8, add_tetmesh_attrib_i8)
ADD_NUM_ATTRIB_IMPL(TetMesh, int32, add_tetmesh_attrib_i32)
ADD_NUM_ATTRIB_IMPL(TetMesh, int64_t, add_tetmesh_attrib_i64)
ADD_NUM_ATTRIB_IMPL(TetMesh, fpreal32, add_tetmesh_attrib_f32)
ADD_NUM_ATTRIB_IMPL(TetMesh, fpreal64, add_tetmesh_attrib_f64)

void add_attrib(
        PointCloud *ptcloud,
        AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<const char *> &strings,
        const std::vector<int64_t> &indices)
{
    add_pointcloud_attrib_str(
            ptcloud, where, name, tuple_size, strings.size(),
            strings.data(), indices.size(), indices.data());
}


void add_attrib(
        PolyMesh *polymesh,
        AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<const char *> &strings,
        const std::vector<int64_t> &indices)
{
    add_polymesh_attrib_str(
            polymesh, where, name, tuple_size, strings.size(),
            strings.data(), indices.size(), indices.data());
}

void add_attrib(
        TetMesh *tetmesh,
        AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<const char *> &strings,
        const std::vector<int64_t> &indices)
{
    add_tetmesh_attrib_str(
            tetmesh, where, name, tuple_size, strings.size(),
            strings.data(), indices.size(), indices.data());
}

template<typename T>
GA_PrimitiveTypeId mesh_prim_type_id();

template<>
GA_PrimitiveTypeId mesh_prim_type_id<PolyMesh>() { return GA_PRIMPOLY; }

template<>
GA_PrimitiveTypeId mesh_prim_type_id<TetMesh>() { return GA_PRIMTETRAHEDRON; }

template<typename T>
AttribLocation mesh_prim_attrib_location();

template<>
AttribLocation mesh_prim_attrib_location<PolyMesh>() { return AttribLocation::Face; }

template<>
AttribLocation mesh_prim_attrib_location<TetMesh>() { return AttribLocation::Cell; }

template<typename T>
AttribLocation mesh_vertex_attrib_location();

template<>
AttribLocation mesh_vertex_attrib_location<PolyMesh>() { return AttribLocation::FaceVertex; }

template<>
AttribLocation mesh_vertex_attrib_location<TetMesh>() { return AttribLocation::CellVertex; }

// Mark all points and vectors in the given detail that intersect the primitives of interest.
std::pair<std::vector<bool>, std::vector<bool>>
mark_points_and_vertices(
        const GU_Detail *detail,
        GA_PrimitiveTypeId prim_type_id)
{
    std::vector<bool> points(detail->getNumPointOffsets(), false);
    std::vector<bool> vertices(detail->getNumVertexOffsets(), false);
    for ( GA_Offset prim_off : detail->getPrimitiveRange() )
    {
        const GEO_Primitive *prim = detail->getGEOPrimitive(prim_off);
        if (prim->getTypeId() == prim_type_id)
        {
            GA_Size num_verts = detail->getPrimitiveVertexCount(prim_off);
            for ( GA_Size idx = 0; idx < num_verts; ++idx ) {
                auto vtx_off = detail->getPrimitiveVertexOffset(prim_off, idx);
                vertices[vtx_off] = true;
                auto pt_off = detail->vertexPoint(vtx_off);
                points[pt_off] = true;
            }
        }
    }

    return std::make_pair(std::move(points), std::move(vertices));
}

template<typename T, typename M, typename S = T>
void
fill_prim_attrib(
        const GU_Detail *detail,
        const GA_AIFTuple *aif,
        const GA_Attribute *attrib,
        std::size_t tuple_size,
        std::size_t num_elem,
        M *mesh)
{
    std::vector<T> data(tuple_size*num_elem);
    int i = 0;
    for ( GA_Offset prim_off : detail->getPrimitiveRange() )
    {
        const GEO_Primitive *prim = detail->getGEOPrimitive(prim_off);
        if (prim->getTypeId() != mesh_prim_type_id<M>()) continue;
        for ( int k = 0, k_end = tuple_size; k < k_end; ++k ) {
            S val;
            aif->get(attrib, prim_off, val, k);
            data[tuple_size*i + k] = val;
        }
        i += 1;
    }

    auto name = attrib->getName().c_str();
    add_attrib(mesh, mesh_prim_attrib_location<M>(), name, tuple_size, data);
}

template<typename T, typename M, typename S = T>
void fill_point_attrib(
        const GU_Detail *detail,
        const GA_AIFTuple *aif,
        const GA_Attribute *attrib,
        std::size_t tuple_size,
        std::size_t num_elem,
        const std::vector<bool> &group,
        M *mesh)
{
    std::vector<T> data(tuple_size*num_elem);
    int i = 0;
    for ( GA_Offset pt_off : detail->getPointRange() )
    {
        if (!group[pt_off]) continue;
        for ( int k = 0, k_end = tuple_size; k < k_end; ++k ) {
            S val;
            aif->get(attrib, pt_off, val, k);
            data[tuple_size*i + k] = val;
        }
        i += 1;
    }

    auto name = attrib->getName().c_str();
    add_attrib(mesh, AttribLocation::Vertex, name, tuple_size, data);
}

template<typename T, typename M, typename S = T>
void fill_vertex_attrib(
        const GU_Detail *detail,
        const GA_AIFTuple *aif,
        const GA_Attribute *attrib,
        std::size_t tuple_size,
        std::size_t num_elem,
        const std::vector<bool> &group,
        M *mesh)
{
    std::vector<T> data(tuple_size*num_elem);
    int i = 0;
    for ( GA_Offset vtx_off : detail->getVertexRange() )
    {
        if (!group[vtx_off]) continue;
        for ( int k = 0, k_end = tuple_size; k < k_end; ++k ) {
            S val;
            aif->get(attrib, vtx_off, val, k);
            data[tuple_size*i + k] = val;
        }
        i += 1;
    }

    auto name = attrib->getName().c_str();
    add_attrib(mesh, mesh_vertex_attrib_location<M>(), name, tuple_size, data);
}

template<typename M>
void fill_prim_str_attrib(
        const GU_Detail *detail,
        const GA_AIFSharedStringTuple *aif,
        const GA_Attribute *attrib,
        std::size_t tuple_size,
        std::size_t num_elem,
        M *mesh)
{
    // Try with different types
    std::vector<int64_t> ids(aif->getTableEntries(attrib), -1);
    std::vector<const char *> strings;
    for (auto it = aif->begin(attrib); !it.atEnd(); ++it) {
        ids[it.getHandle()] = strings.size();
        strings.push_back( it.getString() );
    }

    std::vector<int64_t> indices(tuple_size*num_elem, -1);

    int i = 0;
    for ( GA_Offset prim_off : detail->getPrimitiveRange() )
    {
        const GEO_Primitive *prim = detail->getGEOPrimitive(prim_off);
        if (prim->getTypeId() == mesh_prim_type_id<M>())
        {
            for ( int k = 0, k_end = tuple_size; k < k_end; ++k ) {
                GA_StringIndexType handle = aif->getHandle(attrib, prim_off, k);
                indices[tuple_size*i + k] = handle > -1 ? ids[handle] : -1;
            }
            i += 1;
        }
    }

    auto name = attrib->getName().c_str();
    add_attrib(mesh, mesh_prim_attrib_location<M>(), name, tuple_size, strings, indices);
}

template<typename M>
void fill_point_str_attrib(
        const GU_Detail *detail,
        const GA_AIFSharedStringTuple *aif,
        const GA_Attribute *attrib,
        std::size_t tuple_size,
        std::size_t num_elem,
        const std::vector<bool> &group,
        M *mesh)
{
    // Try with different types
    std::vector<int64_t> ids(aif->getTableEntries(attrib), -1);
    std::vector<const char *> strings;
    for (auto it = aif->begin(attrib); !it.atEnd(); ++it) {
        ids[it.getHandle()] = strings.size();
        strings.push_back( it.getString() );
    }

    std::vector<int64_t> indices(tuple_size*num_elem, -1);

    int i = 0;
    for ( GA_Offset pt_off : detail->getPointRange() )
    {
        if (!group[pt_off]) continue;
        for ( int k = 0, k_end = tuple_size; k < k_end; ++k ) {
            GA_StringIndexType handle = aif->getHandle(attrib, pt_off, k);
            indices[tuple_size*i + k] = handle > -1 ? ids[handle] : -1;
        }
        i += 1;
    }

    auto name = attrib->getName().c_str();
    add_attrib(mesh, AttribLocation::Vertex, name, tuple_size, strings, indices);
}

template<typename M>
void fill_vertex_str_attrib(
        const GU_Detail *detail,
        const GA_AIFSharedStringTuple *aif,
        const GA_Attribute *attrib,
        std::size_t tuple_size,
        std::size_t num_elem,
        const std::vector<bool> &group,
        M *mesh)
{
    // Try with different types
    std::vector<int64_t> ids(aif->getTableEntries(attrib), -1);
    std::vector<const char *> strings;
    for (auto it = aif->begin(attrib); !it.atEnd(); ++it) {
        ids[it.getHandle()] = strings.size();
        strings.push_back( it.getString() );
    }

    std::vector<int64_t> indices(tuple_size*num_elem, -1);

    int i = 0;
    for ( GA_Offset vtx_off : detail->getVertexRange() )
    {
        if (!group[vtx_off]) continue;
        for ( int k = 0, k_end = tuple_size; k < k_end; ++k ) {
            GA_StringIndexType handle = aif->getHandle(attrib, vtx_off, k);
            indices[tuple_size*i + k] = handle > -1 ? ids[handle] : -1;
        }
        i += 1;
    }

    auto name = attrib->getName().c_str();
    add_attrib(mesh, mesh_vertex_attrib_location<M>(), name, tuple_size, strings, indices);
}

template<typename M>
void transfer_primitive_attributes(const GU_Detail* detail, M* mesh, std::size_t num_prims) {
    // Get prim data attributes
    for (auto it = detail->getAttributeDict(GA_ATTRIB_PRIMITIVE).begin(GA_SCOPE_PUBLIC); !it.atEnd(); ++it)
    {
        GA_Attribute *attrib = it.attrib();
        std::size_t tuple_size = attrib->getTupleSize();
        {
            auto aif = attrib->getAIFTuple(); // array of data
            if ( aif )
            {
                switch (aif->getStorage(attrib)) {
                    case GA_STORE_BOOL:
                        fill_prim_attrib<int8, M, int32>(detail, aif, attrib, tuple_size, num_prims, mesh); break;
                    case GA_STORE_INT8:
                        fill_prim_attrib<int8, M, int32>(detail, aif, attrib, tuple_size, num_prims, mesh); break;
                    case GA_STORE_INT32:
                        fill_prim_attrib<int32>(detail, aif, attrib, tuple_size, num_prims, mesh); break;
                    case GA_STORE_INT64:
                        fill_prim_attrib<int64_t>(detail, aif, attrib, tuple_size, num_prims, mesh); break;
                    case GA_STORE_REAL32:
                        fill_prim_attrib<fpreal32>(detail, aif, attrib, tuple_size, num_prims, mesh); break;
                    case GA_STORE_REAL64:
                        fill_prim_attrib<fpreal64>(detail, aif, attrib, tuple_size, num_prims, mesh); break;
                    default: break; // do nothing
                }
            }
        }

        {
            auto aif = attrib->getAIFSharedStringTuple(); // array of strings
            if ( aif ) {
                aif->compactStorage(attrib);
                fill_prim_str_attrib(detail, aif, attrib, tuple_size, num_prims, mesh);
            }
        }

        // don't know how to handle these yet.
        //aif = attrib->getAIFNumericArray(); // variable sized array
        //aif = attrib->getAIFSharedStringArray(); // variable sized array of strings
    }
}

// Transfer attributes from points marked in the given pt_grp
template<typename M>
void transfer_point_attributes(const GU_Detail* detail, M* mesh, const std::vector<bool>& pt_grp)
{
    std::size_t num_points = std::count(pt_grp.begin(), pt_grp.end(), true);
    for (auto it = detail->getAttributeDict(GA_ATTRIB_POINT).begin(GA_SCOPE_PUBLIC); !it.atEnd(); ++it)
    {
        GA_Attribute *attrib = it.attrib();
        if (attrib->getTypeInfo() == GA_TYPE_POINT) // ignore position attribute
            continue;
        std::size_t tuple_size = attrib->getTupleSize();
        {
            auto aif = attrib->getAIFTuple(); // array of data
            if ( aif )
            {
                switch (aif->getStorage(attrib)) {
                    case GA_STORE_BOOL:
                        fill_point_attrib<int8, M, int32>(detail, aif, attrib, tuple_size, num_points, pt_grp, mesh); break;
                    case GA_STORE_INT8:
                        fill_point_attrib<int8, M, int32>(detail, aif, attrib, tuple_size, num_points, pt_grp, mesh); break;
                    case GA_STORE_INT32:
                        fill_point_attrib<int32>(detail, aif, attrib, tuple_size, num_points, pt_grp, mesh); break;
                    case GA_STORE_INT64:
                        fill_point_attrib<int64_t>(detail, aif, attrib, tuple_size, num_points, pt_grp, mesh); break;
                    case GA_STORE_REAL32:
                        fill_point_attrib<fpreal32>(detail, aif, attrib, tuple_size, num_points, pt_grp, mesh); break;
                    case GA_STORE_REAL64:
                        fill_point_attrib<fpreal64>(detail, aif, attrib, tuple_size, num_points, pt_grp, mesh); break;
                    default: break; // do nothing
                }
            }
        }


        {
            auto aif = attrib->getAIFSharedStringTuple(); // array of strings
            if ( aif ) {
                aif->compactStorage(attrib);
                fill_point_str_attrib(detail, aif, attrib, tuple_size, num_points, pt_grp, mesh);
            }
        }

        // don't know how to handle these yet.
        //aif = attrib->getAIFNumericArray(); // variable sized array
        //aif = attrib->getAIFSharedStringArray(); // variable sized array of strings
    }
}

// Transfer attributes from vertices marked in the given vtx_grp
template<typename M>
void transfer_vertex_attributes(const GU_Detail* detail, M* mesh, const std::vector<bool>& vtx_grp)
{
    std::size_t num_vertices = std::count(vtx_grp.begin(), vtx_grp.end(), true);
    for (auto it = detail->getAttributeDict(GA_ATTRIB_VERTEX).begin(GA_SCOPE_PUBLIC); !it.atEnd(); ++it)
    {
        GA_Attribute *attrib = it.attrib();
        std::size_t tuple_size = attrib->getTupleSize();
        {
            auto aif = attrib->getAIFTuple(); // array of data
            if ( aif )
            {
                switch (aif->getStorage(attrib)) {
                    case GA_STORE_BOOL:
                        fill_vertex_attrib<int8, M, int32>(detail, aif, attrib, tuple_size, num_vertices, vtx_grp, mesh); break;
                    case GA_STORE_INT8:
                        fill_vertex_attrib<int8, M, int32>(detail, aif, attrib, tuple_size, num_vertices, vtx_grp, mesh); break;
                    case GA_STORE_INT32:
                        fill_vertex_attrib<int32>(detail, aif, attrib, tuple_size, num_vertices, vtx_grp, mesh); break;
                    case GA_STORE_INT64:
                        fill_vertex_attrib<int64_t>(detail, aif, attrib, tuple_size, num_vertices, vtx_grp, mesh); break;
                    case GA_STORE_REAL32:
                        fill_vertex_attrib<fpreal32>(detail, aif, attrib, tuple_size, num_vertices, vtx_grp, mesh); break;
                    case GA_STORE_REAL64:
                        fill_vertex_attrib<fpreal64>(detail, aif, attrib, tuple_size, num_vertices, vtx_grp, mesh); break;
                    default: break; // do nothing
                }
            }
        }


        {
            auto aif = attrib->getAIFSharedStringTuple(); // array of strings
            if ( aif ) {
                aif->compactStorage(attrib);
                fill_vertex_str_attrib(detail, aif, attrib, tuple_size, num_vertices, vtx_grp, mesh);
            }
        }

        // don't know how to handle these yet.
        //aif = attrib->getAIFNumericArray(); // variable sized array
        //aif = attrib->getAIFSharedStringArray(); // variable sized array of strings
    }
}

template<typename M>
void transfer_attributes(const GU_Detail* detail, M* mesh, std::size_t num_prims)
{
    transfer_primitive_attributes(detail, mesh, num_prims);

    std::vector<bool> pt_grp, vtx_grp;
    std::tie(pt_grp, vtx_grp) = mark_points_and_vertices(detail, mesh_prim_type_id<M>());

    transfer_point_attributes(detail, mesh, pt_grp);
    transfer_vertex_attributes(detail, mesh, vtx_grp);
}

template<typename HandleType, typename ArrayType>
void fill_attrib(HandleType h, ArrayType arr, GA_Offset startoff) {
    std::size_t i = 0;
    auto n = startoff + (arr.size/arr.tuple_size);
    for ( GA_Offset off = startoff; off < n; ++off, ++i ) {
        for ( int j = 0; j < arr.tuple_size; ++j ) {
            h.set(off, j, arr.array[arr.tuple_size*i + j]);
        }
    }
}

/** Retrieve attributes from the mesh using the given iterator.
 */
void retrieve_attributes(GU_Detail *detail, GA_Offset startoff, AttribIter *it, GA_AttributeOwner owner) {
    while ( it ) { // it could be null, but it doesn't change
        auto attrib = attrib_iter_next(it);
        if (!attrib) break;
        auto name = attrib_name(attrib);
        auto type = attrib_data_type(attrib);
        if (type == DataType::I8 ) {
            auto arr = attrib_data_i8(attrib);
            auto h = GA_RWHandleC(detail->addTuple(GA_STORE_INT8, owner, name, arr.tuple_size));
            fill_attrib(h, arr, startoff);
            free_attrib_data_i8(arr);
        } else if (type == DataType::I32 ) {
            auto arr = attrib_data_i32(attrib);
            auto h = GA_RWHandleI(detail->addTuple(GA_STORE_INT32, owner, name, arr.tuple_size));
            fill_attrib(h, arr, startoff);
            free_attrib_data_i32(arr);
        } else if (type == DataType::I64 ) {
            auto arr = attrib_data_i64(attrib);
            auto h = GA_RWHandleID(detail->addTuple(GA_STORE_INT64, owner, name, arr.tuple_size));
            fill_attrib(h, arr, startoff);
            free_attrib_data_i64(arr);
        } else if (type == DataType::F32 ) {
            auto arr = attrib_data_f32(attrib);
            auto h = GA_RWHandleF(detail->addTuple(GA_STORE_REAL32, owner, name, arr.tuple_size));
            fill_attrib(h, arr, startoff);
            free_attrib_data_f32(arr);
        } else if (type == DataType::F64 ) {
            auto arr = attrib_data_f64(attrib);
            auto h = GA_RWHandleD(detail->addTuple(GA_STORE_REAL64, owner, name, arr.tuple_size));
            fill_attrib(h, arr, startoff);
            free_attrib_data_f64(arr);
        } else if (type == DataType::Str ) {
            auto arr = attrib_data_str(attrib);
            auto h = GA_RWHandleS(detail->addTuple(GA_STORE_STRING, owner, name, arr.tuple_size));
            fill_attrib(h, arr, startoff);
            free_attrib_data_str(arr);
        }
        free_attribute(attrib);
    }
    free_attrib_iter(it);
}


template<typename HandleType, typename ArrayType>
void update_attrib(HandleType h, ArrayType arr) {
    int n = (arr.size/arr.tuple_size);
    for ( int j = 0; j < arr.tuple_size; ++j ) {
        h.setBlockFromIndices(GA_Index(0), GA_Size(n), arr.array, arr.tuple_size, j);
    }
}

/** Update attributes of the mesh using the given iterator.
 */
void update_attributes(GU_Detail *detail, AttribIter *it, GA_AttributeOwner owner) {
    while ( it ) { // it could be null, but it doesn't change
        auto attrib = attrib_iter_next(it);
        if (!attrib) break;
        auto name = attrib_name(attrib);
        auto type = attrib_data_type(attrib);
        if (type == DataType::I8 ) {
            auto arr = attrib_data_i8(attrib);
            auto h = GA_RWHandleC(detail->addTuple(GA_STORE_INT8, owner, name, arr.tuple_size));
            update_attrib(h, arr);
            free_attrib_data_i8(arr);
        } else if (type == DataType::I32 ) {
            auto arr = attrib_data_i32(attrib);
            auto h = GA_RWHandleI(detail->addTuple(GA_STORE_INT32, owner, name, arr.tuple_size));
            update_attrib(h, arr);
            free_attrib_data_i32(arr);
        } else if (type == DataType::I64 ) {
            auto arr = attrib_data_i64(attrib);
            auto h = GA_RWHandleID(detail->addTuple(GA_STORE_INT64, owner, name, arr.tuple_size));
            update_attrib(h, arr);
            free_attrib_data_i64(arr);
        } else if (type == DataType::F32 ) {
            auto arr = attrib_data_f32(attrib);
            auto h = GA_RWHandleF(detail->addTuple(GA_STORE_REAL32, owner, name, arr.tuple_size));
            update_attrib(h, arr);
            free_attrib_data_f32(arr);
        } else if (type == DataType::F64 ) {
            auto arr = attrib_data_f64(attrib);
            auto h = GA_RWHandleD(detail->addTuple(GA_STORE_REAL64, owner, name, arr.tuple_size));
            update_attrib(h, arr);
            free_attrib_data_f64(arr);
        } // String attributes are not yet supported by updates
        free_attribute(attrib);
    }
    free_attrib_iter(it);
}

} // namespace (static)

/**
 * Add a tetmesh to the current detail
 */
__attribute__((unused))
void add_tetmesh(GU_Detail* detail, OwnedPtr<TetMesh> tetmesh_ptr) {
    GA_Offset startvtxoff = GA_Offset(detail->getNumVertexOffsets());

    auto tetmesh = tetmesh_ptr.get();

    // add tets.
    if (tetmesh) {
        auto points = get_tetmesh_points(tetmesh);

        auto test_indices = get_tetmesh_indices(tetmesh);
        if (test_indices.size > 0) {
            std::vector<int> indices;
            for (std::size_t i = 0; i < test_indices.size; ++i) {
                indices.push_back(static_cast<int>(test_indices.array[i]));
            }

            GA_Offset startptoff = detail->appendPointBlock(points.size);
            for (exint pt_idx = 0; pt_idx < points.size; ++pt_idx) {
                GA_Offset ptoff = startptoff + pt_idx;
                detail->setPos3(ptoff, UT_Vector3(points.array[pt_idx]));
            }

            GA_Offset startprimoff = GEO_PrimTetrahedron::buildBlock(
                    detail, startptoff, detail->getNumPointOffsets(),
                    indices.size()/4, indices.data());

            retrieve_attributes(detail, startptoff, tetmesh_attrib_iter(tetmesh, AttribLocation::Vertex, 0), GA_ATTRIB_POINT);
            retrieve_attributes(detail, startprimoff, tetmesh_attrib_iter(tetmesh, AttribLocation::Cell, 0), GA_ATTRIB_PRIMITIVE);
            retrieve_attributes(detail, startvtxoff, tetmesh_attrib_iter(tetmesh, AttribLocation::CellVertex, 0), GA_ATTRIB_VERTEX);
        }
        free_point_array(points);
        free_index_array(test_indices);
    }
}

/**
 * Add a polymesh to the current detail
 */
__attribute__((unused))
void add_polymesh(GU_Detail* detail, OwnedPtr<PolyMesh> polymesh_ptr) {
    GA_Offset startvtxoff = GA_Offset(detail->getNumVertexOffsets());

    auto polymesh = polymesh_ptr.get();

    // add polygons
    if (polymesh) {
        auto points = get_polymesh_points(polymesh);

        auto test_indices = get_polymesh_indices(polymesh);
        if (test_indices.size > 0) {
            GA_Offset startptoff = detail->appendPointBlock(points.size);
            for (exint pt_idx = 0; pt_idx < points.size; ++pt_idx) {
                GA_Offset ptoff = startptoff + pt_idx;
                detail->setPos3(ptoff, UT_Vector3(points.array[pt_idx]));
            }

            GEO_PolyCounts polycounts;
            std::vector<int> poly_pt_numbers;
            int prev_n = test_indices.array[0];
            int num_polys_with_same_shape = 0;
            for (std::size_t i = 0; i < test_indices.size; ) {
                auto n = test_indices.array[i++];
                if (n != prev_n) {
                    polycounts.append(n, num_polys_with_same_shape);
                    num_polys_with_same_shape = 0;
                    prev_n = n;
                }
                num_polys_with_same_shape += 1;
                for (std::size_t j = 0; j < n; ++j, ++i) {
                    poly_pt_numbers.push_back(test_indices.array[i]);
                }
            }
            polycounts.append(prev_n, num_polys_with_same_shape); // append last set

            GA_Offset startprimoff = GEO_PrimPoly::buildBlock(
                    detail, startptoff, detail->getNumPointOffsets(),
                    polycounts, poly_pt_numbers.data());

            retrieve_attributes(detail, startprimoff, polymesh_attrib_iter(polymesh, AttribLocation::Face, 0), GA_ATTRIB_PRIMITIVE);
            retrieve_attributes(detail, startvtxoff, polymesh_attrib_iter(polymesh, AttribLocation::FaceVertex, 0), GA_ATTRIB_VERTEX);
            retrieve_attributes(detail, startptoff, polymesh_attrib_iter(polymesh, AttribLocation::Vertex, 0), GA_ATTRIB_POINT);
        }

        free_point_array(points);
        free_index_array(test_indices);
    }
}

/**
 * Add a ptcloud to the current detail
 */
__attribute__((unused))
void add_pointcloud(GU_Detail* detail, OwnedPtr<PointCloud> ptcloud_ptr) {
    auto ptcloud = ptcloud_ptr.get();

    if (ptcloud) {
        auto points = get_pointcloud_points(ptcloud);

        GA_Offset startptoff = detail->appendPointBlock(points.size);

        for (exint pt_idx = 0; pt_idx < points.size; ++pt_idx) {
            GA_Offset ptoff = startptoff + pt_idx;
            detail->setPos3(ptoff, UT_Vector3(points.array[pt_idx]));
        }

        retrieve_attributes(detail, startptoff, pointcloud_attrib_iter(ptcloud, AttribLocation::Vertex, 0), GA_ATTRIB_POINT);
        free_point_array(points);
    }
}

/**
 * Update points in the detail according to what's in the ptcloud
 */
__attribute__((unused))
void update_points(GU_Detail* detail, OwnedPtr<PointCloud> ptcloud_ptr) {
    auto ptcloud = ptcloud_ptr.get();

    if (ptcloud) {
        auto points = get_pointcloud_points(ptcloud);

        for (exint pt_idx = 0; pt_idx < points.size; ++pt_idx) {
            GA_Offset ptoff = detail->pointOffset(pt_idx);
            detail->setPos3(ptoff, UT_Vector3(points.array[pt_idx]));
        }

        update_attributes(detail, pointcloud_attrib_iter(ptcloud, AttribLocation::Vertex, 0), GA_ATTRIB_POINT);
        free_point_array(points);
    }
}

__attribute__((unused))
OwnedPtr<TetMesh> build_tetmesh(const GU_Detail *detail) {
    // Get tets for the body from the first input
    std::vector<double> tet_vertices;
    tet_vertices.reserve(detail->getNumPointOffsets());
    std::vector<std::size_t> tet_indices;
    tet_indices.reserve(detail->getNumVertexOffsets());

    for ( GA_Offset pt_off : detail->getPointRange() )
    {
        UT_Vector3 pt = detail->getPos3(pt_off);
        tet_vertices.push_back( static_cast<double>(pt[0]) );
        tet_vertices.push_back( static_cast<double>(pt[1]) );
        tet_vertices.push_back( static_cast<double>(pt[2]) );
    }

    std::size_t num_tets = 0;
    for ( GA_Offset prim_off : detail->getPrimitiveRange() )
    {
        const GEO_Primitive *prim = detail->getGEOPrimitive(prim_off);
        if (prim->getTypeId() == GA_PRIMTETRAHEDRON) {
            num_tets += 1;
            const GEO_PrimTetrahedron *tet = static_cast<const GEO_PrimTetrahedron*>(prim);
            tet_indices.push_back(detail->pointIndex(detail->vertexPoint(tet->fastVertexOffset(0))));
            tet_indices.push_back(detail->pointIndex(detail->vertexPoint(tet->fastVertexOffset(1))));
            tet_indices.push_back(detail->pointIndex(detail->vertexPoint(tet->fastVertexOffset(2))));
            tet_indices.push_back(detail->pointIndex(detail->vertexPoint(tet->fastVertexOffset(3))));
        }
    }

    // Only creating a mesh if there are tets.
    if (num_tets > 0) {
        TetMesh *tetmesh = make_tetmesh(tet_vertices.size(), tet_vertices.data(),
                                        tet_indices.size(), tet_indices.data());
        assert(tetmesh);

        transfer_attributes(detail, tetmesh, num_tets);
        return OwnedPtr<TetMesh>(tetmesh);
    }
    return OwnedPtr<TetMesh>(nullptr);
}

__attribute__((unused))
OwnedPtr<PolyMesh> build_polymesh(const GU_Detail* detail) {
    std::vector<double> poly_vertices;
    poly_vertices.reserve(detail->getNumPointOffsets());
    std::vector<std::size_t> poly_indices;
    poly_indices.reserve(detail->getNumVertexOffsets());

    for ( GA_Offset pt_off : detail->getPointRange() )
    {
        UT_Vector3 pos = detail->getPos3(pt_off);
        poly_vertices.push_back( static_cast<double>(pos[0]) );
        poly_vertices.push_back( static_cast<double>(pos[1]) );
        poly_vertices.push_back( static_cast<double>(pos[2]) );
    }

    std::size_t num_polys = 0;
    for ( GA_Offset prim_off : detail->getPrimitiveRange() )
    {
        const GEO_Primitive *prim = detail->getGEOPrimitive(prim_off);
        if (prim->getTypeId() == GA_PRIMPOLY) {
            num_polys += 1;
            const GEO_PrimPoly *poly = static_cast<const GEO_PrimPoly*>(prim);
            std::size_t num_verts = poly->getVertexCount();
            poly_indices.push_back(num_verts);
            for ( std::size_t i = 0; i < num_verts; ++i ) {
                GA_Index idx = detail->pointIndex(detail->vertexPoint(poly->getVertexOffset(i)));
                assert(GAisValid(idx));
                poly_indices.push_back(static_cast<std::size_t>(idx));
            }
        }
    }

    // Only creating a mesh if there are polys.
    if (num_polys > 0) {
        PolyMesh *polymesh = make_polymesh(poly_vertices.size(), poly_vertices.data(),
                                           poly_indices.size(), poly_indices.data());
        assert(polymesh);

        transfer_attributes(detail, polymesh, num_polys);
        return OwnedPtr<PolyMesh>(polymesh);
    }
    return OwnedPtr<PolyMesh>(nullptr);
}

__attribute__((unused))
OwnedPtr<PointCloud> build_pointcloud(const GU_Detail* detail) {
    std::vector<double> vertices(detail->getNumPoints());
    std::vector<bool> pt_grp(detail->getNumPointOffsets(), false);

    // We are gonna be smarter here and use block access to point data.
    GA_ROPageHandleV3 P_ph(detail->getP());

    if (P_ph.isValid()) {
        GA_Offset start, end;
        for (GA_Iterator it(detail->getPointRange()); it.blockAdvance(start, end); ) {
            P_ph.setPage(start);
            for (GA_Offset offset = start; offset < end; ++offset) {
                pt_grp[offset] = true;
                auto pos = P_ph.get(offset);
                vertices[detail->pointIndex(offset)] = static_cast<double>( pos[0]);
                vertices[detail->pointIndex(offset)+1] = static_cast<double>( pos[1]);
                vertices[detail->pointIndex(offset)+2] = static_cast<double>( pos[2]);
            }
        }
    }

    PointCloud *ptcloud = make_pointcloud(vertices.size(), vertices.data());
    assert(ptcloud);

    transfer_point_attributes(detail, ptcloud, pt_grp);
    return OwnedPtr<PointCloud>(ptcloud);
}

} // namespace mesh

} // namespace hdkrs
