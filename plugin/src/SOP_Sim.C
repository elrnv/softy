#include "SOP_Sim.h"

// Needed for template generation with the ds file.
#include "SOP_Sim.proto.h"

// Required for proper loading.
#include <UT/UT_DSOVersion.h>

#include <UT/UT_StringHolder.h>
#include <PRM/PRM_Include.h>
#include <PRM/PRM_TemplateBuilder.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <GEO/GEO_PrimTetrahedron.h>
#include <GEO/GEO_PrimPoly.h>
#include <GEO/GEO_PolyCounts.h>
#include <sim_api.h>

#include <vector>
#include <array>
#include <cassert>

const UT_StringHolder SOP_Sim::theSOPTypeName("hdk_sim"_sh);

// Register sop operator
void
newSopOperator(OP_OperatorTable *table)
{
    table->addOperator(new OP_Operator(
                SOP_Sim::theSOPTypeName,   // Internal name
                "Sim",                     // UI name
                SOP_Sim::myConstructor,    // How to build the SOP
                SOP_Sim::buildTemplates(), // My parameters
                0,                              // Min # of sources
                4,                              // Max # of sources
                nullptr,                        // Local variables
                OP_FLAG_GENERATOR));            // Flag it as generator
}


//parm {
//    name "rustlib"
//    cppname "RustLibPath"
//    label "Rust Library Path"
//    type string
//    default { "./" }
//}
static const char *theDsFile = R"THEDSFILE(
{
    name sim

}
)THEDSFILE";


PRM_Template *
SOP_Sim::buildTemplates()
{
    static PRM_TemplateBuilder templ("SOP_Sim.C"_sh, theDsFile);
    return templ.templates();
}

class SOP_SimVerb : public SOP_NodeVerb
{
    public:
        SOP_SimVerb() {}
        virtual ~SOP_SimVerb() {}

        virtual SOP_NodeParms *allocParms() const { return new SOP_SimParms(); }
        virtual UT_StringHolder name() const { return SOP_Sim::theSOPTypeName; }

        virtual CookMode cookMode(const SOP_NodeParms *parms) const { return COOK_GENERATOR; }

        virtual void cook(const CookParms &cookparms) const;

        static const SOP_NodeVerb::Register<SOP_SimVerb> theVerb;
};

const SOP_NodeVerb::Register<SOP_SimVerb> SOP_SimVerb::theVerb;

const SOP_NodeVerb *
SOP_Sim::cookVerb() const
{
    return SOP_SimVerb::theVerb.get();
}

std::ostream& operator<<(std::ostream& out, sim::AttribLocation where) {
    switch (where) {
        case sim::AttribLocation::Vertex: out << "Vertex"; break;
        case sim::AttribLocation::Face: out << "Face"; break;
        case sim::AttribLocation::Cell: out << "Cell"; break;
        case sim::AttribLocation::FaceVertex: out << "FaceVertex"; break;
        case sim::AttribLocation::CellVertex: out << "CellVertex"; break;
        default: break;
    }
    return out;
}

void add_attrib(
        sim::PolyMesh *polymesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<int8> &data)
{
    sim::add_polymesh_attrib_i8( polymesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::PolyMesh *polymesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<int32> &data)
{
    sim::add_polymesh_attrib_i32( polymesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::PolyMesh *polymesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<int64> &data)
{
    sim::add_polymesh_attrib_i64( polymesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::PolyMesh *polymesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<fpreal32> &data)
{
    sim::add_polymesh_attrib_f32( polymesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::PolyMesh *polymesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<fpreal64> &data)
{
    sim::add_polymesh_attrib_f64( polymesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::PolyMesh *polymesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<const char *> &strings,
        const std::vector<int64> &indices)
{
    sim::add_polymesh_attrib_str(
            polymesh, where, name, tuple_size, strings.size(),
            strings.data(), indices.size(), indices.data());
}

void add_attrib(
        sim::TetMesh *tetmesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<int8> &data)
{
    sim::add_tetmesh_attrib_i8( tetmesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::TetMesh *tetmesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<int32> &data)
{
    sim::add_tetmesh_attrib_i32( tetmesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::TetMesh *tetmesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<int64> &data)
{
    sim::add_tetmesh_attrib_i64( tetmesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::TetMesh *tetmesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<fpreal32> &data)
{
    sim::add_tetmesh_attrib_f32( tetmesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::TetMesh *tetmesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<fpreal64> &data)
{
    sim::add_tetmesh_attrib_f64( tetmesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::TetMesh *tetmesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<const char *> &strings,
        const std::vector<int64> &indices)
{
    sim::add_tetmesh_attrib_str(
            tetmesh, where, name, tuple_size, strings.size(),
            strings.data(), indices.size(), indices.data());
}

template<typename T>
GA_PrimitiveTypeId mesh_prim_type_id();

template<>
GA_PrimitiveTypeId mesh_prim_type_id<sim::PolyMesh>() { return GA_PRIMPOLY; }

template<>
GA_PrimitiveTypeId mesh_prim_type_id<sim::TetMesh>() { return GA_PRIMTETRAHEDRON; }

template<typename T>
sim::AttribLocation mesh_prim_attrib_location();

template<>
sim::AttribLocation mesh_prim_attrib_location<sim::PolyMesh>() { return sim::AttribLocation::Face; }

template<>
sim::AttribLocation mesh_prim_attrib_location<sim::TetMesh>() { return sim::AttribLocation::Cell; }

template<typename T>
sim::AttribLocation mesh_vertex_attrib_location();

template<>
sim::AttribLocation mesh_vertex_attrib_location<sim::PolyMesh>() { return sim::AttribLocation::FaceVertex; }

template<>
sim::AttribLocation mesh_vertex_attrib_location<sim::TetMesh>() { return sim::AttribLocation::CellVertex; }

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
void fill_prim_attrib(
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
    add_attrib(mesh, sim::AttribLocation::Vertex, name, tuple_size, data);
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
    std::vector<int64> ids(aif->getTableEntries(attrib), -1);
    std::vector<const char *> strings;
    for (auto it = aif->begin(attrib); !it.atEnd(); ++it) {
        ids[it.getHandle()] = strings.size();
        strings.push_back( it.getString() );
    }

    std::vector<int64> indices(tuple_size*num_elem, -1);

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
    std::vector<int64> ids(aif->getTableEntries(attrib), -1);
    std::vector<const char *> strings;
    for (auto it = aif->begin(attrib); !it.atEnd(); ++it) {
        ids[it.getHandle()] = strings.size();
        strings.push_back( it.getString() );
    }

    std::vector<int64> indices(tuple_size*num_elem, -1);

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
    add_attrib(mesh, sim::AttribLocation::Vertex, name, tuple_size, strings, indices);
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
    std::vector<int64> ids(aif->getTableEntries(attrib), -1);
    std::vector<const char *> strings;
    for (auto it = aif->begin(attrib); !it.atEnd(); ++it) {
        ids[it.getHandle()] = strings.size();
        strings.push_back( it.getString() );
    }

    std::vector<int64> indices(tuple_size*num_elem, -1);

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
void transfer_attributes(const GU_Detail* detail, M* mesh, std::size_t num_prims)
{
    std::vector<bool> pt_grp, vtx_grp;
    std::tie(pt_grp, vtx_grp) = mark_points_and_vertices(detail, mesh_prim_type_id<M>());

    // Get polygon data attributes
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
                        fill_prim_attrib<int64>(detail, aif, attrib, tuple_size, num_prims, mesh); break;
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
                        fill_point_attrib<int64>(detail, aif, attrib, tuple_size, num_points, pt_grp, mesh); break;
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
                        fill_vertex_attrib<int64>(detail, aif, attrib, tuple_size, num_vertices, vtx_grp, mesh); break;
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

/** Retrieve attributes from the simulation mesh using the given iterator.
 */
void retrieve_attributes(GU_Detail *detail, GA_Offset startoff, sim::AttribIter *it, GA_AttributeOwner owner) {
    while ( it ) { // it could be null, but it doesn't change
        auto attrib = sim::attrib_iter_next(it);
        if (!attrib) break;
        auto name = sim::attrib_name(attrib);
        auto type = sim::attrib_data_type(attrib);
        if (type == sim::DataType::I8 ) {
            auto arr = sim::attrib_data_i8(attrib);
            auto h = GA_RWHandleC(detail->addTuple(GA_STORE_INT8, owner, name, arr.tuple_size));
            fill_attrib(h, arr, startoff);
            sim::free_attrib_data_i8(arr);
        } else if (type == sim::DataType::I32 ) {
            auto arr = sim::attrib_data_i32(attrib);
            auto h = GA_RWHandleI(detail->addTuple(GA_STORE_INT32, owner, name, arr.tuple_size));
            fill_attrib(h, arr, startoff);
            sim::free_attrib_data_i32(arr);
        } else if (type == sim::DataType::I64 ) {
            auto arr = sim::attrib_data_i64(attrib);
            auto h = GA_RWHandleID(detail->addTuple(GA_STORE_INT64, owner, name, arr.tuple_size));
            fill_attrib(h, arr, startoff);
            sim::free_attrib_data_i64(arr);
        } else if (type == sim::DataType::F32 ) {
            auto arr = sim::attrib_data_f32(attrib);
            auto h = GA_RWHandleF(detail->addTuple(GA_STORE_REAL32, owner, name, arr.tuple_size));
            fill_attrib(h, arr, startoff);
            sim::free_attrib_data_f32(arr);
        } else if (type == sim::DataType::F64 ) {
            auto arr = sim::attrib_data_f64(attrib);
            auto h = GA_RWHandleD(detail->addTuple(GA_STORE_REAL64, owner, name, arr.tuple_size));
            fill_attrib(h, arr, startoff);
            sim::free_attrib_data_f64(arr);
        } else if (type == sim::DataType::Str ) {
            auto arr = sim::attrib_data_str(attrib);
            auto h = GA_RWHandleS(detail->addTuple(GA_STORE_STRING, owner, name, arr.tuple_size));
            fill_attrib(h, arr, startoff);
            sim::free_attrib_data_str(arr);
        }
        sim::free_attribute(attrib);
    }
    sim::free_attrib_iter(it);
}

/** Add the given simulation meshes into the given detail
 */
void add_meshes(GU_Detail* detail, sim::TetMesh *tetmesh, sim::PolyMesh *polymesh) {
    GA_Offset startvtxoff = GA_Offset(0);
    // add tets.
    if (tetmesh) {
        auto sim_points = sim::get_tetmesh_points(tetmesh);
        std::vector<UT_Vector3> points;

        for (std::size_t i = 0; i < sim_points.size; ++i) {
            points.push_back(UT_Vector3(sim_points.array[i]));
        }

        auto sim_indices = sim::get_tetmesh_indices(tetmesh);
        if (sim_indices.size > 0) {
            std::vector<int> indices;
            for (std::size_t i = 0; i < sim_indices.size; ++i) {
                indices.push_back(static_cast<int>(sim_indices.array[i]));
            }

            GA_Offset startptoff = detail->appendPointBlock(points.size());
            for (exint pt_idx = 0; pt_idx < points.size(); ++pt_idx) {
                GA_Offset ptoff = startptoff + pt_idx;
                detail->setPos3(ptoff, points[pt_idx]);
            }

            GA_Offset startprimoff = GEO_PrimTetrahedron::buildBlock(
                    detail, startptoff, detail->getNumPointOffsets(),
                    indices.size()/4, indices.data());


            retrieve_attributes(detail, startptoff, sim::tetmesh_attrib_iter(tetmesh, sim::AttribLocation::Vertex), GA_ATTRIB_POINT);
            retrieve_attributes(detail, startprimoff, sim::tetmesh_attrib_iter(tetmesh, sim::AttribLocation::Cell), GA_ATTRIB_PRIMITIVE);
            retrieve_attributes(detail, startvtxoff, sim::tetmesh_attrib_iter(tetmesh, sim::AttribLocation::CellVertex), GA_ATTRIB_VERTEX);
        }
        sim::free_point_array(sim_points);
        sim::free_index_array(sim_indices);
    }

    startvtxoff = GA_Offset(detail->getNumVertexOffsets());

    // add polygons
    if (polymesh) {
        auto sim_points = sim::get_polymesh_points(polymesh);
        std::vector<UT_Vector3> points;

        for (std::size_t i = 0; i < sim_points.size; ++i) {
            points.push_back(UT_Vector3(sim_points.array[i]));
        }

        GA_Offset startptoff = detail->appendPointBlock(points.size());
        for (exint pt_idx = 0; pt_idx < points.size(); ++pt_idx) {
            GA_Offset ptoff = startptoff + pt_idx;
            detail->setPos3(ptoff, points[pt_idx]);
        }

        auto sim_indices = sim::get_polymesh_indices(polymesh);
        if (sim_indices.size > 0) {
            GEO_PolyCounts polycounts;
            std::vector<int> poly_pt_numbers;
            int prev_n = sim_indices.array[0];
            int num_polys_with_same_shape = 0;
            for (std::size_t i = 0; i < sim_indices.size; ) {
                auto n = sim_indices.array[i++];
                if (n != prev_n) {
                    polycounts.append(n, num_polys_with_same_shape);
                    num_polys_with_same_shape = 0;
                    prev_n = n;
                }
                num_polys_with_same_shape += 1;
                for (std::size_t j = 0; j < n; ++j, ++i) {
                    poly_pt_numbers.push_back(sim_indices.array[i]);
                }
            }
            polycounts.append(prev_n, num_polys_with_same_shape); // append last set

            GA_Offset startprimoff = GEO_PrimPoly::buildBlock(
                    detail, startptoff, detail->getNumPointOffsets(),
                    polycounts, poly_pt_numbers.data());

            retrieve_attributes(detail, startptoff, sim::polymesh_attrib_iter(polymesh, sim::AttribLocation::Vertex), GA_ATTRIB_POINT);
            retrieve_attributes(detail, startprimoff, sim::polymesh_attrib_iter(polymesh, sim::AttribLocation::Face), GA_ATTRIB_PRIMITIVE);
            retrieve_attributes(detail, startvtxoff, sim::polymesh_attrib_iter(polymesh, sim::AttribLocation::FaceVertex), GA_ATTRIB_VERTEX);
        }

        sim::free_point_array(sim_points);
        sim::free_index_array(sim_indices);

    }
}

sim::TetMesh *build_sim_tetmesh(const GU_Detail *detail) {
    // Get tets for the body from the first input
    std::vector<double> tet_vertices;
    std::vector<std::size_t> tet_indices;
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

    // Only creating a mesh if there are tets. Otherwise we may be simulating cloth only.
    if (num_tets > 0) {
        sim::TetMesh *tetmesh = sim::make_tetmesh(tet_vertices.size(), tet_vertices.data(),
                                                  tet_indices.size(), tet_indices.data());
        assert(tetmesh);

        transfer_attributes(detail, tetmesh, num_tets);
        return tetmesh;
    }
    return nullptr;
}

sim::PolyMesh *build_sim_polymesh(const GU_Detail* detail) {
    // Get polygons for the body from the second input
    std::vector<double> poly_vertices;
    std::vector<std::size_t> poly_indices;

    for ( GA_Offset pt_off : detail->getPointRange() )
    {
        UT_Vector3 pt = detail->getPos3(pt_off);
        poly_vertices.push_back( static_cast<double>(pt[0]) );
        poly_vertices.push_back( static_cast<double>(pt[1]) );
        poly_vertices.push_back( static_cast<double>(pt[2]) );
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

    if (num_polys > 0) {
        sim::PolyMesh *polymesh = sim::make_polymesh(poly_vertices.size(), poly_vertices.data(),
                                                     poly_indices.size(), poly_indices.data());
        assert(polymesh);

        transfer_attributes(detail, polymesh, num_polys);
        return polymesh;
    }
    return nullptr;

}

// Entry point to the SOP
void
SOP_SimVerb::cook(const SOP_NodeVerb::CookParms &cookparms) const
{
    const GU_Detail *input0 = cookparms.inputGeo(0);
    sim::TetMesh *tetmesh = nullptr;
    if (input0) {
        tetmesh = build_sim_tetmesh(input0);
    }

    const GU_Detail *input1 = cookparms.inputGeo(1);
    sim::PolyMesh *polymesh = nullptr;
    if (input1) {
        polymesh = build_sim_polymesh(input1);
    }

    sim::sim( tetmesh, polymesh );

    //auto &&sopparms = cookparms.parms<SOP_SimParms>();
    GU_Detail *detail = cookparms.gdh().gdpNC();

    // Add the simulation meshes into the current detail
    add_meshes(detail, tetmesh, polymesh);

    // We must release the memory using the same api that allocated it.
    sim::free_tetmesh(tetmesh);
    sim::free_polymesh(polymesh);
}
