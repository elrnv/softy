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
                1,                              // Min # of sources
                4,                              // Max # of sources
                nullptr,                        // Local variables
                OP_FLAG_GENERATOR));            // Flag it as generator
}


static const char *theDsFile = R"THEDSFILE(
{
    name sim

    parm {
        name "rustlib"
        cppname "RustLibPath"
        label "Rust Library Path"
        type string
        default { "./" }
  }
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
    std::cerr << "found bool poly attrib: " << std::string(name) << " on " << where << std::endl;
    sim::add_polymesh_attrib_i8( polymesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::PolyMesh *polymesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<int32> &data)
{
    std::cerr << "found i32 poly attrib: " << std::string(name) << " on " << where << std::endl;
    sim::add_polymesh_attrib_i32( polymesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::PolyMesh *polymesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<int64> &data)
{
    std::cerr << "found i64 poly attrib: " << std::string(name) << " on " << where << std::endl;
    sim::add_polymesh_attrib_i64( polymesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::PolyMesh *polymesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<fpreal32> &data)
{
    std::cerr << "found f32 poly attrib: " << std::string(name) << " on " << where << std::endl;
    sim::add_polymesh_attrib_f32( polymesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::PolyMesh *polymesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<fpreal64> &data)
{
    std::cerr << "found f64 poly attrib: " << std::string(name) << " on " << where << std::endl;
    sim::add_polymesh_attrib_f64( polymesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::PolyMesh *polymesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<const char *> &strings,
        const std::vector<uint64> &indices)
{
    std::cerr << "found string poly attrib: " << std::string(name) << " on " << where << std::endl;
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
    std::cerr << "found bool tet attrib: " << std::string(name) << " on " << where << std::endl;
    sim::add_tetmesh_attrib_i8( tetmesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::TetMesh *tetmesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<int32> &data)
{
    std::cerr << "found i32 tet attrib: " << std::string(name) << " on " << where << std::endl;
    sim::add_tetmesh_attrib_i32( tetmesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::TetMesh *tetmesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<int64> &data)
{
    std::cerr << "found i64 tet attrib: " << std::string(name) << " on " << where << std::endl;
    sim::add_tetmesh_attrib_i64( tetmesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::TetMesh *tetmesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<fpreal32> &data)
{
    std::cerr << "found f32 tet attrib: " << std::string(name) << " on " << where << std::endl;
    sim::add_tetmesh_attrib_f32( tetmesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::TetMesh *tetmesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<fpreal64> &data)
{
    std::cerr << "found f64 tet attrib: " << std::string(name) << " on " << where << std::endl;
    sim::add_tetmesh_attrib_f64( tetmesh, where, name, tuple_size, data.size(), data.data() );
}

void add_attrib(
        sim::TetMesh *tetmesh,
        sim::AttribLocation where,
        const char *name,
        std::size_t tuple_size,
        const std::vector<const char *> &strings,
        const std::vector<uint64> &indices)
{
    std::cerr << "found string tet attrib: " << std::string(name) << " on " << where << std::endl;
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
        if (prim->getTypeId() == mesh_prim_type_id<M>())
        {
            for ( int k = 0, k_end = tuple_size; k < k_end; ++k ) {
                S val; 
                aif->get(attrib, prim_off, val, k);
                data[tuple_size*i + k] = val;
            }
            i += 1;
        }
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
        M *mesh)
{
    std::vector<T> data(tuple_size*num_elem);
    int i = 0;
    for ( GA_Offset pt_off : detail->getPointRange() )
    {
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

    std::vector<uint64> indices(tuple_size*num_elem, 0);

    int i = 0;
    for ( GA_Offset prim_off : detail->getPrimitiveRange() )
    {
        const GEO_Primitive *prim = detail->getGEOPrimitive(prim_off);
        if (prim->getTypeId() == mesh_prim_type_id<M>())
        {
            for ( int k = 0, k_end = tuple_size; k < k_end; ++k ) {
                GA_StringIndexType handle = aif->getHandle(attrib, prim_off, k);
                indices[tuple_size*i + k] = ids[handle];
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
        M *mesh)
{
    // Try with different types
    std::vector<int64> ids(aif->getTableEntries(attrib), -1);
    std::vector<const char *> strings;
    for (auto it = aif->begin(attrib); !it.atEnd(); ++it) {
        ids[it.getHandle()] = strings.size();
        strings.push_back( it.getString() );
    }

    std::vector<uint64> indices(tuple_size*num_elem, 0);

    int i = 0;
    for ( GA_Offset pt_off : detail->getPointRange() )
    {
        for ( int k = 0, k_end = tuple_size; k < k_end; ++k ) {
            GA_StringIndexType handle = aif->getHandle(attrib, pt_off, k);
            indices[tuple_size*i + k] = ids[handle];
        }
        i += 1;
    }

    auto name = attrib->getName().c_str();
    add_attrib(mesh, sim::AttribLocation::Vertex, name, tuple_size, strings, indices);
}

template<typename M>
void transfer_attributes(const GU_Detail* detail, M* mesh, std::size_t num_elem)
{
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
                        fill_prim_attrib<int8, M, int32>(detail, aif, attrib, tuple_size, num_elem, mesh); break;
                    case GA_STORE_INT32:
                        fill_prim_attrib<int32>(detail, aif, attrib, tuple_size, num_elem, mesh); break;
                    case GA_STORE_INT64:
                        fill_prim_attrib<int64>(detail, aif, attrib, tuple_size, num_elem, mesh); break;
                    case GA_STORE_REAL32:
                        fill_prim_attrib<fpreal32>(detail, aif, attrib, tuple_size, num_elem, mesh); break;
                    case GA_STORE_REAL64:
                        fill_prim_attrib<fpreal64>(detail, aif, attrib, tuple_size, num_elem, mesh); break;
                    default: break; // do nothing
                }
            }
        }

        {
            auto aif = attrib->getAIFSharedStringTuple(); // array of strings
            if ( aif ) {
                aif->compactStorage(attrib);
                fill_prim_str_attrib(detail, aif, attrib, tuple_size, num_elem, mesh);
            }
        }

        // don't know how to handle these yet.
        //aif = attrib->getAIFNumericArray(); // variable sized array
        //aif = attrib->getAIFSharedStringArray(); // variable sized array of strings
    }

    std::size_t num_points = detail->getNumPoints();
    for (auto it = detail->getAttributeDict(GA_ATTRIB_POINT).begin(GA_SCOPE_PUBLIC); !it.atEnd(); ++it)
    {
        GA_Attribute *attrib = it.attrib();
        std::size_t tuple_size = attrib->getTupleSize();
        {
            auto aif = attrib->getAIFTuple(); // array of data
            if ( aif )
            {
                switch (aif->getStorage(attrib)) {
                    case GA_STORE_BOOL:
                        fill_point_attrib<int8, M, int32>(detail, aif, attrib, tuple_size, num_points, mesh); break;
                    case GA_STORE_INT32:
                        fill_point_attrib<int32>(detail, aif, attrib, tuple_size, num_points, mesh); break;
                    case GA_STORE_INT64:
                        fill_point_attrib<int64>(detail, aif, attrib, tuple_size, num_points, mesh); break;
                    case GA_STORE_REAL32:
                        fill_point_attrib<fpreal32>(detail, aif, attrib, tuple_size, num_points, mesh); break;
                    case GA_STORE_REAL64:
                        fill_point_attrib<fpreal64>(detail, aif, attrib, tuple_size, num_points, mesh); break;
                    default: break; // do nothing
                }
            }
        }


        {
            auto aif = attrib->getAIFSharedStringTuple(); // array of strings
            if ( aif ) {
                aif->compactStorage(attrib);
                fill_point_str_attrib(detail, aif, attrib, tuple_size, num_points, mesh);
            }
        }

        // don't know how to handle these yet.
        //aif = attrib->getAIFNumericArray(); // variable sized array
        //aif = attrib->getAIFSharedStringArray(); // variable sized array of strings
    }
}

// Entry point to the SOP
void
SOP_SimVerb::cook(const SOP_NodeVerb::CookParms &cookparms) const
{
    // Get tets for the body from the first input
    std::vector<double> tet_vertices;
    std::vector<std::size_t> tet_indices;
    sim::TetMesh * tetmesh = nullptr;
    const GU_Detail *input0 = cookparms.inputGeo(0);
    assert(input0); // required by node constraints
    {
        for ( GA_Offset pt_off : input0->getPointRange() )
        {
            UT_Vector3 pt = input0->getPos3(pt_off);
            tet_vertices.push_back( static_cast<double>(pt[0]) );
            tet_vertices.push_back( static_cast<double>(pt[1]) );
            tet_vertices.push_back( static_cast<double>(pt[2]) );
        }

        std::size_t num_tets = 0;
        for ( GA_Offset prim_off : input0->getPrimitiveRange() )
        {
            const GEO_Primitive *prim = input0->getGEOPrimitive(prim_off);
            if (prim->getTypeId() == GA_PRIMTETRAHEDRON) {
                num_tets += 1;
                const GEO_PrimTetrahedron *tet = static_cast<const GEO_PrimTetrahedron*>(prim);
                tet_indices.push_back(input0->pointIndex(input0->vertexPoint(tet->fastVertexOffset(0))));
                tet_indices.push_back(input0->pointIndex(input0->vertexPoint(tet->fastVertexOffset(1))));
                tet_indices.push_back(input0->pointIndex(input0->vertexPoint(tet->fastVertexOffset(2))));
                tet_indices.push_back(input0->pointIndex(input0->vertexPoint(tet->fastVertexOffset(3))));
            }
        }

        tetmesh = sim::make_tetmesh(tet_vertices.size(), tet_vertices.data(),
                                    tet_indices.size(), tet_indices.data());
        assert(tetmesh);

        transfer_attributes(input0, tetmesh, num_tets);
    }

    // Get polygons for the body from the second input
    std::vector<double> poly_vertices;
    std::vector<std::size_t> poly_indices;
    sim::PolyMesh * polymesh = nullptr;

    const GU_Detail *input1 = cookparms.inputGeo(1);

    if (input1) {
        for ( GA_Offset pt_off : input1->getPointRange() )
        {
            UT_Vector3 pt = input1->getPos3(pt_off);
            poly_vertices.push_back( static_cast<double>(pt[0]) );
            poly_vertices.push_back( static_cast<double>(pt[1]) );
            poly_vertices.push_back( static_cast<double>(pt[2]) );
        }

        std::size_t num_polys = 0;
        for ( GA_Offset prim_off : input1->getPrimitiveRange() )
        {
            const GEO_Primitive *prim = input1->getGEOPrimitive(prim_off);
            if (prim->getTypeId() == GA_PRIMPOLY) {
                num_polys += 1;
                const GEO_PrimPoly *poly = static_cast<const GEO_PrimPoly*>(prim);
                std::size_t num_verts = poly->getVertexCount();
                poly_indices.push_back(num_verts);
                for ( std::size_t i = 0; i < num_verts; ++i ) {
                    GA_Index idx = input1->pointIndex(input1->vertexPoint(poly->getVertexOffset(i)));
                    assert(GAisValid(idx));
                    poly_indices.push_back(static_cast<std::size_t>(idx));
                }
            }
        }

        polymesh = sim::make_polymesh(poly_vertices.size(), poly_vertices.data(),
                                      poly_indices.size(), poly_indices.data());
        assert(polymesh);

        transfer_attributes(input1, polymesh, num_polys);
    }

    sim::sim( tetmesh, polymesh );

    auto &&sopparms = cookparms.parms<SOP_SimParms>();
    GU_Detail *detail = cookparms.gdh().gdpNC();

    // add tets.
    if (tetmesh) {
        auto sim_points = sim::get_tetmesh_points(tetmesh);
        std::vector<UT_Vector3> points;

        for (std::size_t i = 0; i < sim_points.size; ++i) {
            points.push_back(UT_Vector3(sim_points.array[i]));
        }

        auto sim_indices = sim::get_tetmesh_indices(tetmesh);
        std::vector<int> indices;
        for (std::size_t i = 0; i < sim_indices.size; ++i) {
            indices.push_back(static_cast<int>(sim_indices.array[i]));
        }

        GA_Offset startptoff = detail->appendPointBlock(points.size());
        for (exint pt_idx = 0; pt_idx < points.size(); ++pt_idx) {
            GA_Offset ptoff = startptoff + pt_idx;
            detail->setPos3(ptoff, points[pt_idx]);
        }

        GEO_PrimTetrahedron::buildBlock(
                detail, startptoff, detail->getNumPointOffsets(),
                indices.size()/4, indices.data());

        sim::free_point_array(sim_points);
        sim::free_index_array(sim_indices);
    }

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

            GEO_PrimPoly::buildBlock(
                    detail, startptoff, detail->getNumPointOffsets(),
                    polycounts, poly_pt_numbers.data());
        }

        sim::free_point_array(sim_points);
        sim::free_index_array(sim_indices);
    }

}
