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
#include <sim_api.h>

#include <vector>
#include <array>

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

// Entry point to the SOP
void
SOP_SimVerb::cook(const SOP_NodeVerb::CookParms &cookparms) const
{
    auto &&sopparms = cookparms.parms<SOP_SimParms>();
    GU_Detail *detail = cookparms.gdh().gdpNC();

    // Get tets for the body from the first input
    std::vector<double> tet_vertices;
    std::vector<std::size_t> tet_indices;
    sim::TetMesh * tetmesh = nullptr;
    const GU_Detail *input0 = cookparms.inputGeo(0);
    if (input0) {
        for ( GA_Offset pt_off : input0->getPointRange() )
        {
            UT_Vector3 pt = input0->getPos3(pt_off);
            tet_vertices.push_back( static_cast<double>(pt[0]) );
            tet_vertices.push_back( static_cast<double>(pt[1]) );
            tet_vertices.push_back( static_cast<double>(pt[2]) );
        }


        for ( GA_Offset prim_off : input0->getPrimitiveRange() )
        {
            const GEO_Primitive *prim = input0->getGEOPrimitive(prim_off);
            if (prim->getTypeId() == GA_PRIMTETRAHEDRON) {
                const GEO_PrimTetrahedron *tet = static_cast<const GEO_PrimTetrahedron*>(prim);
                tet_indices.push_back(input0->pointIndex(tet->fastVertexOffset(0)));
                tet_indices.push_back(input0->pointIndex(tet->fastVertexOffset(1)));
                tet_indices.push_back(input0->pointIndex(tet->fastVertexOffset(2)));
                tet_indices.push_back(input0->pointIndex(tet->fastVertexOffset(3)));
            }
        }

        tetmesh = sim::make_tetmesh(tet_vertices.size(), tet_vertices.data(),
                                    tet_indices.size(), tet_indices.data());
    }

    // Get polygons for the body from the second input
    std::vector<double> poly_vertices;
    std::vector<std::size_t> poly_indices;
    sim::PolyMesh * polymesh = nullptr;

    const GU_Detail *input1 = cookparms.inputGeo(1);

    if (input1) {
        using sim::AttribLocation;
        std::size_t num_polys = 0;
        for ( GA_Offset prim_off : input1->getPrimitiveRange() )
        {
            for ( GA_Offset pt_off : input1->getPointRange() )
            {
                UT_Vector3 pt = input1->getPos3(pt_off);
                poly_vertices.push_back( static_cast<double>(pt[0]) );
                poly_vertices.push_back( static_cast<double>(pt[1]) );
                poly_vertices.push_back( static_cast<double>(pt[2]) );
            }

            const GEO_Primitive *prim = input1->getGEOPrimitive(prim_off);
            if (prim->getTypeId() == GA_PRIMPOLY) {
                num_polys += 1;
                const GEO_PrimPoly *poly = static_cast<const GEO_PrimPoly*>(prim);
                std::size_t num_verts = poly->getVertexCount();
                poly_indices.push_back(num_verts);
                for ( std::size_t i = 0; i < num_verts; ++i ) {
                    poly_indices.push_back(poly->getVertexIndex(i));
                }
            }
        }

        polymesh = sim::make_polymesh(poly_vertices.size(), poly_vertices.data(),
                                      poly_indices.size(), poly_indices.data());

        // Get polygon data attributes
        for (auto it = input1->getAttributeDict(GA_ATTRIB_PRIMITIVE).begin(GA_SCOPE_PUBLIC); !it.atEnd(); ++it)
        {
            GA_Attribute *attrib = it.attrib();
            auto name = attrib->getName().c_str();
            std::size_t tuple_size = attrib->getTupleSize();
            {
                const GA_AIFTuple *aif = attrib->getAIFTuple(); // array of data
                if ( aif )
                {
                    // Try with different types
                    auto storage = aif->getStorage(attrib);
                    if (storage == GA_STORE_BOOL || storage == GA_STORE_INT32 || storage == GA_STORE_INT64) {
                        std::vector<int64> data(tuple_size*num_polys);
                        int i = 0;
                        for ( GA_Offset prim_off : input1->getPrimitiveRange() )
                        {
                            const GEO_Primitive *prim = input1->getGEOPrimitive(prim_off);
                            if (prim->getTypeId() == GA_PRIMPOLY)
                            {
                                for ( int k = 0, k_end = tuple_size; k < k_end; ++k )
                                    aif->get(attrib, prim_off, data[tuple_size*i + k], k);
                                i += 1;
                            }
                        }

                        auto n = data.size();

                        if ( storage == GA_STORE_BOOL ) {
                            std::cerr << "found bool poly attrib: " << name << std::endl;
                            sim::add_polymesh_attrib_i64(
                                    polymesh, AttribLocation::Face, name, tuple_size,n, data.data() );
                        } else if ( storage == GA_STORE_INT32 ) {
                            std::cerr << "found int32 poly attrib: " << name << std::endl;
                            sim::add_polymesh_attrib_i64(
                                    polymesh, AttribLocation::Face, name, tuple_size, n, data.data() );
                        } else if ( storage == GA_STORE_INT64 ) {
                            std::cerr << "found int64 poly attrib: " << name << std::endl;
                            sim::add_polymesh_attrib_i64(
                                    polymesh, AttribLocation::Face, name, tuple_size, n, data.data() );
                        }
                    } else if (storage == GA_STORE_REAL32 || storage == GA_STORE_REAL64) {
                        std::vector<fpreal64> data(tuple_size*num_polys);
                        int i = 0;
                        for ( GA_Offset prim_off : input1->getPrimitiveRange() )
                        {
                            const GEO_Primitive *prim = input1->getGEOPrimitive(prim_off);
                            if (prim->getTypeId() == GA_PRIMPOLY)
                            {
                                for ( int k = 0, k_end = tuple_size; k < k_end; ++k )
                                    aif->get(attrib, prim_off, data[tuple_size*i + k], k);
                                i += 1;
                            }
                        }

                        auto n = data.size();

                        if ( storage == GA_STORE_REAL32) {
                            std::cerr << "found f32 poly attrib: " << name << std::endl;
                            sim::add_polymesh_attrib_f64(
                                    polymesh, AttribLocation::Face, name, tuple_size, n, data.data() );
                        } else if ( storage == GA_STORE_REAL64 ) {
                            std::cerr << "found f64 poly attrib: " << name << std::endl;
                            sim::add_polymesh_attrib_f64(
                                    polymesh, AttribLocation::Face, name, tuple_size, n, data.data() );
                        }
                    }
                }
            }


            {
                auto aif = attrib->getAIFSharedStringTuple(); // array of strings
                if ( aif ) {
                    // Try with different types
                    aif->compactStorage(attrib);
                    std::vector<int64> ids(aif->getTableEntries(attrib), -1);
                    std::vector<const char *> strings;
                    for (auto it = aif->begin(attrib); !it.atEnd(); ++it) {
                        ids[it.getHandle()] = strings.size();
                        strings.push_back( it.getString() );
                    }

                    std::vector<int64> indices(tuple_size*num_polys);

                    int i = 0;
                    for ( GA_Offset prim_off : input1->getPrimitiveRange() )
                    {
                        const GEO_Primitive *prim = input1->getGEOPrimitive(prim_off);
                        if (prim->getTypeId() == GA_PRIMPOLY)
                        {
                            for ( int k = 0, k_end = tuple_size; k < k_end; ++k ) {
                                GA_StringIndexType handle = aif->getHandle(attrib, prim_off, k);
                                indices[tuple_size*i + k] = ids[handle];
                            }
                            i += 1;
                        }
                    }

                    sim::add_polymesh_attrib_str(
                            polymesh, AttribLocation::Face, name, tuple_size, strings.size(),
                            strings.data(), indices.size(), indices.data());
                }
            }

            // don't know how to handle these yet.
            //aif = attrib->getAIFNumericArray(); // variable sized array
            //aif = attrib->getAIFSharedStringArray(); // variable sized array of strings
        }

        //for (auto it = gdp->getAttributeDict(GA_ATTRIB_POINT).begin(GA_SCOPE_PUBLIC); !it.atEnd(); it.advance())
        //{
        //    GA_Attribute *attrib = it.attrib();

        //}
    }


//    sim::sim( tet_vertices.data(), tet_indices.data(),
//              tri_vertices.data(), tri_indices.data() );
}
