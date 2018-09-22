#include "SOP_Sim.h"

// Needed for template generation with the ds file.
#include "SOP_Sim.proto.h"

#include <hdkrs/mesh.h>
#include <hdkrs/interrupt.h>

// Required for proper loading.
#include <UT/UT_DSOVersion.h>

#include <UT/UT_Interrupt.h>
#include <UT/UT_StringHolder.h>
#include <PRM/PRM_Include.h>
#include <PRM/PRM_TemplateBuilder.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <GEO/GEO_PrimTetrahedron.h>
#include <GEO/GEO_PrimPoly.h>
#include <GEO/GEO_PolyCounts.h>
#include <sim-hdk.h>

#include <vector>
#include <array>
#include <cassert>
#include <mutex>

/*
using Solver = hdkrs::FemEngine;

namespace hdkrs {
    template<>
    inline OwnedPtr<Solver>::~OwnedPtr() {
        free_solver(_ptr);
    }
}

std::mutex ourRegistryLock;

// Solver key counter. Having this is not strictly necessary, but it helps identify an unoccupied
// key in the registry quickly. We essentially increment this key indefinitely to generate new keys.
// We use an unsigned int here for mod arithmetic.
std::size_t ourGlobalSolverKey = 0; 

// Global solver registry. We use a map instead of a vector to avoid dealing with fragmentation.
std::unordered_map<std::size_t, hdkrs::OwnedPtr<Solver>> ourSolvers;
*/

const UT_StringHolder SOP_Sim::theSOPTypeName("hdk_softy"_sh);

// Register sop operator
void
newSopOperator(OP_OperatorTable *table)
{
    table->addOperator(new OP_Operator(
                SOP_Sim::theSOPTypeName,   // Internal name
                "Softy",                     // UI name
                SOP_Sim::myConstructor,    // How to build the SOP
                SOP_Sim::buildTemplates(), // My parameters
                1,                              // Min # of sources
                2,                              // Max # of sources
                nullptr,                        // Local variables
                OP_FLAG_GENERATOR));            // Flag it as generator
}

static const char *theDsFile = R"THEDSFILE(
{
    name softy

    parm {
        name "timestep"
        cppname "TimeStep"
        label "Time Step"
        type float
        default { "1.0/$FPS" }
        range { 0 10 }
    }

    parm {
        name "gravity"
        label "Gravity"
        type float
        default { "9.81" }
        range { 0 20 }
    }

    groupsimple {
        name "material"
        label "Material"
        grouptag { "group_type" "simple" }

        parm {
            name "shapestiffness"
            cppname "ShapeStiffness"
            label "Shape Stiffness"
            type float
            default { "10" }
            range { 0 100 }
        }
        parm {
            name "volumestiffness"
            cppname "VolumeStiffness"
            label "Volume Stiffness"
            type float
            default { "1750" }
            range { 0 10000 }
        }
        parm {
            name "density"
            label "Density"
            type float
            default { "1000" }
            range { 0 2000 }
        }
        parm {
            name "damping"
            label "Damping"
            type float
            default { "1.0" }
            range { 0 1000 }
        }
    }

    groupsimple {
        name "optimization"
        label "Optimization"
        grouptag { "group_type" "simple" }

        parm {
            name "tolerance"
            label "Error Tolerance"
            type float
            default { "1e-9" }
            range { 0.0 1.0 }
        }
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

#if 0

// Retrieve a solver from the registry by looking at the corresponding registry id attribute in the
// detail.
// PRE: We assume that ourRegistryLock has already been acquired by this thread.
std::pair<Solver *, std::size_t>
SOP_SimVerb::getExistingSolver(const GU_Detail *detail) const
{
    GA_ROHandleID attrib(detail->findIntTuple(GA_ATTRIB_GLOBAL, "softy", 1));
    if (attrib.isInvalid())
        return std::make_pair(nullptr, -1);

    std::size_t solver_id = static_cast<std::size_t>(attrib.get(GA_Offset(0)));

    auto solver_it = ourSolvers.find(solver_id);
    if ( solver_it == ourSolvers.end() )
        return std::make_pair(nullptr, -1);
    else
        return std::make_pair(*solver_it, solver_id);
}

// Get an existing solver from the registry or create a new one if the registry solver is invalid.
Solver *
SOP_SimVerb::getSolver(
        const SOP_NodeVerb::CookParms &cookparms,
        hdkrs::TetMesh *tetmesh,
        hdkrs::PolyMesh* polymesh,
        hdkrs::SimParams params) const
{
    using namespace hdkrs;
    const GU_Detail *detail = cookparms.inputGeo(0);
    if (!detail) {
        cookparms.sopAddError(UT_ERROR_OUTSTREAM, "First input is not found");
        return nullptr;
    }

    Solver* solver = nullptr;
    std::size_t solver_id = -1;
    {
        std::lock_guard<std::mutex> guard(ourRegistryLock);

        std::tie(solver, solver_id) = getExistingSolver(detail);

        if (!solver) {
            InterruptChecker interrupt_checker("Solving Softy");
            SolverResult res = new_solver(tetmesh, polymesh, sim_params, &interrupt_checker, interrupt::check_interrupt);
            if (!res.solver) {
                UT_ASSERT(res.result.tag == CookResultTag::Error);
                cookparms.sopAddError(UT_ERROR_OUTSTREAM, res.result.message);
                return nullptr;
            }

            solver = res.solver;
            std::size_t count = 0;
            do {
                ourGlobalSolverKey += 1;
                count += 1;
            } while (ourSolvers.find(ourGlobalSolverKey) != ourSolvers.end() && count != 0);
            solver_id = ourGlobalSolverKey;
            if (ourSolvers.find(solver_id) != ourSolvers.end()) {
                cookparms.sopAddError(UT_ERROR_OUTSTREAM, "No more storage available for new solvers");
                return nullptr;
            }
            ourSolvers.emplace(solver_id, solver);
        } else {
            // We found a solver. Need to check that it's a valid solver with the right parameters.
            // Otherwise we can be using a stale or malicious id.
        }
    }

    // Create the registry id attribute on the output detail so we don't lose the solver for the
    // next time step.
    GU_Detail *detail = cookparms.gdh().gdpNC();
    GA_RWHandleI attrib(detail->addIntTuple(GA_ATTRIB_GLOBAL, "softy", 1, GA_Defaults(0), 0, 0, GA_STORE_INT64));
    UT_ASSERT(solver);
    attrib.set(GA_Offset(0), solver_id);

    return solver;
}
#endif


void
write_solver_data(GU_Detail *detail, hdkrs::SolveResult res) {
    using namespace hdkrs;
    using namespace hdkrs::mesh;

    // Create the registry id attribute on the output detail so we don't lose the solver for the
    // next time step.
    GA_RWHandleID attrib( detail->addIntTuple(GA_ATTRIB_GLOBAL, "softy", 1, GA_Defaults(-1), 0, 0,
                GA_STORE_INT64) );
    attrib.set(GA_Offset(0), res.solver_id);

    // Add the simulation meshes into the current detail
    OwnedPtr<TetMesh> tetmesh = res.tetmesh;
    OwnedPtr<PolyMesh> polymesh = res.polymesh;

    add_polymesh(detail, std::move(polymesh));
    add_tetmesh(detail, std::move(tetmesh));
}

// Entry point to the SOP
void
SOP_SimVerb::cook(const SOP_NodeVerb::CookParms &cookparms) const
{
    using namespace hdkrs;
    using namespace hdkrs::mesh;

    auto &&sopparms = cookparms.parms<SOP_SimParms>();

    OwnedPtr<TetMesh> tetmesh = nullptr;
    OwnedPtr<PolyMesh> polymesh = nullptr;

    // Gather simulation parameters
    SimParams sim_params;
    sim_params.time_step = sopparms.getTimeStep();
    sim_params.material.damping = sopparms.getDamping();
    sim_params.material.density = sopparms.getDensity();
    sim_params.material.bulk_modulus = sopparms.getVolumeStiffness()*1e6;
    sim_params.material.shear_modulus = sopparms.getShapeStiffness()*1e6;
    sim_params.gravity = sopparms.getGravity();
    sim_params.tolerance = sopparms.getTolerance();

    interrupt::InterruptChecker interrupt_checker("Solving Softy");

    const GU_Detail *input0 = cookparms.inputGeo(0);
    UT_ASSERT(input0);

    int64 solver_id = -1;

    // Retrieve a unique ID for the solver being used by softy.
    // This will allow us to reuse existing memory in the solver.
    GA_ROHandleID attrib(input0->findIntTuple(GA_ATTRIB_GLOBAL, "softy", 1));
    if (!attrib.isInvalid()) {
        solver_id = attrib.get(GA_Offset(0));
    }

    if (solver_id < 0) {
        // If there is no previously allocated solver we can use, we need to extract the geometry
        // from the detail. Otherwise, the geometry stored in the solver itself will be used.

        tetmesh = build_tetmesh(input0);

        const GU_Detail *input1 = cookparms.inputGeo(1);
        if (input1) {
            polymesh = build_polymesh(input1);
        }
    }

    SolveResult res = hdkrs::cook(
            solver_id,
            tetmesh.release(),
            polymesh.release(),
            sim_params,
            &interrupt_checker,
            interrupt::check_interrupt);

    switch (res.cook_result.tag) {
        case CookResultTag::Success:
            cookparms.sopAddMessage(UT_ERROR_OUTSTREAM, res.cook_result.message); break;
        case CookResultTag::Warning:
            cookparms.sopAddWarning(UT_ERROR_OUTSTREAM, res.cook_result.message); break;
        case CookResultTag::Error:
            cookparms.sopAddError(UT_ERROR_OUTSTREAM, res.cook_result.message); break;
    }

    GU_Detail *detail = cookparms.gdh().gdpNC();

    write_solver_data(detail, res);

    free_result(res.cook_result);
}
