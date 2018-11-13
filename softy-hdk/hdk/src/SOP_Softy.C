#include "SOP_Softy.h"

// Needed for template generation with the ds file.
#include "SOP_Softy.proto.h"

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
#include <softy-hdk.h>

#include <vector>
#include <array>
#include <cassert>
#include <sstream>

const UT_StringHolder SOP_Softy::theSOPTypeName("hdk_softy"_sh);

// Register sop operator
void
newSopOperator(OP_OperatorTable *table)
{
    table->addOperator(new OP_Operator(
                SOP_Softy::theSOPTypeName,   // Internal name
                "Softy",                     // UI name
                SOP_Softy::myConstructor,    // How to build the SOP
                SOP_Softy::buildTemplates(), // My parameters
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

        groupradio {
            name "stiffness_type"
            label "Shear Bulk"
            grouptag { "group_type" "radio" }

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
        }

        groupradio {
            name "stiffness_type_1"
            label "Young Poisson"
            grouptag { "group_type" "radio" }

            parm {
                name "youngmodulus"
                cppname "YoungModulus"
                label "Young Modulus"
                type float
                default { "1000" }
                range { 0 10000 }
            }

            parm {
                name "poissonratio"
                cppname "PoissonRatio"
                label "Poisson Ratio"
                type float
                default { "0.45" }
                range { 0 0.5 }
            }
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
        parm {
            name "maxiterations"
            cppname "MaxIterations"
            label "Max Iterations"
            type integer
            default { "800" }
            range { 0 1000 }
        }
    }

    parm {
        name "volumeconstraint"
        cppname "VolumeConstraint"
        label "Enable Volume Constraint"
        type toggle
        default { "off" }
    }

    groupsimple {
        name "smoothcontact"
        label "Smooth Contact"
        grouptag { "group_type" "simple" }
        disablewhen "{ hasinput(1) == 0 }"

        parm {
            name "contactradius"
            cppname "ContactRadius"
            label "Contact Radius"
            type float
            default { "1" }
            range { 0.0 10.0 }
        }

        parm {
            name "smoothtol"
            cppname "SmoothTol"
            label "Smoothness Tolerance"
            type log
            default { "1e-5" }
            range { 0.0 1.0 }
        }
    }

    parm {
        name "clearcache"
        label "Clear Cache"
        type button
        default { "0" }
        range { 0 1 }
    }
}
)THEDSFILE";

int SOP_Softy::clearSolverCache(void *data, int index, float t, const PRM_Template *) {
    hdkrs::clear_solver_registry();
    return 0;
}


PRM_Template *
SOP_Softy::buildTemplates()
{
    static PRM_TemplateBuilder templ("SOP_Softy.C"_sh, theDsFile);
    templ.setCallback("clearcache", SOP_Softy::clearSolverCache);
    return templ.templates();
}

class SOP_SoftyVerb : public SOP_NodeVerb
{
    public:
        SOP_SoftyVerb() {}
        virtual ~SOP_SoftyVerb() {}

        virtual SOP_NodeParms *allocParms() const { return new SOP_SoftyParms(); }
        virtual UT_StringHolder name() const { return SOP_Softy::theSOPTypeName; }

        virtual CookMode cookMode(const SOP_NodeParms *parms) const { return COOK_GENERATOR; }

        virtual void cook(const CookParms &cookparms) const;

        static const SOP_NodeVerb::Register<SOP_SoftyVerb> theVerb;
};

const SOP_NodeVerb::Register<SOP_SoftyVerb> SOP_SoftyVerb::theVerb;

const SOP_NodeVerb *
SOP_Softy::cookVerb() const
{
    return SOP_SoftyVerb::theVerb.get();
}

void
write_solver_data(GU_Detail *detail, hdkrs::StepResult res, int64 solver_id) {
    using namespace hdkrs;
    using namespace hdkrs::mesh;

    // Create the registry id attribute on the output detail so we don't lose the solver for the
    // next time step.
    GA_RWHandleID attrib( detail->addIntTuple(GA_ATTRIB_GLOBAL, "softy", 1, GA_Defaults(-1), 0, 0,
                GA_STORE_INT64) );
    attrib.set(GA_Offset(0), solver_id);

    // Add the simulation meshes into the current detail
    OwnedPtr<TetMesh> tetmesh = res.tetmesh;
    //OwnedPtr<PolyMesh> polymesh = res.polymesh;

    add_tetmesh(detail, std::move(tetmesh));
    //add_polymesh(detail, std::move(polymesh));
}

// Entry point to the SOP
void
SOP_SoftyVerb::cook(const SOP_NodeVerb::CookParms &cookparms) const
{
    using namespace hdkrs;
    using namespace hdkrs::mesh;

    auto &&sopparms = cookparms.parms<SOP_SoftyParms>();

    // Gather simulation parameters
    SimParams sim_params;
    sim_params.time_step = sopparms.getTimeStep();

    if (sopparms.getStiffness_type() == 0) { 
        sim_params.material.bulk_modulus = sopparms.getVolumeStiffness()*1e6;
        sim_params.material.shear_modulus = sopparms.getShapeStiffness()*1e6;
    } else {
        // K = E / 3(1-2v)
        // G = E / 2(1+v)
        auto nu = sopparms.getPoissonRatio();
        auto E = sopparms.getYoungModulus()*1e6;
        sim_params.material.bulk_modulus = E / (3*(1.0 - 2*nu));
        sim_params.material.shear_modulus = E / (2*(1+nu));
    }

    sim_params.material.damping = sopparms.getDamping();
    sim_params.material.density = sopparms.getDensity();
    sim_params.gravity = sopparms.getGravity();
    sim_params.tolerance = sopparms.getTolerance();
    sim_params.max_iterations = sopparms.getMaxIterations();
    sim_params.volume_constraint = sopparms.getVolumeConstraint();
    sim_params.contact_radius = sopparms.getContactRadius();
    sim_params.smoothness_tolerance = sopparms.getSmoothTol();

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

    OwnedPtr<TetMesh> tetmesh = nullptr;
    OwnedPtr<PolyMesh> polymesh = nullptr;

    if (solver_id < 0) {
        // If there is no previously allocated solver we can use, we need to extract the geometry
        // from the detail. Otherwise, the geometry stored in the solver itself will be used.

        tetmesh = build_tetmesh(input0);

        const GU_Detail *input1 = cookparms.inputGeo(1);
        if (input1) {
            polymesh = build_polymesh(input1);
        }
    }

    SolverResult solver_res = hdkrs::get_solver(solver_id, tetmesh.release(), polymesh.release(), sim_params);

    if (solver_res.id < 0) {
        assert(solver_res.cook_result.tag == CookResultTag::Error);
        std::stringstream ss;
        ss << "Failed to create or retrieve a solver. ";
        ss << solver_res.cook_result.message;
        cookparms.sopAddError(UT_ERROR_OUTSTREAM, ss.str().c_str());
        return;
    }

    UT_ASSERT(solver_id < 0 || solver_id == solver_res.id);

    OwnedPtr<PointCloud> tetmesh_ptcloud = nullptr;
    OwnedPtr<PointCloud> polymesh_ptcloud = nullptr;

    if (solver_id >= 0) {
        tetmesh_ptcloud = build_pointcloud(input0);
        const GU_Detail *input1 = cookparms.inputGeo(1);
        if (input1) {
            polymesh_ptcloud = build_pointcloud(input1);
        }
    }

    StepResult res = hdkrs::step(
            solver_res.solver,
            tetmesh_ptcloud.release(),
            polymesh_ptcloud.release(),
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

    write_solver_data(detail, res, solver_res.id);

    free_result(res.cook_result);
}
