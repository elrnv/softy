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
#include <string>
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
                0,                              // Min # of sources
                2,                              // Max # of sources
                nullptr,                        // Local variables
                OP_FLAG_GENERATOR));            // Flag it as generator
    el_softy_init_env_logger();
}

static const char *theDsFile = R"THEDSFILE(
{
    name softy

    parm {
        name "clearcache"
        label "Clear Cache"
        type button
        default { "0" }
        range { 0 1 }
    }

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

    parm {
        name "logfile"
        cppname "LogFile"
        label "Log File"
        type file
        default { "" }
    }

    group {
        name "material"
        label "Material"

        multiparm {
            name  "materials"
            label "Number of Materials"
            default 0
            parmtag { "multistartoffset" "1" }

            parm {
                name "materialid#"
                cppname "MaterialId"
                label "Material Id"
                type label
                default { "#" }
            }

            parm {
                name "objtype#"
                cppname "ObjectType"
                label "Object Type"
                type ordinal
                default { "0" }
                menu {
                    "solid" "Solid"
                    "shell" "Shell"
                    "rigid" "Rigid"
                }
            }

            parm {
                name "elasticitymodel#"
                cppname "ElasticityModel"
                label "Elasticity Model"
                type ordinal
                default { "0" }
                hidewhen "{ objtype# != solid }"
                menu {
                    "snh" "Stable Neo-Hookean"
                    "nh" "Neo-Hookean"
                }
            }

            parm {
                name "bendingstiffness#"
                cppname "BendingStiffness"
                label "Bending Stiffness"
                type float
                default { "0" }
                range { 0 100 }
                hidewhen "{ objtype# != shell }"
            }

            parm {
                name "density#"
                label "Density"
                type float
                default { "1000" }
                range { 0 2000 }
            }
            parm {
                name "damping#"
                label "Damping"
                type float
                default { "0.0" }
                range { 0 1000 }
                hidewhen "{ objtype# == rigid }"
            }

            parm {
                name "stiffnesstype#"
                cppname "StiffnessType"
                label "Stiffness Type"
                type ordinal
                default { "1" }
                hidewhen "{ objtype# == rigid }"
                menu {
                    "shearbulk" "Shear and Bulk Moduli"
                    "youngpoisson" "Young's Modulus and Poisson's Ratio"
                }
            }

            parm {
                name "shapestiffness#"
                cppname "ShapeStiffness"
                label "Shape Stiffness"
                type float
                default { "10" }
                range { 0 100 }
                hidewhen "{ stiffnesstype# == youngpoisson } { objtype# == rigid }"
            }

            parm {
                name "volumestiffness#"
                cppname "VolumeStiffness"
                label "Volume Stiffness"
                type float
                default { "1750" }
                range { 0 10000 }
                hidewhen "{ stiffnesstype# == youngpoisson } { objtype# == rigid }"
            }

            parm {
                name "youngmodulus#"
                cppname "YoungModulus"
                label "Young Modulus"
                type float
                default { "3.24" }
                range { 0 1000 }
                hidewhen "{ stiffnesstype# == shearbulk } { objtype# == rigid }"
            }

            parm {
                name "poissonratio#"
                cppname "PoissonRatio"
                label "Poisson Ratio"
                type float
                default { "0.49" }
                range { 0 0.5 }
                hidewhen "{ stiffnesstype# == shearbulk } { objtype# == rigid }"
            }
        }
    }

    group {
        name "constraints"
        label "Constraints"

        parm {
            name "volumeconstraint"
            cppname "VolumeConstraint"
            label "Enable Volume Constraint"
            type toggle
            default { "off" }
        }

        parm {
            name "frictioniterations"
            cppname "FrictionIterations"
            label "Friction Iterations"
            type integer
            default { "1" }
            range { 0 10 }
        }

        multiparm {
            name    "frictionalcontacts"
            cppname "FrictionalContacts"
            label    "Frictional Contacts"
            default 0

            parm {
                name "objectmaterialid#"
                label "Implicit Material Id"
                type integer
                default { "0" }
                range { 0! 1 }
            }

            parm {
                name "collidermaterialids#"
                label "Point Material Ids"
                type string
                default { "" }
            }

            parm {
                name "kernel#"
                label "Kernel"
                type ordinal
                default { "1" }
                menu {
                    "interpolating" "Local Interpolating"
                    "approximate" "Local approximately interpolating"
                    "cubic" "Local cubic"
                    "global" "Global inverse squared distance"
                }
            }

            parm {
                name "contacttype#"
                cppname "ContactType"
                label "Contact Type"
                type ordinal
                default { "0" }
                menu {
                    "linearized" "Linearized Point"
                    "point" "Point"
                }
            }

            parm {
                name "radiusmult#"
                cppname "RadiusMultiplier"
                label "Radius Multiplier"
                type float
                default { "1" }
                range { 0.0 10.0 }
            }

            parm {
                name "smoothtol#"
                cppname "SmoothTol"
                label "Smoothness Tolerance"
                type log
                default { "1e-5" }
                range { 0.0 1.0 }
                hidewhen "{ kernel# == interpolating } { kernel# == cubic }"
            }

            parm {
                name "contactoffset#"
                cppname "ContactOffset"
                label "Contact Offset"
                type log
                default { "0.0" }
                range { 0.0 1.0 }
            }

            parm {
                name "usefixed#"
                label "Use Fixed"
                type toggle
                default { "off" }
            }

            parm {
                name "friction#"
                label "Friction"
                type toggle
                default { "off" }
            }

            groupsimple {
                name "frictionparams#"
                label "Friction Parameters"
                grouptag { "group_type" "simple" }
                hidewhen "{ friction# == 0 }"

                parm {
                    name "smoothingweight#"
                    label "Smoothing Weight"
                    type float
                    default { "0.5" }
                    range { 0! 1! }
                }
                parm {
                    name "dynamiccof#"
                    label "Dynamic Coefficient"
                    type float
                    default { "0.5" }
                    range { 0 2 }
                }
                parm {
                    name "frictiontolerance#"
                    label "Tolerance"
                    type log
                    default { "1e-5" }
                    range { 0.0 1.0 }
                }
                parm {
                    name "frictioninneriterations#"
                    label "Inner Iterations"
                    type integer
                    default { "40" }
                    range { 0 10 }
                }
            }

        }
    }

    group {
        name "optimization"
        label "Optimization"

        parm {
            name "clearvelocity"
            cppname "ClearVelocity"
            label "Clear Velocity"
            type toggle
            default { "off" }
        }
        parm {
            name "innertolerance"
            cppname "InnerTolerance"
            label "Inner Error Tolerance"
            type log
            default { "1e-9" }
            range { 0.0 1.0 }
        }
        parm {
            name "maxinneriterations"
            cppname "MaxInnerIterations"
            label "Max Inner Iterations"
            type integer
            default { "500" }
            range { 0 1000 }
        }
        parm {
            name "outertolerance"
            cppname "OuterTolerance"
            label "Outer Error Tolerance"
            type log
            default { "1e-5" }
            range { 0.0 1.0 }
        }
        parm {
            name "maxouteriterations"
            cppname "MaxOuterIterations"
            label "Max Outer Iterations"
            type integer
            default { "10" }
            range { 0 1000 }
        }

        groupcollapsible {
            name    "ipoptoptions"
            label   "Ipopt Options"
            grouptag { "group_type" "collapsible" }

            parm {
                name "mustrategy"
                cppname "MuStrategy"
                label "Mu Strategy"
                type ordinal
                default { "1" }
                menu {
                    "monotone" "Monotone"
                    "adaptive" "Adaptive"
                }
            }

            parm {
                name "maxgradientscaling"
                cppname "MaxGradientScaling"
                label "Max Gradient Scaling"
                type log
                default { "100.0" }
                range { 0.0 100.0 }
            }

            parm {
                name "printlevel"
                cppname "PrintLevel"
                label "Print Level"
                type integer
                default { "0" }
                range { 0! 12! }
            }

            parm {
                name "derivativetest"
                cppname "DerivativeTest"
                label "Derivative Test"
                type integer
                default { "0" }
                range { 0! 2! }
            }
        }

    }

}
)THEDSFILE";



int SOP_Softy::clearSolverCache(void *data, int index, float t, const PRM_Template *) {
    el_softy_clear_solver_registry();
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
write_solver_data(GU_Detail *detail, EL_SoftyStepResult res, int64 solver_id) {
    using namespace hdkrs;
    using namespace hdkrs::mesh;

    // Create the registry id attribute on the output detail so we don't lose the solver for the
    // next time step.
    GA_RWHandleID attrib( detail->addIntTuple(GA_ATTRIB_GLOBAL, "softy", 1, GA_Defaults(-1), 0, 0,
                GA_STORE_INT64) );
    attrib.set(GA_Offset(0), solver_id);

    // Add the simulation meshes into the current detail
    OwnedPtr<HR_TetMesh> tetmesh = res.tetmesh;
    OwnedPtr<HR_PolyMesh> polymesh = res.polymesh;

    add_tetmesh(detail, std::move(tetmesh));
    add_polymesh(detail, std::move(polymesh));
}

// Entry point to the SOP
void
SOP_SoftyVerb::cook(const SOP_NodeVerb::CookParms &cookparms) const
{
    using namespace hdkrs;
    using namespace hdkrs::mesh;

    const GU_Detail *input0 = cookparms.inputGeo(0);
    const GU_Detail *input1 = cookparms.inputGeo(1);
    if(!input0 && !input1) {
        // No inputs, nothing to do.
        return;
    }

    auto &&sopparms = cookparms.parms<SOP_SoftyParms>();

    // Gather simulation parameters
    EL_SoftySimParams sim_params;
    sim_params.time_step = sopparms.getTimeStep();
    sim_params.gravity = sopparms.getGravity();
    sim_params.log_file = sopparms.getLogFile().c_str();

    // Get material properties.
    const auto &sop_materials = sopparms.getMaterials();
    std::vector<EL_SoftyMaterialProperties> materials_vec;
    for (const auto & sop_mtl : sop_materials) {
        EL_SoftyMaterialProperties mtl_props;
        auto sop_objtype = static_cast<SOP_SoftyEnums::ObjectType>(sop_mtl.objtype);
        switch (sop_objtype) {
            case SOP_SoftyEnums::ObjectType::SOLID:
                mtl_props.object_type = EL_SoftyObjectType::Solid;
                break;
            case SOP_SoftyEnums::ObjectType::SHELL:
                mtl_props.object_type = EL_SoftyObjectType::Shell;
                break;
            case SOP_SoftyEnums::ObjectType::RIGID:
                mtl_props.object_type = EL_SoftyObjectType::Rigid;
                break;
        }

        auto sop_elasticity_model = static_cast<SOP_SoftyEnums::ElasticityModel>(sop_mtl.elasticitymodel);
        switch (sop_elasticity_model) {
            case SOP_SoftyEnums::ElasticityModel::SNH:
                mtl_props.elasticity_model = EL_SoftyElasticityModel::StableNeoHookean;
                break;
            case SOP_SoftyEnums::ElasticityModel::NH:
                mtl_props.elasticity_model = EL_SoftyElasticityModel::NeoHookean;
                break;
        }

        auto sop_stiffnesstype = static_cast<SOP_SoftyEnums::StiffnessType>(sop_mtl.stiffnesstype);
        if (sop_stiffnesstype == SOP_SoftyEnums::StiffnessType::SHEARBULK) { 
            mtl_props.bulk_modulus = sop_mtl.volumestiffness*1.0e3;
            mtl_props.shear_modulus = sop_mtl.shapestiffness*1.0e3;
        } else {
            // K = E / 3(1-2v)
            // G = E / 2(1+v)
            auto nu = sop_mtl.poissonratio;
            auto E = sop_mtl.youngmodulus*1.0e3;
            mtl_props.bulk_modulus = E / (3.0*(1.0 - 2.0*nu));
            mtl_props.shear_modulus = E / (2.0*(1.0 + nu));
        }

        mtl_props.bending_stiffness = sop_mtl.bendingstiffness;
        mtl_props.damping = sop_mtl.damping;
        mtl_props.density = sop_mtl.density;
        materials_vec.push_back(mtl_props);
    }

    sim_params.materials
        = EL_SoftyMaterials{ materials_vec.data(), materials_vec.size() };
    
    sim_params.clear_velocity = sopparms.getClearVelocity();
    sim_params.tolerance = sopparms.getInnerTolerance();
    sim_params.max_iterations = sopparms.getMaxInnerIterations();
    sim_params.outer_tolerance = sopparms.getOuterTolerance();
    sim_params.max_outer_iterations = sopparms.getMaxOuterIterations();

    sim_params.volume_constraint = sopparms.getVolumeConstraint();
    sim_params.friction_iterations = sopparms.getFrictionIterations();

    bool collider_material_id_parse_error = false;

    // Get frictional contact params.
    const auto &sop_frictional_contacts = sopparms.getFrictionalContacts();
    std::vector<EL_SoftyFrictionalContactParams> frictional_contact_vec;
    std::vector<std::vector<uint32_t>> fc_collider_material_ids;
    for (const auto & sop_fc: sop_frictional_contacts) {
        EL_SoftyFrictionalContactParams fc_params;
        fc_params.object_material_id = sop_fc.objectmaterialid;
        
        // Parse a string of integers into an std::vector<uint32_t>
        UT_String ut_collider_material_ids_str(sop_fc.collidermaterialids);

        fc_collider_material_ids.push_back(std::vector<uint32_t>());
        std::stringstream ss(ut_collider_material_ids_str.toStdString());
        std::string token;
        while (std::getline(ss, token, ' ')) {
            if (!token.empty()) {
                try {
                    fc_collider_material_ids.back().push_back(std::stoul(token));
                }
                catch (...) {
                    collider_material_id_parse_error = true;
                }
            }
        }

        fc_params.collider_material_ids = EL_SoftyColliderMaterialIds {
            fc_collider_material_ids.back().data(),
            fc_collider_material_ids.back().size(),
        };

        switch (static_cast<SOP_SoftyEnums::Kernel>(sop_fc.kernel)) {
            case SOP_SoftyEnums::Kernel::INTERPOLATING:
                fc_params.kernel = EL_SoftyKernel::Interpolating;
                break;
            case SOP_SoftyEnums::Kernel::APPROXIMATE:
                fc_params.kernel = EL_SoftyKernel::Approximate;
                break;
            case SOP_SoftyEnums::Kernel::CUBIC:
                fc_params.kernel = EL_SoftyKernel::Cubic;
                break;
            case SOP_SoftyEnums::Kernel::GLOBAL:
                fc_params.kernel = EL_SoftyKernel::Global;
                break;
        }

        switch (static_cast<SOP_SoftyEnums::ContactType>(sop_fc.contacttype)) {
            case SOP_SoftyEnums::ContactType::LINEARIZED:
                fc_params.contact_type = EL_SoftyContactType::LinearizedPoint;
                break;
            case SOP_SoftyEnums::ContactType::POINT:
                fc_params.contact_type = EL_SoftyContactType::Point;
                break;
        }

        fc_params.radius_multiplier = sop_fc.radiusmult;
        fc_params.smoothness_tolerance = sop_fc.smoothtol;
        fc_params.contact_offset = sop_fc.contactoffset;
        fc_params.use_fixed = sop_fc.usefixed;
        if (sop_fc.friction) {
            fc_params.smoothing_weight = sop_fc.smoothingweight;
            fc_params.dynamic_cof = sop_fc.dynamiccof;
            fc_params.friction_inner_iterations = sop_fc.frictioninneriterations;
        } else {
            fc_params.dynamic_cof = 0.0;
            fc_params.friction_inner_iterations = 0;
        }
        fc_params.friction_tolerance = sop_fc.frictiontolerance;
        frictional_contact_vec.push_back(fc_params);
    }

    sim_params.frictional_contacts = EL_SoftyFrictionalContacts{
        frictional_contact_vec.data(),
        frictional_contact_vec.size()
    };
    
    sim_params.print_level = sopparms.getPrintLevel();
    sim_params.derivative_test = sopparms.getDerivativeTest();

    switch (static_cast<SOP_SoftyEnums::MuStrategy>(sopparms.getMuStrategy())) {
        case SOP_SoftyEnums::MuStrategy::MONOTONE:
            sim_params.mu_strategy = EL_SoftyMuStrategy::Monotone;
            break;
        case SOP_SoftyEnums::MuStrategy::ADAPTIVE:
            sim_params.mu_strategy = EL_SoftyMuStrategy::Adaptive;
            break;
    }

    sim_params.max_gradient_scaling = sopparms.getMaxGradientScaling();

    interrupt::InterruptChecker interrupt_checker("Solving Softy");

    int64 solver_id = -1;

    // Select the main input (if input0 is missing we are solving for cloth only).
    const GU_Detail *main_input = input0;
    if (!main_input) {
        main_input = input1;
    }

    // Retrieve a unique ID for the solver being used by softy.
    // This will allow us to reuse existing memory in the solver.
    GA_ROHandleID attrib(main_input->findIntTuple(GA_ATTRIB_GLOBAL, "softy", 1));
    if (!attrib.isInvalid()) {
        solver_id = attrib.get(GA_Offset(0));
    }

    OwnedPtr<HR_TetMesh> tetmesh = nullptr;
    OwnedPtr<HR_PolyMesh> polymesh = nullptr;

    if (solver_id < 0) {
        // If there is no previously allocated solver we can use, we need to extract the geometry
        // from the detail. Otherwise, the geometry stored in the solver itself will be used.

        if (input0) {
            tetmesh = build_tetmesh(input0);
        }

        if (input1) {
            polymesh = build_polymesh(input1);
        }
    }

    EL_SoftySolverResult solver_res = el_softy_get_solver(solver_id, tetmesh.release(), polymesh.release(), sim_params);

    if (solver_res.id < 0) {
        assert(solver_res.cook_result.tag == HRCookResultTag::HR_ERROR);
        std::stringstream ss;
        ss << "Failed to create or retrieve a solver. ";
        ss << solver_res.cook_result.message;
        cookparms.sopAddError(UT_ERROR_OUTSTREAM, ss.str().c_str());
        std::cerr << ss.str() << std::endl;
        return;
    }

    // Check that we either have a new solver or we found a valid old one.
    UT_ASSERT(solver_id < 0 || solver_id == solver_res.id);

    OwnedPtr<HR_PointCloud> tetmesh_ptcloud = nullptr;
    OwnedPtr<HR_PointCloud> polymesh_ptcloud = nullptr;

    if (solver_id >= 0) {
        // If it's an old one, update its meshes.
        if (input0) {
            tetmesh_ptcloud = build_pointcloud(input0);
        }
        if (input1) {
            polymesh_ptcloud = build_pointcloud(input1);
        }
    }

    EL_SoftyStepResult res = el_softy_step(
            solver_res.solver,
            tetmesh_ptcloud.release(),
            polymesh_ptcloud.release(),
            &interrupt_checker,
            interrupt::check_interrupt);

    switch (res.cook_result.tag) {
        case HRCookResultTag::HR_SUCCESS:
            cookparms.sopAddMessage(UT_ERROR_OUTSTREAM, res.cook_result.message); break;
        case HRCookResultTag::HR_WARNING:
            cookparms.sopAddWarning(UT_ERROR_OUTSTREAM, res.cook_result.message); break;
        case HRCookResultTag::HR_ERROR:
            cookparms.sopAddError(UT_ERROR_OUTSTREAM, res.cook_result.message);
            std::cerr << res.cook_result.message << std::endl;
            break;
    }
    if (collider_material_id_parse_error) {
        cookparms.sopAddWarning(UT_ERROR_OUTSTREAM, "Failed to parse some of the frictional contact collider material ids");
    }

    GU_Detail *detail = cookparms.gdh().gdpNC();

    write_solver_data(detail, res, solver_res.id);

    hr_free_result(res.cook_result);
}
