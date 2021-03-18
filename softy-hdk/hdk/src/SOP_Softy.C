#include "SOP_Softy.h"

// Needed for template generation with the ds file.
#include "SOP_Softy.proto.h"

#include <rust/cxx.h>
#include <softy/src/lib.rs.h>

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

#include <vector>
#include <array>
#include <cassert>
#include <string>
#include <sstream>
#include <utility>

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
    softy::init_env_logger();
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
                label "Material Id:  #"
                type label
                default { "" }
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
                label "Young's Modulus"
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
            default { "10" }
            range { 0 50 }
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
                label "Use Fixed for Implicit"
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
                    name "frictionforwarding#"
                    label "Friction Forwarding"
                    type float
                    default { "1.0" }
                    range { 0! 1 }
                }
                parm {
                    name "smoothingweight#"
                    label "Smoothing Weight"
                    type float
                    default { "0.0" }
                    range { 0! 1 }
                }
                parm {
                    name "dynamiccof#"
                    label "Dynamic Coefficient"
                    type float
                    default { "0.2" }
                    range { 0 2 }
                }
                parm {
                    name "frictiontolerance#"
                    label "Tolerance"
                    type log
                    default { "1e-10" }
                    range { 0.0 1.0 }
                }
                parm {
                    name "frictioninneriterations#"
                    label "Inner Iterations"
                    type integer
                    default { "50" }
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
            default { "200" }
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
            default { "50" }
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
                default { "1.0" }
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
    softy::clear_solver_registry();
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
write_solver_data(GU_Detail *detail, softy::StepResult res, int64 solver_id) {
    // Create the registry id attribute on the output detail so we don't lose the solver for the
    // next time step.
    GA_RWHandleID attrib( detail->addIntTuple(GA_ATTRIB_GLOBAL, "softy", 1, GA_Defaults(-1), 0, 0,
                GA_STORE_INT64) );
    attrib.set(GA_Offset(0), solver_id);

    // Add the simulation meshes into the current detail
    if (detail) {
        softy::add_meshes(*detail, std::move(res.meshes));
    }
}

std::pair<softy::SimParams, bool> build_sim_params(const SOP_SoftyParms& sopparms) {
    // Gather simulation parameters
    softy::SimParams sim_params;
    sim_params.time_step = sopparms.getTimeStep();
    sim_params.gravity = sopparms.getGravity();
    sim_params.log_file = sopparms.getLogFile().c_str();

    // Get material properties.
    const auto &sop_materials = sopparms.getMaterials();
    for (const auto & sop_mtl : sop_materials) {
        softy::MaterialProperties mtl_props;
        auto sop_objtype = static_cast<SOP_SoftyEnums::ObjectType>(sop_mtl.objtype);
        switch (sop_objtype) {
            case SOP_SoftyEnums::ObjectType::SOLID:
                mtl_props.object_type = softy::ObjectType::Solid;
                break;
            case SOP_SoftyEnums::ObjectType::SHELL:
                mtl_props.object_type = softy::ObjectType::Shell;
                break;
            case SOP_SoftyEnums::ObjectType::RIGID:
                mtl_props.object_type = softy::ObjectType::Rigid;
                break;
        }

        auto sop_elasticity_model = static_cast<SOP_SoftyEnums::ElasticityModel>(sop_mtl.elasticitymodel);
        switch (sop_elasticity_model) {
            case SOP_SoftyEnums::ElasticityModel::SNH:
                mtl_props.elasticity_model = softy::ElasticityModel::StableNeoHookean;
                break;
            case SOP_SoftyEnums::ElasticityModel::NH:
                mtl_props.elasticity_model = softy::ElasticityModel::NeoHookean;
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
        sim_params.materials.push_back(mtl_props);
    }

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
    rust::Vec<softy::FrictionalContactParams> frictional_contact_vec;
    for (const auto & sop_fc: sop_frictional_contacts) {
        softy::FrictionalContactParams fc_params;
        fc_params.object_material_id = sop_fc.objectmaterialid;
        
        // Parse a string of integers into an std::vector<uint32_t>
        UT_String ut_collider_material_ids_str(sop_fc.collidermaterialids);

        std::stringstream ss(ut_collider_material_ids_str.toStdString());
        std::string token;
        while (std::getline(ss, token, ' ')) {
            if (!token.empty()) {
                try {
                    fc_params.collider_material_ids.push_back(std::stoul(token));
                }
                catch (...) {
                    collider_material_id_parse_error = true;
                }
            }
        }

        switch (static_cast<SOP_SoftyEnums::Kernel>(sop_fc.kernel)) {
            case SOP_SoftyEnums::Kernel::INTERPOLATING:
                fc_params.kernel = softy::Kernel::Interpolating;
                break;
            case SOP_SoftyEnums::Kernel::APPROXIMATE:
                fc_params.kernel = softy::Kernel::Approximate;
                break;
            case SOP_SoftyEnums::Kernel::CUBIC:
                fc_params.kernel = softy::Kernel::Cubic;
                break;
            case SOP_SoftyEnums::Kernel::GLOBAL:
                fc_params.kernel = softy::Kernel::Global;
                break;
        }

        switch (static_cast<SOP_SoftyEnums::ContactType>(sop_fc.contacttype)) {
            case SOP_SoftyEnums::ContactType::LINEARIZED:
                fc_params.contact_type = softy::ContactType::LinearizedPoint;
                break;
            case SOP_SoftyEnums::ContactType::POINT:
                fc_params.contact_type = softy::ContactType::Point;
                break;
        }

        fc_params.radius_multiplier = sop_fc.radiusmult;
        fc_params.smoothness_tolerance = sop_fc.smoothtol;
        fc_params.contact_offset = sop_fc.contactoffset;
        fc_params.use_fixed = sop_fc.usefixed;
        if (sop_fc.friction) {
            fc_params.smoothing_weight = sop_fc.smoothingweight;
            fc_params.friction_forwarding = sop_fc.frictionforwarding;
            fc_params.dynamic_cof = sop_fc.dynamiccof;
            fc_params.friction_inner_iterations = sop_fc.frictioninneriterations;
        } else {
            fc_params.dynamic_cof = 0.0;
            fc_params.friction_inner_iterations = 0;
        }
        fc_params.friction_tolerance = sop_fc.frictiontolerance;
        sim_params.frictional_contacts.push_back(fc_params);
    }
    
    sim_params.print_level = sopparms.getPrintLevel();
    sim_params.derivative_test = sopparms.getDerivativeTest();

    switch (static_cast<SOP_SoftyEnums::MuStrategy>(sopparms.getMuStrategy())) {
        case SOP_SoftyEnums::MuStrategy::MONOTONE:
            sim_params.mu_strategy = softy::MuStrategy::Monotone;
            break;
        case SOP_SoftyEnums::MuStrategy::ADAPTIVE:
            sim_params.mu_strategy = softy::MuStrategy::Adaptive;
            break;
    }

    sim_params.max_gradient_scaling = sopparms.getMaxGradientScaling();

    return std::make_pair(sim_params, collider_material_id_parse_error);
}

// Entry point to the SOP
void
SOP_SoftyVerb::cook(const SOP_NodeVerb::CookParms &cookparms) const
{
    const GU_Detail *input0 = cookparms.inputGeo(0);
    const GU_Detail *input1 = cookparms.inputGeo(1);
    if(!input0 && !input1) {
        // No inputs, nothing to do.
        return;
    }

    softy::SimParams sim_params;
    bool collider_material_id_parse_error;
    std::tie(sim_params, collider_material_id_parse_error) =
      build_sim_params(std::move(cookparms.parms<SOP_SoftyParms>()));

    auto interrupt_checker = std::make_unique<hdkrs::InterruptChecker>("Solving Softy");

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

    auto meshes = softy::new_meshes();

    if (solver_id < 0) {
        // If there is no previously allocated solver we can use, we need to extract the geometry
        // from the detail. Otherwise, the geometry stored in the solver itself will be used.

        if (input0) {
            meshes->set_tetmesh(*input0);
        }
        if (input1) {
            meshes->set_polymesh(*input1);
        }
    }

    softy::SolverResult solver_res = softy::get_solver(solver_id, std::move(meshes), sim_params);

    if (solver_res.id < 0) {
        assert(solver_res.cook_result.tag == hdkrs::CookResultTag::ERROR);
        std::stringstream ss;
        ss << "Failed to create or retrieve a solver. ";
        ss << solver_res.cook_result.message;
        cookparms.sopAddError(UT_ERROR_OUTSTREAM, ss.str().c_str());
        std::cerr << ss.str() << std::endl;
        return;
    }

    // Check that we either have a new solver or we found a valid old one.
    UT_ASSERT(solver_id < 0 || solver_id == solver_res.id);

    auto mesh_points = softy::new_mesh_points();

    if (solver_id >= 0) {
        if (input0)
            mesh_points->set_tetmesh_points(*input0);
        if (input1)
            mesh_points->set_polymesh_points(*input1);
    }

    softy::StepResult res = softy::step(std::move(solver_res.solver), std::move(mesh_points), std::move(interrupt_checker));

    switch (res.cook_result.tag) {
        case hdkrs::CookResultTag::SUCCESS:
            cookparms.sopAddMessage(UT_ERROR_OUTSTREAM, res.cook_result.message.c_str()); break;
        case hdkrs::CookResultTag::WARNING:
            cookparms.sopAddWarning(UT_ERROR_OUTSTREAM, res.cook_result.message.c_str()); break;
        case hdkrs::CookResultTag::ERROR:
            cookparms.sopAddError(UT_ERROR_OUTSTREAM, res.cook_result.message.c_str());
            std::cerr << res.cook_result.message << std::endl;
            break;
    }
    if (collider_material_id_parse_error) {
        cookparms.sopAddWarning(UT_ERROR_OUTSTREAM, "Failed to parse some of the frictional contact collider material ids");
    }

    GU_Detail *detail = cookparms.gdh().gdpNC();

    write_solver_data(detail, std::move(res), solver_res.id);
}
