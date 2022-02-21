#include "SOP_SoftyScene.h"

// Needed for template generation with the ds file.
#include "SOP_SoftyScene.proto.h"

#include <rust/cxx.h>
#include <scene/src/lib.rs.h>

// Required for proper loading.
#include <UT/UT_DSOVersion.h>

#include <UT/UT_Interrupt.h>
#include <UT/UT_StringHolder.h>
#include <PRM/PRM_Include.h>
#include <PRM/PRM_TemplateBuilder.h>
#include <OP/OP_Operator.h>
#include <OP/OP_AutoLockInputs.h>
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

const UT_StringHolder SOP_SoftyScene::theSOPTypeName("hdk_softy_scene"_sh);

// Register sop operator
void newSopOperator(OP_OperatorTable *table)
{
    table->addOperator(new OP_Operator(
        SOP_SoftyScene::theSOPTypeName,   // Internal name
        "Softy Scene",                     // UI name
        SOP_SoftyScene::myConstructor,    // How to build the SOP
        SOP_SoftyScene::buildTemplates(), // My parameters
        0,                           // Min # of sources
        1,                           // Max # of sources
        nullptr,                     // Local variables
        OP_FLAG_GENERATOR));         // Flag it as generator
    softy::init_env_logger();
}

static const char *theDsFile = R"THEDSFILE(
{
    name softy_scene

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
        name "scenefile"
        cppname "SceneFile"
        label "Scene File"
        type file
        default { "" }
    }

    parm {
        name "framerange"
        cppname "FrameRange"
        label "Frame Range"
        type intvector2
        size 2
        default { "$FSTART" "$FEND" }
    }

    parm {
        name "solvertype"
        cppname "SolverType"
        label "Solver Type"
        type ordinal
        default { "0" }
        menu {
            "newton" "Newton"
            "newtonbt" "Newton with Backtracking"
            "newtonassistbt" "Newton with Assisted Backtracking"
            "trustregion" "Trust Region"
        }
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
            }

            parm {
                name "stiffnesstype#"
                cppname "StiffnessType"
                label "Stiffness Type"
                type ordinal
                default { "1" }
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
                hidewhen "{ stiffnesstype# == youngpoisson }"
            }

            parm {
                name "volumestiffness#"
                cppname "VolumeStiffness"
                label "Volume Stiffness"
                type float
                default { "1750" }
                range { 0 10000 }
                hidewhen "{ stiffnesstype# == youngpoisson }"
            }

            parm {
                name "youngmodulus#"
                cppname "YoungModulus"
                label "Young's Modulus"
                type float
                default { "3.24" }
                range { 0 1000 }
                hidewhen "{ stiffnesstype# == shearbulk }"
            }

            parm {
                name "poissonratio#"
                cppname "PoissonRatio"
                label "Poisson Ratio"
                type float
                default { "0.49" }
                range { 0 0.5 }
                hidewhen "{ stiffnesstype# == shearbulk }"
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
                name "dynamiccof#"
                label "Dynamic Friction"
                type float
                default { "0.0" }
                range { 0 2 }
            }
        }
    }

    group {
        name "solver"
        label "Solver"

        parm {
            name "timeintegration"
            cppname "TimeIntegration"
            label "Time Integration"
            type ordinal
            default { "0" }
            menu {
                "be" "Backward Euler (BE)"
                "tr" "Trapezoidal Rule (TR)"
                "bdf2" "BDF2"
                "trbdf2" "TR-BDF2"
            }
        }
        parm {
            name "velocityclearfrequency"
            cppname "VelocityClearFrequency"
            label "Velocity Clear Frequency"
            type float
            default { "0.0" }
            range { 0 100000 }
        }

        parm {
            name "innertolerance"
            cppname "InnerTolerance"
            label "Inner Error Tolerance"
            type log
            default { "0.0" }
            range { 0.0 1.0 }
        }
        parm {
            name "maxinneriterations"
            cppname "MaxInnerIterations"
            label "Max Inner Iterations"
            type integer
            default { "0" }
            range { 0 1000 }
        }
        parm {
            name    "residualcriterion"
            cppname "ResidualCriterion"
            label ""
            nolabel
            type    toggle
            joinnext
            default { "on" }
        }
        parm {
            name    "residualtolerance"
            cppname "ResidualTolerance"
            label   "Residual Tolerance"
            type log
            default { "1e-5" }
            range { 0.0 1.0 }
            disablewhen "{ residualcriterion == 0 }"
        }
        parm {
            name    "accelerationcriterion"
            cppname "AccelerationCriterion"
            label ""
            nolabel
            type    toggle
            joinnext
            default { "off" }
        }
        parm {
            name    "accelerationtolerance"
            cppname "AccelerationTolerance"
            label   "Acceleration Tolerance"
            type log
            default { "1e-5" }
            range { 0.0 1.0 }
            disablewhen "{ accelerationcriterion == 0 }"
        }
        parm {
            name    "velocitycriterion"
            cppname "VelocityCriterion"
            label ""
            nolabel
            type    toggle
            joinnext
            default { "on" }
        }
        parm {
            name "velocitytolerance"
            cppname "VelocityTolerance"
            label   "Velocity Tolerance"
            type log
            default { "1e-5" }
            range { 0.0 1.0 }
            disablewhen "{ velocitycriterion == 0 }"
        }
        parm {
            name "maxouteriterations"
            cppname "MaxOuterIterations"
            label "Max Outer Iterations"
            type integer
            default { "50" }
            range { 0 1000 }
        }
        parm {
            name "derivativetest"
            cppname "DerivativeTest"
            label "Derivative Test"
            type integer
            default { "0" }
            range { 0! 2! }
        }
        parm {
            name "frictiontolerance"
            cppname "FrictionTolerance"
            label "Friction Tolerance"
            type log
            default { "1e-5" }
            range { 0.0 1.0 }
        }
        parm {
            name "contacttolerance"
            cppname "ContactTolerance"
            label "Contact Tolerance"
            type log
            default { "1e-5" }
            range { 0.0 1.0 }
        }
        parm {
            name "contactiterations"
            cppname "ContactIterations"
            label "Contact Iterations"
            type integer
            default { "5" }
            range { 0 50 }
        }
        parm {
            name "frictionprofile"
            cppname "FrictionProfile"
            label "Friction Profile"
            type ordinal
            default { "0" }
            menu {
                "stabilized" "Stabilized"
                "quadratic" "Quadratic"
            }
        }
    }

}
)THEDSFILE";

PRM_Template *
SOP_SoftyScene::buildTemplates()
{
    static PRM_TemplateBuilder templ("SOP_SoftyScene.C"_sh, theDsFile);
    return templ.templates();
}

class SOP_SoftySceneVerb : public SOP_NodeVerb
{
public:
    SOP_SoftySceneVerb() {}
    virtual ~SOP_SoftySceneVerb() {}

    virtual SOP_NodeParms *allocParms() const { return new SOP_SoftySceneParms(); }
    virtual UT_StringHolder name() const { return SOP_SoftyScene::theSOPTypeName; }

    virtual CookMode cookMode(const SOP_NodeParms *parms) const { return COOK_DUPLICATE; }

    virtual void cook(const CookParms &cookparms) const;

    static const SOP_NodeVerb::Register<SOP_SoftySceneVerb> theVerb;
};

const SOP_NodeVerb::Register<SOP_SoftySceneVerb> SOP_SoftySceneVerb::theVerb;

const SOP_NodeVerb *
SOP_SoftyScene::cookVerb() const
{
    return SOP_SoftySceneVerb::theVerb.get();
}

//void write_solver_data(GU_Detail *detail, softy::StepResult res, int64 solver_id)
//{
//    // Create the registry id attribute on the output detail so we don't lose the solver for the
//    // next time step.
//    GA_RWHandleID attrib(detail->addIntTuple(GA_ATTRIB_GLOBAL, "softy_scene", 1, GA_Defaults(-1), 0, 0,
//                                             GA_STORE_INT64));
//    attrib.set(GA_Offset(0), solver_id);
//
//    // Add the simulation meshes into the current detail
//    if (detail)
//    {
//        softy::add_mesh(*detail, std::move(res.mesh));
//    }
//}

std::pair<softy::SimParams, bool> build_sim_params(const SOP_SoftySceneParms &sopparms)
{
    // Gather simulation parameters
    softy::SimParams sim_params;
    sim_params.time_step = sopparms.getTimeStep();
    sim_params.gravity = sopparms.getGravity();

    // Get material properties.
    const auto &sop_materials = sopparms.getMaterials();
    for (const auto &sop_mtl : sop_materials)
    {
        softy::MaterialProperties mtl_props;
        auto sop_objtype = static_cast<SOP_SoftySceneEnums::ObjectType>(sop_mtl.objtype);
        switch (sop_objtype)
        {
        case SOP_SoftySceneEnums::ObjectType::SOLID:
            mtl_props.object_type = softy::ObjectType::Solid;
            break;
        case SOP_SoftySceneEnums::ObjectType::SHELL:
            mtl_props.object_type = softy::ObjectType::Shell;
            break;
        }

        auto sop_elasticity_model = static_cast<SOP_SoftySceneEnums::ElasticityModel>(sop_mtl.elasticitymodel);
        switch (sop_elasticity_model)
        {
        case SOP_SoftySceneEnums::ElasticityModel::SNH:
            mtl_props.elasticity_model = softy::ElasticityModel::StableNeoHookean;
            break;
        case SOP_SoftySceneEnums::ElasticityModel::NH:
            mtl_props.elasticity_model = softy::ElasticityModel::NeoHookean;
            break;
        }

        auto sop_stiffnesstype = static_cast<SOP_SoftySceneEnums::StiffnessType>(sop_mtl.stiffnesstype);
        if (sop_stiffnesstype == SOP_SoftySceneEnums::StiffnessType::SHEARBULK)
        {
            mtl_props.bulk_modulus = sop_mtl.volumestiffness * 1.0e3;
            mtl_props.shear_modulus = sop_mtl.shapestiffness * 1.0e3;
        }
        else
        {
            // K = E / 3(1-2v)
            // G = E / 2(1+v)
            auto nu = sop_mtl.poissonratio;
            auto E = sop_mtl.youngmodulus * 1.0e3;
            mtl_props.bulk_modulus = E / (3.0 * (1.0 - 2.0 * nu));
            mtl_props.shear_modulus = E / (2.0 * (1.0 + nu));
        }

        mtl_props.bending_stiffness = sop_mtl.bendingstiffness;
        mtl_props.damping = sop_mtl.damping;
        mtl_props.density = sop_mtl.density;
        sim_params.materials.push_back(mtl_props);
    }

    switch (static_cast<SOP_SoftySceneEnums::SolverType>(sopparms.getSolverType()))
    {
    case SOP_SoftySceneEnums::SolverType::NEWTON:
        sim_params.solver_type = softy::SolverType::Newton;
        break;
    case SOP_SoftySceneEnums::SolverType::NEWTONBT:
        sim_params.solver_type = softy::SolverType::NewtonBacktracking;
        break;
    case SOP_SoftySceneEnums::SolverType::NEWTONASSISTBT:
        sim_params.solver_type = softy::SolverType::NewtonAssistedBacktracking;
        break;
    case SOP_SoftySceneEnums::SolverType::TRUSTREGION:
        sim_params.solver_type = softy::SolverType::TrustRegion;
        break;
    }
    switch (static_cast<SOP_SoftySceneEnums::TimeIntegration>(sopparms.getTimeIntegration()))
    {
        case SOP_SoftySceneEnums::TimeIntegration::BE:
            sim_params.time_integration = softy::TimeIntegration::BE;
        break;
        case SOP_SoftySceneEnums::TimeIntegration::TR:
            sim_params.time_integration = softy::TimeIntegration::TR;
        break;
        case SOP_SoftySceneEnums::TimeIntegration::BDF2:
            sim_params.time_integration = softy::TimeIntegration::BDF2;
        break;
        case SOP_SoftySceneEnums::TimeIntegration::TRBDF2:
            sim_params.time_integration = softy::TimeIntegration::TRBDF2;
        break;
    }
    sim_params.velocity_clear_frequency = sopparms.getVelocityClearFrequency();
    sim_params.tolerance = sopparms.getInnerTolerance();
    sim_params.max_iterations = sopparms.getMaxInnerIterations();
    sim_params.residual_criterion = sopparms.getResidualCriterion();
    sim_params.residual_tolerance = sopparms.getResidualTolerance();
    sim_params.acceleration_criterion = sopparms.getAccelerationCriterion();
    sim_params.acceleration_tolerance = sopparms.getAccelerationTolerance();
    sim_params.velocity_criterion = sopparms.getVelocityCriterion();
    sim_params.velocity_tolerance = sopparms.getVelocityTolerance();
    sim_params.max_outer_iterations = sopparms.getMaxOuterIterations();

    sim_params.volume_constraint = sopparms.getVolumeConstraint();

    bool collider_material_id_parse_error = false;

    // Get frictional contact params.
    const auto &sop_frictional_contacts = sopparms.getFrictionalContacts();
    rust::Vec<softy::FrictionalContactParams> frictional_contact_vec;
    for (const auto &sop_fc : sop_frictional_contacts)
    {
        softy::FrictionalContactParams fc_params;
        fc_params.object_material_id = sop_fc.objectmaterialid;

        // Parse a string of integers into an std::vector<uint32_t>
        UT_String ut_collider_material_ids_str(sop_fc.collidermaterialids);

        std::stringstream ss(ut_collider_material_ids_str.toStdString());
        std::string token;
        while (std::getline(ss, token, ' '))
        {
            if (!token.empty())
            {
                try
                {
                    fc_params.collider_material_ids.push_back(std::stoul(token));
                }
                catch (...)
                {
                    collider_material_id_parse_error = true;
                }
            }
        }

        switch (static_cast<SOP_SoftySceneEnums::Kernel>(sop_fc.kernel))
        {
            case SOP_SoftySceneEnums::Kernel::INTERPOLATING:
                fc_params.kernel = softy::Kernel::Interpolating;
                break;
            case SOP_SoftySceneEnums::Kernel::APPROXIMATE:
                fc_params.kernel = softy::Kernel::Approximate;
                break;
            case SOP_SoftySceneEnums::Kernel::CUBIC:
                fc_params.kernel = softy::Kernel::Cubic;
                break;
            case SOP_SoftySceneEnums::Kernel::GLOBAL:
                fc_params.kernel = softy::Kernel::Global;
                break;
        }

        fc_params.radius_multiplier = sop_fc.radiusmult;
        fc_params.smoothness_tolerance = sop_fc.smoothtol;
        fc_params.contact_offset = sop_fc.contactoffset;
        fc_params.use_fixed = sop_fc.usefixed;
        fc_params.dynamic_cof = sop_fc.dynamiccof;
        switch (static_cast<SOP_SoftySceneEnums::FrictionProfile>(sopparms.getFrictionProfile()))
        {
            case SOP_SoftySceneEnums::FrictionProfile::STABILIZED:
                fc_params.friction_profile = softy::FrictionProfile::Stabilized;
                break;
            case SOP_SoftySceneEnums::FrictionProfile::QUADRATIC:
                fc_params.friction_profile = softy::FrictionProfile::Quadratic;
                break;
        }
        sim_params.frictional_contacts.push_back(fc_params);
    }

    sim_params.derivative_test = sopparms.getDerivativeTest();
    sim_params.friction_tolerance = sopparms.getFrictionTolerance();
    sim_params.contact_tolerance = sopparms.getContactTolerance();
    sim_params.contact_iterations = sopparms.getContactIterations();

    return std::make_pair(sim_params, collider_material_id_parse_error);
}

// Entry point to the SOP
void SOP_SoftySceneVerb::cook(const SOP_NodeVerb::CookParms &cookparms) const
{
    auto add_cook_result_message = [&](hdkrs::CookResult &cook_result) -> bool {
        auto &msg = cook_result.message;
        switch (cook_result.tag) {
            case hdkrs::CookResultTag::SUCCESS:
                if (msg != rust::String()) {
                    cookparms.sopAddMessage(UT_ERROR_OUTSTREAM, msg.c_str());
                }
                break;
            case hdkrs::CookResultTag::WARNING:
                cookparms.sopAddWarning(UT_ERROR_OUTSTREAM, msg.c_str());
                break;
            case hdkrs::CookResultTag::ERROR:
                cookparms.sopAddError(UT_ERROR_OUTSTREAM, msg.c_str());
                std::cerr << msg.c_str() << std::endl;
                return false;
        }
        return true;
    };

    auto context = cookparms.getContext();

    // Create mesh from first frame.
    auto mesh = softy::new_mesh();

    const GU_Detail *input0 = cookparms.inputGeo(0);
    if (!input0)
    {
        // No inputs, nothing to do.
        return;
    }

    mesh->set(*input0);

    softy::SimParams sim_params;
    bool collider_material_id_parse_error;
    std::tie(sim_params, collider_material_id_parse_error) =
            build_sim_params(std::move(cookparms.parms<SOP_SoftySceneParms>()));

    auto scene_result = softy::new_scene(std::move(mesh), sim_params);
    if (!add_cook_result_message(scene_result.cook_result)) {
        return;
    }

    auto scene = std::move(scene_result.scene);

    // Add keyframes
    auto frame_range = cookparms.parms<SOP_SoftySceneParms>().getFrameRange();
    int begin_frame = frame_range[0];
    int end_frame = frame_range[1];

    std::vector<UT_Vector3> prev_pos;
    prev_pos.reserve(input0->getNumPoints());
    GA_Offset ptoff;
    GA_FOR_ALL_PTOFF(input0, ptoff) {
        prev_pos.push_back(input0->getPos3(ptoff));
    }

    for (int frame = begin_frame; frame <= end_frame; ++frame) {
        OP_AutoLockInputs inputs(cookparms.getNode()); // Unlocked for next loop iteration.

        // Convert to seconds
        fpreal t = OPgetDirector()->getChannelManager()->getTime((fpreal)frame);
        context.setTime(t);

        // Lock input at the specified time.
        if (inputs.lockInput(0, context) >= UT_ERROR_ABORT)
            continue;

        // Check if the positions have changed.
        fpreal norm_squared = 0.0;
        size_t i = 0;
        GA_Offset ptoff;
        GA_FOR_ALL_PTOFF(input0, ptoff) {
            if (i >= prev_pos.size()) {
                break;
            }
            auto new_pos = input0->getPos3(ptoff);
            norm_squared += (prev_pos[i] - new_pos).length2();
            prev_pos[i] = new_pos; // Update positions for next iteration.
            i += 1;
        }

        if (norm_squared <= 0.0) {
            continue;
        }

        auto points = softy::new_point_cloud();
        points->set(*input0);
        auto keyframe_result = scene->add_keyframe(frame-begin_frame, std::move(points));

        if (!add_cook_result_message(keyframe_result.cook_result)) {
            return;
        }
    }

    auto scene_path = cookparms.parms<SOP_SoftySceneParms>().getSceneFile().c_str();
    auto save_result = scene->save(scene_path);

    if (!add_cook_result_message(save_result.cook_result)) {
        return;
    }

    if (collider_material_id_parse_error)
    {
        cookparms.sopAddWarning(UT_ERROR_OUTSTREAM, "Failed to parse some of the frictional contact collider material ids");
    }
}
