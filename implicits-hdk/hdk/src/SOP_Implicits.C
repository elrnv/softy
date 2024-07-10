#include "SOP_Implicits.h"

// Needed for template generation with the ds file.
#include "SOP_Implicits.proto.h"

#include <rust/cxx.h>
#include <implicits/src/lib.rs.h>

// Required for proper loading.
#include <UT/UT_DSOVersion.h>

#include <UT/UT_Interrupt.h>
#include <UT/UT_StringHolder.h>
#include <PRM/PRM_Include.h>
#include <PRM/PRM_TemplateBuilder.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>

const UT_StringHolder SOP_Implicits::theSOPTypeName("hdk_implicits");

// Register sop operator
void
newSopOperator(OP_OperatorTable *table)
{
    table->addOperator(new OP_Operator(
                SOP_Implicits::theSOPTypeName,   // Internal name
                "Implicits",                     // UI name
                SOP_Implicits::myConstructor,    // How to build the SOP
                SOP_Implicits::buildTemplates(), // My parameters
                2,                              // Min # of sources
                2,                              // Max # of sources
                nullptr,                        // Local variables
                OP_FLAG_GENERATOR));            // Flag it as generator
}

static const char *theDsFile = R"THEDSFILE(
{
    name implicits

    parm {
        name "action"
        label "Action"
        type ordinal
        default { "0" }
        menu {
            "potential"   "Compute Potential"
            "project"     "Project Out"
        }
    }

    parm {
        name "isovalue"
        cppname "IsoValue"
        label "Iso Value"
        type float
        default { "0.0" }
        range { 0.0 10.0 }
        hidewhen "{ action == potential }"
    }

    parm {
        name "projectbelow"
        cppname "ProjectBelow"
        label "Project Below"
        type toggle
        default { "off" }
        hidewhen "{ action == potential }"
    }

    parm {
        name "kernel"
        label "Kernel"
        type ordinal
        default { "1" }
        menu {
            "smooth" "Local smooth"
            "approximate" "Local approximately interpolating"
            "cubic" "Local cubic"
            "interpolating" "Local interpolating"
            "global" "Global inverse squared distance"
            "hrbf" "HRBF potential"
        }
    }

    parm {
        name "sampletype"
        cppname "SampleType"
        label "Sample Type"
        type ordinal
        default { "0" }
        menu {
            "vertex"   "Vertex"
            "face"     "Face"
        }
    }

    parm {
        name "radiusmultiplier"
        cppname "RadiusMultiplier"
        label "Radius Multiplier"
        type float
        default { "1.1" }
        range { 0.0 10.0 }
        hidewhen "{ kernel == global } { kernel == hrbf }"
    }

    parm {
        name "usebaseradius"
        cppname "UseBaseRadius"
        type toggle 
        nolabel
        joinnext
        default { "off" }
        hidewhen "{ kernel == global } { kernel == hrbf }"
    }

    parm {
        name "baseradius"
        cppname "BaseRadius"
        label "Base Radius"
        type float
        default { "0.0" }
        range { 0.0 10.0 }
        hidewhen "{ kernel == global } { kernel == hrbf }"
        disablewhen "{ usebaseradius == 0 }"
    }

    parm {
        name "tolerance"
        label "Tolerance"
        type log
        default { "1e-5" }
        range { 0.0 1.0 }
        hidewhen "{ kernel == cubic } { kernel == hrbf } { kernel == interpolating }"
    }

    parm {
        name "debug"
        label "Debug"
        type toggle
        default { "off" }
        disablewhen "{ action == project }"
    }

    groupsimple {
        name "bgfield"
        label "Background Field"

        parm {
            name "bgpotential"
            cppname "BgPotential"
            label "Source"
            type ordinal
            default { "0" }
            menu {
                "zero"       "Zero"
                "input"      "From Input"
                "distance"   "Signed Distance to Closest"
            }
            hidewhen "{ kernel == global } { kernel == hrbf }"
        }

        parm {
            name "bgweighted"
            cppname "BgWeighted"
            label "Weighted"
            type toggle
            default { "off" }
        }

    }
}
)THEDSFILE";


PRM_Template *
SOP_Implicits::buildTemplates()
{
    static PRM_TemplateBuilder templ("SOP_Implicits.C", theDsFile);
    return templ.templates();
}

class SOP_ImplicitsVerb : public SOP_NodeVerb
{
    public:
        SOP_ImplicitsVerb() {}
        virtual ~SOP_ImplicitsVerb() {}

        virtual SOP_NodeParms *allocParms() const { return new SOP_ImplicitsParms(); }
        virtual UT_StringHolder name() const { return SOP_Implicits::theSOPTypeName; }

        virtual CookMode cookMode(const SOP_NodeParms *parms) const { return COOK_INPLACE; }

        virtual void cook(const CookParms &cookparms) const;

        static const SOP_NodeVerb::Register<SOP_ImplicitsVerb> theVerb;
};

const SOP_NodeVerb::Register<SOP_ImplicitsVerb> SOP_ImplicitsVerb::theVerb;

const SOP_NodeVerb *
SOP_Implicits::cookVerb() const
{
    return SOP_ImplicitsVerb::theVerb.get();
}

// Entry point to the SOP
void
SOP_ImplicitsVerb::cook(const SOP_NodeVerb::CookParms &cookparms) const
{
    using namespace hdkrs;

    auto &&sopparms = cookparms.parms<SOP_ImplicitsParms>();
    GU_Detail *detail = cookparms.gdh().gdpNC();
    if (!detail) {
        cookparms.sopAddError(UT_ERROR_OUTSTREAM, "Missing input query points");
        return;
    }

    const GU_Detail *input1 = cookparms.inputGeo(1);
    if (!input1) {
        cookparms.sopAddError(UT_ERROR_OUTSTREAM, "Missing polygonal surface");
        return;
    }

    try {
        // Initialize interrupt checker
        auto interrupt_checker = std::make_unique<hdkrs::InterruptChecker>("Solving MLS");

        // Gather query points and surface mesh
        auto querymesh = implicits::build_pointcloud(*detail);
        auto polymesh = implicits::build_polymesh(*input1);

        // Gather parameters
        ISO_Params iso_params = iso_default_params();
        iso_params.tolerance = sopparms.getTolerance();
        iso_params.radius_multiplier = sopparms.getRadiusMultiplier();
        if (sopparms.getUseBaseRadius()) {
            iso_params.base_radius = sopparms.getBaseRadius();
        }
        // Note that when casting directly to an enum, the options must be in the same order in the enum as given
        // by the parameter value.
        iso_params.kernel = static_cast<ISO_KernelType>(sopparms.getKernel());
        iso_params.background_field = static_cast<ISO_BackgroundFieldType>(sopparms.getBgPotential());
        iso_params.weighted = sopparms.getBgWeighted();
        iso_params.sample_type = static_cast<ISO_SampleType>(sopparms.getSampleType());

        implicits::Params params = implicits::default_params();
        params.action = static_cast<implicits::Action>(sopparms.getAction());
        params.iso_value = sopparms.getIsoValue();
        params.project_below = sopparms.getProjectBelow();
        params.debug = sopparms.getDebug();
        params.iso_params = iso_params;

        hdkrs::CookResult res = implicits::cook(*querymesh, *polymesh, params, std::move(interrupt_checker));

        switch (res.tag) {
            case hdkrs::CookResultTag::SUCCESS: cookparms.sopAddMessage(UT_ERROR_OUTSTREAM, res.message.c_str()); break;
            case hdkrs::CookResultTag::WARNING: cookparms.sopAddWarning(UT_ERROR_OUTSTREAM, res.message.c_str()); break;
            case hdkrs::CookResultTag::ERROR: cookparms.sopAddError(UT_ERROR_OUTSTREAM, res.message.c_str()); break;
        }

        // Add the query points back into the current detail
        implicits::update_points(*detail, *querymesh);

    } catch (const std::runtime_error& e) {
        cookparms.sopAddError(UT_ERROR_OUTSTREAM, (std::string("Error building meshes: ")  + e.what()).c_str());
    }
}
