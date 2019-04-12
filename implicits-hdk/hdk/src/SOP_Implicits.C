#include "SOP_Implicits.h"

// Needed for template generation with the ds file.
#include "SOP_Implicits.proto.h"

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
#include <implicits-hdk.h>

const UT_StringHolder SOP_Implicits::theSOPTypeName("hdk_implicits"_sh);

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
        name "kernel"
        label "Kernel"
        type ordinal
        default { "1" }
        menu {
            "interpolating" "Local interpolating"
            "approximate" "Local approximately interpolating"
            "cubic" "Local cubic"
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
        name "tolerance"
        label "Tolerance"
        type log
        default { "1e-5" }
        range { 0.0 1.0 }
        hidewhen "{ kernel == cubic } { kernel == hrbf } { kernel == interpolating }"
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
    static PRM_TemplateBuilder templ("SOP_Implicits.C"_sh, theDsFile);
    return templ.templates();
}

class SOP_ImplicitsVerb : public SOP_NodeVerb
{
    public:
        SOP_ImplicitsVerb() {}
        virtual ~SOP_ImplicitsVerb() {}

        virtual SOP_NodeParms *allocParms() const { return new SOP_ImplicitsParms(); }
        virtual UT_StringHolder name() const { return SOP_Implicits::theSOPTypeName; }

        virtual CookMode cookMode(const SOP_NodeParms *parms) const { return COOK_GENERATOR; }

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
    using namespace hdkrs::mesh;
    using namespace interrupt;

    auto &&sopparms = cookparms.parms<SOP_ImplicitsParms>();
    const GU_Detail *input0 = cookparms.inputGeo(0);
    OwnedPtr<HR_PolyMesh> samplemesh(nullptr);
    if (input0) {
        samplemesh = build_polymesh(input0);
    }

    const GU_Detail *input1 = cookparms.inputGeo(1);
    OwnedPtr<HR_PolyMesh> polymesh(nullptr);
    if (input1) {
        polymesh = build_polymesh(input1);
    }

    InterruptChecker interrupt_checker("Solving MLS");

    // Gather parameters
    Params params;
    params.action = static_cast<int>(sopparms.getAction());
    params.iso_value = sopparms.getIsoValue();
    params.tolerance = sopparms.getTolerance();
    params.radius_multiplier = sopparms.getRadiusMultiplier();
    params.kernel = static_cast<int>(sopparms.getKernel());
    params.background_potential = static_cast<int>(sopparms.getBgPotential());
    params.background_potential_weighted = sopparms.getBgWeighted();
    params.sample_type = static_cast<int>(sopparms.getSampleType());

    HR_CookResult res = el_iso_cook( samplemesh.get(), polymesh.get(), params, &interrupt_checker, check_interrupt );

    switch (res.tag) {
        case HRCookResultTag::HR_SUCCESS: cookparms.sopAddMessage(UT_ERROR_OUTSTREAM, res.message); break;
        case HRCookResultTag::HR_WARNING: cookparms.sopAddWarning(UT_ERROR_OUTSTREAM, res.message); break;
        case HRCookResultTag::HR_ERROR: cookparms.sopAddError(UT_ERROR_OUTSTREAM, res.message); break;
    }

    hr_free_result(res);

    GU_Detail *detail = cookparms.gdh().gdpNC();

    // Add the samples back into the current detail
    add_polymesh(detail, std::move(samplemesh));
}
