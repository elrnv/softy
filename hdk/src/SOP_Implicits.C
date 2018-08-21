#include "SOP_Implicits.h"

// Needed for template generation with the ds file.
#include "SOP_Implicits.proto.h"

#include "mesh.h"
#include "interrupt.h"

// Required for proper loading.
#include <UT/UT_DSOVersion.h>

#include <UT/UT_Interrupt.h>
#include <UT/UT_StringHolder.h>
#include <PRM/PRM_Include.h>
#include <PRM/PRM_TemplateBuilder.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <implicitshdk.h>

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
        name "kernel"
        label "Kernel"
        type ordinal
        default { "interpolating" }
        menu {
            "interpolating" "Local interpolating"
            "approximate" "Local approximately interpolating"
            "cubic" "Local cubic"
            "global" "Global inverse squared distance"
            "hrbf" "HRBF potential"
        }
    }
    parm {
        name "radius"
        label "Radius"
        type float
        default { "20" }
        range { 0.0 100.0 }
        hidewhen "{ kernel == global } { kernel == hrbf }"
    }

    parm {
        name "tolerance"
        label "Tolerance"
        type float
        default { "1e-5" }
        range { 0.0 1.0 }
        hidewhen "{ kernel == cubic } { kernel == hrbf } { kernel == interpolating }"
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
    using namespace mesh;
    using namespace interrupt;

    auto &&sopparms = cookparms.parms<SOP_ImplicitsParms>();
    const GU_Detail *input0 = cookparms.inputGeo(0);
    implicits::PolyMesh *samplemesh = nullptr;
    if (input0) {
        samplemesh = build_polymesh(input0);
    }

    const GU_Detail *input1 = cookparms.inputGeo(1);
    implicits::PolyMesh *polymesh = nullptr;
    if (input1) {
        polymesh = build_polymesh(input1);
    }

    InterruptChecker interrupt("Solving MLS");

    // Gather parameters
    implicits::Params implicits_params;
    implicits_params.tolerance = sopparms.getTolerance();
    implicits_params.radius = sopparms.getRadius();
    implicits_params.kernel = static_cast<int>(sopparms.getKernel());

    implicits::CookResult res = implicits::cook( samplemesh, polymesh, implicits_params, &interrupt, check_interrupt );

    switch (res.tag) {
        case implicits::CookResultTag::Success: cookparms.sopAddMessage(UT_ERROR_OUTSTREAM, res.message); break;
        case implicits::CookResultTag::Warning: cookparms.sopAddWarning(UT_ERROR_OUTSTREAM, res.message); break;
        case implicits::CookResultTag::Error: cookparms.sopAddError(UT_ERROR_OUTSTREAM, res.message); break;
    }

    implicits::free_result(res);

    GU_Detail *detail = cookparms.gdh().gdpNC();

    // Add the samples back into the current detail
    add_polymesh(detail, samplemesh);

    // We must release the memory using the same api that allocated it.
    implicits::free_polymesh(polymesh);
    implicits::free_polymesh(samplemesh);
}
