#pragma once

#include <SOP/SOP_Node.h>
#include <UT/UT_StringHolder.h>

class SOP_Softy : public SOP_Node
{
public:
    static PRM_Template *buildTemplates();
    static OP_Node *myConstructor(OP_Network* net, const char * name, OP_Operator * op)
    {
        return new SOP_Softy(net, name, op);
    }

    static const UT_StringHolder theSOPTypeName;

    virtual const SOP_NodeVerb *cookVerb() const override;

    static int clearSolverCache(void *data, int index, float t, const PRM_Template *);
    static int clearSolverLogs(void *data, int index, float t, const PRM_Template *);

protected:
    SOP_Softy(OP_Network *net, const char *name, OP_Operator *op)
        : SOP_Node(net, name, op)
    {
        // All verb SOPs must manage data IDs to track what's changed
        // from cook to cook
        mySopFlags.setManagesDataIDs(true);
    }

    virtual ~SOP_Softy() {}

    // Since this SOP implements a verb, cookMySop just delegates to the verb.
    virtual OP_ERROR cookMySop(OP_Context &context) override
    {
        return cookMyselfAsVerb(context);
    }
};
