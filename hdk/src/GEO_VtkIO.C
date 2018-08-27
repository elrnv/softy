#include <GU/GU_Detail.h>
#include <GU/GU_PrimVolume.h>
#include <GEO/GEO_AttributeHandle.h>
#include <GEO/GEO_IOTranslator.h>
#include <UT/UT_IStream.h>
#include <SOP/SOP_Node.h>
#include <UT/UT_IOTable.h>
#include <boost/variant.hpp>
#include <hdkrs/io.h>
#include <iostream>

#include "GEO_VtkIO.h"

using namespace hdkrs;

GEO_IOTranslator *
GEO_VtkIO::duplicate() const
{
    return new GEO_VtkIO(*this);
}

const char *
GEO_VtkIO::formatName() const
{
    return "Visualization ToolKit (VTK) Legacy Format";
}

int
GEO_VtkIO::checkExtension(const char *name) 
{
    UT_String sname(name);
    if (sname.fileExtension() && !strcmp(sname.fileExtension(), ".vtk"))
        return true;
    return false;
}

int
GEO_VtkIO::checkMagicNumber(unsigned magic)
{
    return 0;
}

struct AddMesh : public boost::static_visitor<bool>
{
    AddMesh(GEO_Detail* detail) : detail(static_cast<GU_Detail*>(detail)) {}
    bool operator()( OwnedPtr<TetMesh> tetmesh ) const
    {
        mesh::add_tetmesh(detail, std::move(tetmesh));
        return true;
    }
    bool operator()( OwnedPtr<PolyMesh> polymesh ) const
    {
        mesh::add_polymesh(detail, std::move(polymesh));
        return true;
    }
    bool operator()( boost::blank nothing ) const
    {
        return false;
    }

    GU_Detail* detail;
};

GA_Detail::IOStatus
GEO_VtkIO::fileLoad(GEO_Detail *detail, UT_IStream &is, bool)
{
    using namespace hdkrs;
    if (!detail) // nothing to do
        return GA_Detail::IOStatus(true);

    UT_WorkBuffer buf;
    bool success = is.getAll(buf);
    if (!success)
        return GA_Detail::IOStatus(success);
    exint size = buf.length();
    auto data = buf.buffer();
    io::MeshVariant mesh = io::parse_vtk_mesh(data, size);
    boost::apply_visitor( AddMesh(detail), std::move(mesh) );
    return GA_Detail::IOStatus(success);
}

GA_Detail::IOStatus
GEO_VtkIO::fileSave(const GEO_Detail *detail, std::ostream &os)
{
    if (!detail) // nothing to do
        return GA_Detail::IOStatus(true);

    // Try to save the tetmesh first
    OwnedPtr<TetMesh> tetmesh = mesh::build_tetmesh(static_cast<const GU_Detail*>(detail));
    if (tetmesh) {
        auto buf = io::ByteBuffer::write_vtk_mesh(std::move(tetmesh));
        os.write(buf.data(), buf.size());
        return GA_Detail::IOStatus(true);
    }

    // If no tets are found we try to save the polymesh
    OwnedPtr<PolyMesh> polymesh = mesh::build_polymesh(static_cast<const GU_Detail*>(detail));
    if (polymesh) {
        auto buf = io::ByteBuffer::write_vtk_mesh(std::move(polymesh));
        os.write(buf.data(), buf.size());
        return GA_Detail::IOStatus(true);
    }

    return GA_Detail::IOStatus(false);
}

void newGeometryIO(void *)
{
    GU_Detail::registerIOTranslator(new GEO_VtkIO());
    UT_ExtensionList *geoextension;
    geoextension = UTgetGeoExtensions();
    if (!geoextension->findExtension("vtk"))
        geoextension->addExtension("vtk");
}
