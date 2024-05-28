#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include "dedge.h"
#include "subdivide.h"
#include "meshstats.h"
#include "hierarchy.h"
#include "field.h"
#include "normal.h"
#include "extract.h"
#include "bvh.h"
#include "common.h"

namespace nb = nanobind;
using namespace nb::literals;

std::tuple<
    nb::ndarray<nb::numpy, const Float, nb::ndim<2>>,
    nb::ndarray<nb::numpy, const uint32_t, nb::ndim<2>>>
remesh(
    nb::ndarray<Float, nb::ndim<2>, nb::device::cpu> verts,
    nb::ndarray<uint32_t, nb::ndim<2>, nb::device::cpu> faces,
    int vertex_count,
    int rosy = 4, int posy = 4, Float scale = -1.0, int face_count = -1,
    Float creaseAngle = 0.0,
    bool align_to_boundaries = false, bool extrinsic = true,
    int smooth_iter = 0, int knn_points = 1000, bool deterministic = false)
{
    MatrixXu F;
    F.resize(faces.shape(1), faces.shape(0));
    MatrixXf V, N;
    V.resize(verts.shape(1), verts.shape(0));
    VectorXf A;
    std::set<uint32_t> crease_in, crease_out;
    BVH *bvh = nullptr;
    AdjacencyMatrix adj = nullptr;

    // Convert the numpy arrays to Eigen matrices
    auto Vv = verts.view();
    for (int i = 0; i < Vv.shape(0); i++)
    {
        for (int j = 0; j < Vv.shape(1); j++)
        {
            V(j, i) = Vv(i, j);
        }
    }
    auto Fv = faces.view();
    for (int i = 0; i < Fv.shape(0); i++)
    {
        for (int j = 0; j < Fv.shape(1); j++)
        {
            F(j, i) = Fv(i, j);
        }
    }
    bool pointcloud = F.size() == 0;

    Timer<> timer;
    MeshStats stats = compute_mesh_stats(F, V, deterministic);

    if (pointcloud)
    {
        bvh = new BVH(&F, &V, &N, stats.mAABB);
        bvh->build();
        adj = generate_adjacency_matrix_pointcloud(V, N, bvh, stats, knn_points, deterministic);
        A.resize(V.cols());
        A.setConstant(1.0f);
    }

    if (scale < 0 && vertex_count < 0 && face_count < 0)
    {
        cout << "No target vertex count/face count/scale argument provided. "
                "Setting to the default of 1/16 * input vertex count."
             << endl;
        vertex_count = V.cols() / 16;
    }

    if (scale > 0)
    {
        float face_area = posy == 4 ? (scale * scale) : (std::sqrt(3.f) / 4.f * scale * scale);
        face_count = stats.mSurfaceArea / face_area;
        vertex_count = posy == 4 ? face_count : (face_count / 2);
    }
    else if (face_count > 0)
    {
        float face_area = stats.mSurfaceArea / face_count;
        vertex_count = posy == 4 ? face_count : (face_count / 2);
        scale = posy == 4 ? std::sqrt(face_area) : (2 * std::sqrt(face_area * std::sqrt(1.f / 3.f)));
    }
    else if (vertex_count > 0)
    {
        face_count = posy == 4 ? vertex_count : (vertex_count * 2);
        float face_area = stats.mSurfaceArea / face_count;
        scale = posy == 4 ? std::sqrt(face_area) : (2 * std::sqrt(face_area * std::sqrt(1.f / 3.f)));
    }

    cout << "Output mesh goals (approximate)" << endl;
    cout << "   Vertex count           = " << vertex_count << endl;
    cout << "   Face count             = " << face_count << endl;
    cout << "   Edge length            = " << scale << endl;

    MultiResolutionHierarchy mRes;

    if (!pointcloud)
    {
        /* Subdivide the mesh if necessary */
        VectorXu V2E, E2E;
        VectorXb boundary, nonManifold;
        if (stats.mMaximumEdgeLength * 2 > scale || stats.mMaximumEdgeLength > stats.mAverageEdgeLength * 2)
        {
            cout << "Input mesh is too coarse for the desired output edge length "
                    "(max input mesh edge length="
                 << stats.mMaximumEdgeLength
                 << "), subdividing .." << endl;
            build_dedge(F, V, V2E, E2E, boundary, nonManifold);
            subdivide(F, V, V2E, E2E, boundary, nonManifold, std::min(scale / 2, (float)stats.mAverageEdgeLength * 2), deterministic);
        }

        /* Compute a directed edge data structure */
        build_dedge(F, V, V2E, E2E, boundary, nonManifold);

        /* Compute adjacency matrix */
        adj = generate_adjacency_matrix_uniform(F, V2E, E2E, nonManifold);

        /* Compute vertex/crease normals */
        if (creaseAngle >= 0)
            generate_crease_normals(F, V, V2E, E2E, boundary, nonManifold, creaseAngle, N, crease_in);
        else
            generate_smooth_normals(F, V, V2E, E2E, nonManifold, N);

        /* Compute dual vertex areas */
        compute_dual_vertex_areas(F, V, V2E, E2E, nonManifold, A);

        mRes.setE2E(std::move(E2E));
    }

    /* Build multi-resolution hierarrchy */
    mRes.setAdj(std::move(adj));
    mRes.setF(std::move(F));
    mRes.setV(std::move(V));
    mRes.setA(std::move(A));
    mRes.setN(std::move(N));
    mRes.setScale(scale);
    mRes.build(deterministic);
    mRes.resetSolution();

    if (align_to_boundaries && !pointcloud)
    {
        mRes.clearConstraints();
        for (uint32_t i = 0; i < 3 * mRes.F().cols(); ++i)
        {
            if (mRes.E2E()[i] == INVALID)
            {
                uint32_t i0 = mRes.F()(i % 3, i / 3);
                uint32_t i1 = mRes.F()((i + 1) % 3, i / 3);
                Vector3f p0 = mRes.V().col(i0), p1 = mRes.V().col(i1);
                Vector3f edge = p1 - p0;
                if (edge.squaredNorm() > 0)
                {
                    edge.normalize();
                    mRes.CO().col(i0) = p0;
                    mRes.CO().col(i1) = p1;
                    mRes.CQ().col(i0) = mRes.CQ().col(i1) = edge;
                    mRes.CQw()[i0] = mRes.CQw()[i1] = mRes.COw()[i0] =
                        mRes.COw()[i1] = 1.0f;
                }
            }
        }
        mRes.propagateConstraints(rosy, posy);
    }

    if (bvh)
    {
        bvh->setData(&mRes.F(), &mRes.V(), &mRes.N());
    }
    else if (smooth_iter > 0)
    {
        bvh = new BVH(&mRes.F(), &mRes.V(), &mRes.N(), stats.mAABB);
        bvh->build();
    }

    cout << "Preprocessing is done. (total time excluding file I/O: "
         << timeString(timer.reset()) << ")" << endl;

    Optimizer optimizer(mRes, false);
    optimizer.setRoSy(rosy);
    optimizer.setPoSy(posy);
    optimizer.setExtrinsic(extrinsic);

    cout << "Optimizing orientation field .. ";
    cout.flush();
    optimizer.optimizeOrientations(-1);
    optimizer.notify();
    optimizer.wait();
    cout << "done. (took " << timeString(timer.reset()) << ")" << endl;

    std::map<uint32_t, uint32_t> sing;
    compute_orientation_singularities(mRes, sing, extrinsic, rosy);
    cout << "Orientation field has " << sing.size() << " singularities." << endl;
    timer.reset();

    cout << "Optimizing position field .. ";
    cout.flush();
    optimizer.optimizePositions(-1);
    optimizer.notify();
    optimizer.wait();
    cout << "done. (took " << timeString(timer.reset()) << ")" << endl;

    // std::map<uint32_t, Vector2i> pos_sing;
    // compute_position_singularities(mRes, sing, pos_sing, extrinsic, rosy, posy);
    // cout << "Position field has " << pos_sing.size() << " singularities." << endl;
    // timer.reset();

    optimizer.shutdown();

    MatrixXf O_extr, N_extr, Nf_extr;
    std::vector<std::vector<TaggedLink>> adj_extr;
    extract_graph(mRes, extrinsic, rosy, posy, adj_extr, O_extr, N_extr,
                  crease_in, crease_out, deterministic);

    MatrixXu F_extr;
    extract_faces(adj_extr, O_extr, N_extr, Nf_extr, F_extr, posy,
                  mRes.scale(), crease_out, true, posy == 4, bvh, smooth_iter);
    cout << "Extraction is done. (total time: " << timeString(timer.reset()) << ")" << endl;

    cout << "Faces " << F_extr.cols() << ", " << F_extr.rows() << endl;
    cout << "Verts " << O_extr.cols() << ", " << O_extr.rows() << endl;
    // Create new numpy arrays which house the F and V matrices
    const uint32_t num_verts = O_extr.cols();
    const uint32_t num_faces = F_extr.cols();
    const uint32_t face_dim = F_extr.rows();

    size_t shapea[2] = {num_verts, 3};
    size_t shapeb[2] = {num_faces, face_dim};
    return std::make_tuple(nb::ndarray<nb::numpy, const Float, nb::ndim<2>>(
                               O_extr.transpose().data(), 2, shapea, nb::handle()),
                           nb::ndarray<nb::numpy, const uint32_t, nb::ndim<2>>(
                               F_extr.transpose().data(), 2, shapeb, nb::handle()));
}

NB_MODULE(_pynim, m)
{
    m.def("remesh", &remesh,
          "verts"_a, "faces"_a, "vertex_count"_a, "rosy"_a = 4, "posy"_a = 4, "scale"_a = -1.0,
          "face_count"_a = -1, "creaseAngle"_a = 0.0,
          "align_to_boundaries"_a = false, "extrinsic"_a=true,
          "smooth_iter"_a = 0, "knn_points"_a = 1000, "deterministic"_a = false,
          "Remeshes the input mesh and returns the new mesh as a tuple of vertices and faces.");
}
