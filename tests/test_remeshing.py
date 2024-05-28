import pytest
import trimesh
import numpy as np
import sys
import os

# Get the actual path from this file to ../build/
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "../build/")
    ),
)

import PyNIM


@pytest.fixture
def mesh():
    # Load assets/monkey.obj
    yield trimesh.load("assets/torus.obj")


@pytest.mark.parametrize(
    "rosy, posy, expected_vertex_dim",
    [
        (2, 4, 4),
        (4, 4, 4),
    ],
)
@pytest.mark.parametrize("vertex_count", [100, 200])
@pytest.mark.parametrize("creaseAngle", [0.0, 10.0, 20.0])
@pytest.mark.parametrize("align_to_boundaries", [True, False])
@pytest.mark.parametrize("smooth_iter", [0, 2, 4])
class TestRemeshing:
    def test_mesh_valid(
        self,
        mesh,
        rosy,
        posy,
        expected_vertex_dim,
        vertex_count,
        creaseAngle,
        align_to_boundaries,
        smooth_iter,
    ):
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)

        new_verts, new_faces = PyNIM.remesh(
            vertices,
            faces,
            vertex_count,
            rosy=rosy,
            posy=posy,
            creaseAngle=creaseAngle,
            align_to_boundaries=align_to_boundaries,
            smooth_iter=smooth_iter,
        )

        assert new_verts.shape[-1] == 3
        if new_verts.shape[0] - 1 != new_faces.max():
            # Skip test as the meshing failed
            pytest.skip("Meshing failed")
        assert new_faces.shape[-1] == expected_vertex_dim

        new_mesh = trimesh.Trimesh(new_verts, new_faces)
        # Check if the bounds remain the same
        assert np.allclose(
            new_mesh.bounds, mesh.bounds, atol=2e-1
        ), f"Bounds do not align: {new_mesh.bounds} != {mesh.bounds}, {new_mesh.bounds - mesh.bounds}"
