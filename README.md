# PyNanoInstantMeshes

Python bindings for [Instant Meshes](https://github.com/wjakob/instant-meshes), a tool for automatic remeshing of 3D surfaces.

## Installation

```bash
pip install PyNanoInstantMeshes
```

## Usage

```python
import numpy as np
import pynanoinstantmeshes as PyNIM

# vertices: float32 numpy array of shape (N, 3)
# faces:    uint32  numpy array of shape (M, 3)
new_verts, new_faces = PyNIM.remesh(vertices, faces, vertex_count=1000)
```

### Example with trimesh

```python
import trimesh
import numpy as np
import pynanoinstantmeshes as PyNIM

# Load or create a mesh
mesh = trimesh.primitives.Capsule(radius=1, height=2).to_mesh()

vertices = np.asarray(mesh.vertices, dtype=np.float32)
faces    = np.asarray(mesh.faces,    dtype=np.uint32)

# Remesh to approximately 500 vertices
new_verts, new_faces = PyNIM.remesh(vertices, faces, vertex_count=500)

result = trimesh.Trimesh(vertices=new_verts, faces=new_faces)
result.show()
```

## API Reference

### `pynanoinstantmeshes.remesh`

```python
pynanoinstantmeshes.remesh(
    verts,
    faces,
    vertex_count,
    rosy=4,
    posy=4,
    scale=-1.0,
    face_count=-1,
    creaseAngle=0.0,
    align_to_boundaries=False,
    extrinsic=True,
    smooth_iter=0,
    knn_points=1000,
    deterministic=False,
) -> tuple[np.ndarray, np.ndarray]
```

Remeshes the input mesh and returns the new mesh as a `(vertices, faces)` tuple.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verts` | `np.ndarray` (N×3, float32) | — | Input vertex positions. |
| `faces` | `np.ndarray` (M×3, uint32) | — | Input triangle faces. Pass an empty array to treat `verts` as a point cloud. |
| `vertex_count` | `int` | — | Target number of vertices in the output mesh. Mutually exclusive with `scale` and `face_count`; the first positive value among the three is used. |
| `rosy` | `int` | `4` | Rotational symmetry order of the orientation field (`2` or `4`). |
| `posy` | `int` | `4` | Positional symmetry order (`4` = quad output, `6` = triangle output). |
| `scale` | `float` | `-1.0` | Target edge length. When positive, overrides `vertex_count` and `face_count`. |
| `face_count` | `int` | `-1` | Target number of faces in the output mesh. Used when positive and `scale ≤ 0`. |
| `creaseAngle` | `float` | `0.0` | Dihedral-angle threshold (degrees) for crease detection. Edges whose dihedral angle exceeds this value are treated as sharp creases. |
| `align_to_boundaries` | `bool` | `False` | When `True`, aligns the orientation field to mesh boundaries. |
| `extrinsic` | `bool` | `True` | Use extrinsic (world-space) rather than intrinsic (surface-space) smoothness for the orientation field. |
| `smooth_iter` | `int` | `0` | Number of Laplacian smoothing iterations applied to the output mesh. |
| `knn_points` | `int` | `1000` | Number of nearest neighbours used when processing a point cloud (`faces` is empty). |
| `deterministic` | `bool` | `False` | Use a deterministic (single-threaded) algorithm for reproducible results. |

#### Returns

A tuple `(new_verts, new_faces)` where:

- `new_verts` – `np.ndarray` of shape `(V, 3)`, float32 — output vertex positions.
- `new_faces` – `np.ndarray` of shape `(F, D)`, uint32 — output face indices, where `D = 4` for quad meshes (`posy=4`) and `D = 3` for triangle meshes (`posy=6`).

## See Also

- [tests/test_remeshing.py](tests/test_remeshing.py) — comprehensive test cases covering all parameters.
- [Instant Meshes paper](https://igl.ethz.ch/projects/instant-meshes/) — the underlying algorithm.
