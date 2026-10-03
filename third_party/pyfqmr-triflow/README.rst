pyfqmr-triflow
==============

Triflow-customized fork of pyfqmr_ — a Cython wrapper around a C++
Fast Quadric Mesh Reduction library (sp4cerat_).

.. _pyfqmr: https://github.com/Kramer84/pyfqmr-Fast-quadric-Mesh-Reduction
.. _sp4cerat: https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification

This fork is maintained solely as a build-time dependency of triflow.
It is **not** published to PyPI; triflow users compile it from source.

Relationship to upstream
------------------------

Forked from ``Kramer84/pyfqmr`` at commit ``26f8fa0`` (upstream
``v0.5.0``). The distribution name is ``pyfqmr-triflow`` and the
import name is ``pyfqmr_triflow`` so this fork can coexist with
upstream ``pyfqmr`` in the same Python environment.

Building from source
--------------------

.. code:: bash

    git clone <your-private-repo-url>
    cd pyfqmr-triflow
    pip install .

Build requires a C++ compiler, Cython, and NumPy. See
``pyproject.toml`` for the full build-time dependency list. No
runtime dependency beyond NumPy.

Usage
-----

.. code:: python

    import pyfqmr_triflow as pyfqmr
    import trimesh

    mesh = trimesh.load("input.stl")
    s = pyfqmr.Simplify()
    s.setMesh(mesh.vertices, mesh.faces)
    s.simplify_mesh(
        target_count=1500,
        aggressiveness=7,
        preserve_border=False,
        custom_quadratics=...,      # see below, optional
        target_vertex_ids=...,      # see below, optional
        min_edge_length=-1.0,       # see below, optional
    )
    verts, faces, normals = s.getMesh()

What's different from upstream
------------------------------

API additions on ``Simplify.simplify_mesh``:

- ``custom_quadratics`` — per-vertex ``(n_vertices, 10)`` float64
  array of symmetric-4x4 quadric matrices added on top of the
  geometric quadric at initialization. Used to bias collapse
  targets toward arbitrary positions (e.g. a skeleton, an SDF
  sample, a smoothed surface) without polluting the reported
  geometric error. The 10 values per vertex are the upper-triangular
  entries of the 4x4 matrix in order::

      [m11, m12, m13, m14, m22, m23, m24, m33, m34, m44]

- ``target_vertex_ids`` — per-vertex int32 array tagging each mesh
  vertex with a group id (or ``-1`` for ungrouped). Collapses are
  constrained by three rules:

  1. A vertex tagged as its own "seed" (``tid == vertex_index``)
     can only be collapsed *into*, not *away from*, a non-seed.
  2. An edge between two different target groups may only be
     collapsed if the edge is shorter than ``min_edge_length``.
  3. A collapse that would split a target group's ring (detected
     via union-find) is rejected.

- ``min_edge_length`` — threshold controlling the intra-group
  collapse rule above. Default ``-1.0`` means "no short-edge
  override".

Behavioral changes to the core algorithm:

- Quadrics are area-weighted at initialization, and the reported
  error is normalized by the sum of the two endpoint areas. This
  makes the error scale-invariant and reduces the influence of tiny
  triangles on collapse order.
- A non-manifold pinch check rejects any collapse whose shared-ring
  structure would produce a non-manifold vertex.
- Border tracking is per-edge (``Triangle::border[3]``) rather than
  per-vertex only. Collapses along a border edge are allowed; a
  collapse whose two endpoints are both borders but whose shared
  edge is not a border is rejected.
- Border endpoints receive a weighted orthogonal-plane quadric to
  pin them in place. The weight (``border_weight``) is a
  namespace-level constant in ``Simplify.h``, not currently exposed
  to the Python API.
- The quadric-optimal collapse position falls back to the
  midpoint / endpoints when it lies more than 2x the edge length
  away from the midpoint. This avoids runaway positions from
  ill-conditioned quadric solves.
- ``flipped()`` no longer early-returns on the first bad triangle;
  it always scans the full one-ring so that the ``deleted[]``
  output array is fully populated. This is required by the border
  propagation code.
- Termination now also breaks when an iteration makes no progress
  and the threshold has saturated at ``threshold_lossless``.
- All reported error in ``calculate_error`` is computed against the
  *geometric* quadric ``q_geom``, not the combined geometric +
  custom quadric, so the custom-quadratic bias does not corrupt the
  error reported to the iteration-threshold logic.

Reserved but not exposed
------------------------

The following tunables exist as namespace-level variables in
``Simplify.h`` but are not wired to the Python API. They are kept
for future extensibility:

- ``border_weight`` (double) — weight applied to the orthogonal
  border quadric; default ``1000.0``.
- ``binary_search_steps`` (int) — number of binary-search iterations
  when the quadric-optimal point causes a triangle flip; default
  ``0`` (code path disabled).
- ``override_quadratics`` (bool) — placeholder for a future
  "replace instead of add" custom-quadratic mode.

Grep for ``[triflow]`` in ``pyfqmr_triflow/Simplify.h`` to find the
fork delta inline.

License
-------

MIT, inherited from upstream ``pyfqmr``. See ``LICENSE``.
