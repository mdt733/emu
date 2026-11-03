"""Opt-in v5043 per-world convex triangular-prism collision geometry for MJX-JAX.

Each 50 cm tile has two *independent* convex meshes. We shear their top AND
buried bottom in vertical Z, so top triangles meet at exact shared vertices,
side faces remain vertical, and the number of geoms / vertices never changes.

This is an experimental MJX-JAX path; never mutate MuJoCo's compiled native
mesh vertices during simulation. MJX receives a batched, regenerated
ConvexMesh JAX pytree with fresh face and edge normals at each physics step.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from .terrain_carpet_mjx import TILE_COUNT, TILE_M, TILE_SIZE_Z_M

PRISM_DEPTH_M = .65  # below top: safely spans simultaneous 15deg + signed XY-step corners
PRISMS_PER_TILE = 2
PRISM_COUNT = TILE_COUNT * PRISMS_PER_TILE
PRISM_BODY_NAMES = tuple(f"carpet_prism_body_{i}" for i in range(PRISM_COUNT))
PRISM_GEOM_NAMES = tuple(f"carpet_prism_geom_{i}" for i in range(PRISM_COUNT))
PRISM_MESH_NAMES = tuple(f"carpet_prism_mesh_{i}" for i in range(PRISM_COUNT))


def _local_triangle(triangle):
    h = TILE_M * .5
    if triangle == 0:
        return [(-h, -h), (h, -h), (-h, h)]
    if triangle == 1:
        return [(h, h), (-h, h), (h, -h)]
    raise ValueError(triangle)


def prism_mesh_xml(root, *, active: bool):
    """Add 18 unique six-vertex, five-face convex prism mesh geoms.

    Unique mesh asset IDs are essential: MJX's runtime collision templates
    live in Model._impl.mesh_convex indexed by mesh dataid.
    """
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("prism mesh needs worldbody")
    # CCW surface faces, top z=0, buried underside follows its top vertex by PRISM_DEPTH_M.
    faces = ((0, 1, 2), (5, 4, 3),
             (0, 3, 4), (0, 4, 1),
             (1, 4, 5), (1, 5, 2),
             (2, 5, 3), (2, 3, 0))
    face_string = " ".join(str(v) for face in faces for v in face)
    for k in range(PRISM_COUNT):
        xy = _local_triangle(k % 2)
        verts = [(x, y, 0.) for x, y in xy] + [
            (x, y, -PRISM_DEPTH_M) for x, y in xy]
        vertex_string = " ".join(
            f"{c:.8f}" for vert in verts for c in vert)
        ET.SubElement(asset, "mesh", dict(
            name=PRISM_MESH_NAMES[k],
            vertex=vertex_string, face=face_string))
        body = ET.SubElement(world, "body", dict(
            name=PRISM_BODY_NAMES[k], mocap="true",
            pos=f"0 0 {-40-k}"))
        ET.SubElement(body, "geom", dict(
            name=PRISM_GEOM_NAMES[k], type="mesh",
            mesh=PRISM_MESH_NAMES[k],
            contype="4" if active else "0",
            conaffinity="0", friction="1.5 0.005 0.0001",
            rgba="0.39 0.65 0.39 1"))


def _quat_matrix(q):
    """Scalar-first compiled MuJoCo geom quaternion, local -> body."""
    w,x,y,z = (float(v) for v in q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]],
        dtype=np.float32)


class PrismMeshBuilder:
    """Static face/edge topology; JAX-dynamic six vertices and normals."""

    def __init__(self, mj_model, template, geom_ids):
        self.templates = tuple(template._impl.mesh_convex)
        self.ids = tuple(int(g) for g in geom_ids)
        if len(self.ids) != PRISM_COUNT:
            raise ValueError("Expected 18 independent triangular prism geoms")
        self.meta = []
        used = set()
        for k, gid in enumerate(self.ids):
            mesh_id = int(mj_model.geom_dataid[gid])
            if mesh_id in used or mesh_id < 0:
                raise ValueError("Each prism must have a distinct mesh dataid")
            used.add(mesh_id)
            src = self.templates[mesh_id]
            if src is None:
                raise ValueError("MuJoCo did not compile convex collision mesh")
            verts = np.asarray(src.vert)
            faces = np.asarray(src.face)
            if verts.shape != (6, 3) or faces.shape[:2] != (5, 4):
                raise ValueError(
                    f"Prism {k} changed convex topology: {verts.shape}/{faces.shape}")
            # The compiler may recenter/rotate mesh assets for inertia. World
            # XY/Z of each vertex must use compiled geom pose, not raw STL XY.
            R = _quat_matrix(mj_model.geom_quat[gid])
            geom_pos = np.asarray(mj_model.geom_pos[gid], dtype=np.float32)
            physical = verts @ R.T + geom_pos
            xys = physical[:,:2]
            expected = np.asarray(_local_triangle(k%2),dtype=np.float32)
            for xy in expected:
                if np.min(np.linalg.norm(xys-xy,axis=1)) > 1e-3:
                    raise ValueError(f"Prism {k} XY changed under MuJoCo compiler")
            top_z = float(np.max(physical[:,2]))
            if float(np.max(physical[:,2])-np.min(physical[:,2])) < PRISM_DEPTH_M-.01:
                raise ValueError("Prism must have substantial buried vertical sides")
            # Faces and edge normals in ConvexMesh are already materialized,
            # not indexed. Recover immutable index maps ONCE before JIT.
            face_index = np.linalg.norm(
                faces[:,:,None,:]-verts[None,None,:,:],axis=-1).argmin(axis=-1)
            if np.max(np.min(np.linalg.norm(
                faces[:,:,None,:]-verts[None,None,:,:],axis=-1),axis=-1))>1e-4:
                raise ValueError("Unable to recover triangular prism face indices")
            normals = np.asarray(src.face_normal)
            edge_normals = np.asarray(src.edge_face_normal)
            normal_index = np.linalg.norm(
                edge_normals[:,:,None,:]-normals[None,None,:,:],axis=-1).argmin(axis=-1)
            if np.max(np.min(np.linalg.norm(
                edge_normals[:,:,None,:]-normals[None,None,:,:],axis=-1),axis=-1))>1e-4:
                raise ValueError("Unable to recover prism edge normal indices")
            self.meta.append(dict(
                mesh_id=mesh_id, top_z=top_z,
                xy=xys, up=R.T@np.array([0.,0.,1.],dtype=np.float32),
                face_index=face_index.astype(np.int32),
                normal_index=normal_index.astype(np.int32)))
        if len(used)!=PRISM_COUNT:
            raise ValueError("Prism assets shared a compiled mesh")

    def batch_template(self, jnp, worlds):
        meshes = list(self.templates)
        for meta in self.meta:
            m = meshes[meta['mesh_id']]
            meshes[meta['mesh_id']] = m.replace(
                vert=jnp.broadcast_to(m.vert,(worlds,)+m.vert.shape),
                face=jnp.broadcast_to(m.face,(worlds,)+m.face.shape),
                face_normal=jnp.broadcast_to(
                    m.face_normal,(worlds,)+m.face_normal.shape),
                edge_face_normal=jnp.broadcast_to(
                    m.edge_face_normal,(worlds,)+m.edge_face_normal.shape))
        return tuple(meshes)

    def in_axes(self, template):
        """JAX vmap spec: only dynamically deformed mesh fields are batched."""
        result = []
        for dataid, m in enumerate(template._impl.mesh_convex):
            if m is None:
                result.append(None)
            elif any(meta['mesh_id']==dataid for meta in self.meta):
                result.append(m.replace(
                    vert=0,face=0,face_normal=0,edge=None,edge_face_normal=0))
            else:
                result.append(m.replace(
                    vert=None,face=None,face_normal=None,
                    edge=None,edge_face_normal=None))
        return tuple(result)

    def deform(self, jnp, heights, base_meshes):
        """Per-world 4-corner normalized hfield samples -> 18 exact solids.

        heights [world, 36] uses (a,b,c,d) per square and the same diagonal
        as sample_surface. Shapes and topology never change after initial JIT.
        """
        world_count=heights.shape[0]
        h=heights.reshape((world_count,TILE_COUNT,4))
        meshes=list(base_meshes)
        for k,meta in enumerate(self.meta):
            tile=k//2
            a,b,c,d=(h[:,tile,i]*TILE_SIZE_Z_M-.5*TILE_SIZE_Z_M
                     for i in range(4))
            if k%2==0:
                gx=(b-a)/TILE_M
                gy=(c-a)/TILE_M
                z_anchor=a
                ax=-.5*TILE_M
                ay=-.5*TILE_M
            else:
                gx=(d-c)/TILE_M
                gy=(d-b)/TILE_M
                z_anchor=d
                ax=.5*TILE_M
                ay=.5*TILE_M
            xy=jnp.asarray(meta['xy'],dtype=jnp.float32)
            goal=(z_anchor[:,None]+
                  gx[:,None]*(xy[None,:,0]-ax)+
                  gy[:,None]*(xy[None,:,1]-ay))
            delta=goal-jnp.asarray(meta['top_z'],dtype=jnp.float32)
            source=self.templates[meta['mesh_id']]
            normal_up=jnp.asarray(meta['up'],dtype=jnp.float32)
            verts=(source.vert[None,:,:]+
                   delta[:,:,None]*normal_up[None,None,:])
            fi=jnp.asarray(meta['face_index'],dtype=jnp.int32)
            faces=verts[:,fi,:]
            v1=faces[:,:,1,:]-faces[:,:,0,:]
            v2=faces[:,:,-1,:]-faces[:,:,0,:]
            ns=jnp.cross(v1,v2)
            ns=ns/jnp.maximum(jnp.linalg.norm(ns,axis=-1,keepdims=True),1e-8)
            ni=jnp.asarray(meta['normal_index'],dtype=jnp.int32)
            updated=source.replace(
                vert=verts,face=faces,face_normal=ns,
                edge_face_normal=ns[:,ni,:])
            meshes[meta['mesh_id']]=updated
        return tuple(meshes)
