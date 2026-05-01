"""Lightweight MJCF → USD converter using mujoco + usd-core (no Isaac Sim needed)."""
import mujoco
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import numpy as np

model = mujoco.MjModel.from_xml_path("models/spot_scene.xml")
data = mujoco.MjData(model)
mujoco.mj_kinematics(model, data)

stage = Usd.Stage.CreateNew("models/spot_scene.usd")
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/Spot")
body_paths = {0: "/World/Spot"}

# Bodies
for i in range(model.nbody):
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
    bname = bname.replace(" ", "_").replace("-", "_").replace(".", "_")
    if i == 0:
        path = "/World/Spot"
    else:
        pid = model.body_parentid[i]
        parent_path = body_paths.get(pid, "/World/Spot")
        path = f"{parent_path}/{bname}"
    body_paths[i] = path
    xf = UsdGeom.Xform.Define(stage, path)
    pos = model.body_pos[i]
    quat = model.body_quat[i]
    xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    xf.AddOrientOp().Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))

# Geoms
for g in range(model.ngeom):
    gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
    gname = gname.replace(" ", "_").replace("-", "_").replace(".", "_")
    body_id = model.geom_bodyid[g]
    bp = body_paths.get(body_id, "/World/Spot")
    gp = f"{bp}/{gname}"
    gtype = model.geom_type[g]

    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[g]
        p = UsdGeom.Mesh.Define(stage, gp)
        vs = model.mesh_vertadr[mid]
        vn = model.mesh_vertnum[mid]
        fs = model.mesh_faceadr[mid]
        fn = model.mesh_facenum[mid]
        verts = model.mesh_vert[vs : vs + vn]
        faces = model.mesh_face[fs : fs + fn]
        p.GetPointsAttr().Set([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
        p.GetFaceVertexCountsAttr().Set([3] * fn)
        p.GetFaceVertexIndicesAttr().Set([int(x) for x in faces.flatten()])
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        p = UsdGeom.Cube.Define(stage, gp)
        p.GetSizeAttr().Set(2.0)
        sz = model.geom_size[g]
        UsdGeom.Xformable(p).AddScaleOp().Set(Gf.Vec3f(float(sz[0]), float(sz[1]), float(sz[2])))
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        p = UsdGeom.Sphere.Define(stage, gp)
        p.GetRadiusAttr().Set(float(model.geom_size[g][0]))
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        p = UsdGeom.Capsule.Define(stage, gp)
        p.GetRadiusAttr().Set(float(model.geom_size[g][0]))
        p.GetHeightAttr().Set(float(model.geom_size[g][1]) * 2)
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        p = UsdGeom.Cylinder.Define(stage, gp)
        p.GetRadiusAttr().Set(float(model.geom_size[g][0]))
        p.GetHeightAttr().Set(float(model.geom_size[g][1]) * 2)
    elif gtype == mujoco.mjtGeom.mjGEOM_PLANE:
        p = UsdGeom.Mesh.Define(stage, gp)
        s = 10.0
        p.GetPointsAttr().Set([Gf.Vec3f(-s,-s,0), Gf.Vec3f(s,-s,0), Gf.Vec3f(s,s,0), Gf.Vec3f(-s,s,0)])
        p.GetFaceVertexCountsAttr().Set([4])
        p.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    else:
        continue

    gpos = model.geom_pos[g]
    gquat = model.geom_quat[g]
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(gp))
    xf.AddTranslateOp(opSuffix="local").Set(Gf.Vec3d(float(gpos[0]), float(gpos[1]), float(gpos[2])))
    xf.AddOrientOp(opSuffix="local").Set(Gf.Quatf(float(gquat[0]), float(gquat[1]), float(gquat[2]), float(gquat[3])))

# Joints
for j in range(model.njnt):
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
    jname = jname.replace(" ", "_").replace("-", "_").replace(".", "_")
    body_id = model.jnt_bodyid[j]
    jp = f"{body_paths.get(body_id, '/World/Spot')}/{jname}"
    jtype = model.jnt_type[j]

    if jtype == mujoco.mjtJoint.mjJNT_HINGE:
        rev = UsdPhysics.RevoluteJoint.Define(stage, jp)
        ax = model.jnt_axis[j]
        if abs(ax[0]) > 0.5:
            rev.GetAxisAttr().Set("X")
        elif abs(ax[1]) > 0.5:
            rev.GetAxisAttr().Set("Y")
        else:
            rev.GetAxisAttr().Set("Z")
        if model.jnt_limited[j]:
            lo = float(np.degrees(model.jnt_range[j][0]))
            hi = float(np.degrees(model.jnt_range[j][1]))
            rev.GetLowerLimitAttr().Set(lo)
            rev.GetUpperLimitAttr().Set(hi)

stage.GetRootLayer().Save()
print(f"Done: models/spot_scene.usd")
print(f"  Bodies: {model.nbody}, Geoms: {model.ngeom}, Joints: {model.njnt}, Meshes: {model.nmesh}")
