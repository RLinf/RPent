"""Test: pure gripper-DOWN front/back pinch with grip_site ~3cm IN FRONT of the bar
(so the back finger sits in the gap behind the bar, body clears the stacked top handle).
Joint-space IK + PD; then close (front/back pinch) + pull +y. goal_swap_t0 middle drawer."""
import os, numpy as np, mujoco
os.environ.setdefault("MUJOCO_GL","egl"); os.environ["LIBERO_TYPE"]="pro"
import liberopro.liberopro.benchmark as bench
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation as R
SUITE=os.environ.get("SUITE","libero_goal_swap"); TASK=int(os.environ.get("TASK","0"))
HANDLE=np.array([float(x) for x in os.environ.get("HANDLE","-0.247,-0.152,1.015").split(",")])
JOINTNAME=os.environ.get("JOINTNAME","wooden_cabinet_1_middle_level")
b=bench.get_benchmark(SUITE)()
env=OffScreenRenderEnv(bddl_file_name=b.get_task_bddl_file_path(TASK),camera_heights=128,camera_widths=128)
env.seed(0); env.reset(); env.set_init_state(b.get_task_init_states(TASK)[0])
for _ in range(5): env.step(np.zeros(7))
m,d=env.sim.model._model, env.sim.data._data
site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"gripper0_grip_site")
AQ=[env.sim.model.get_joint_qpos_addr(f"robot0_joint{i}") for i in range(1,8)]
AD=[m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)]
JL=np.array([m.jnt_range[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)])
jadr=env.sim.model.get_joint_qpos_addr(JOINTNAME)
gf=[env.sim.model.get_joint_qpos_addr(f"gripper0_finger_joint{i}") for i in (1,2)]
def setq(q,grip=0.04):
    for jj,a in enumerate(AQ): d.qpos[a]=q[jj]
    d.qpos[gf[0]]=grip; d.qpos[gf[1]]=-grip
def ik(tp,tR,q0,it=600):
    sq=d.qpos.copy(); sv=d.qvel.copy(); q=q0.copy(); pe=we=9
    for _ in range(it):
        setq(q); mujoco.mj_forward(m,d)
        p=d.site_xpos[site].copy(); Rc=d.site_xmat[site].reshape(3,3).copy()
        perr=tp-p; werr=R.from_matrix(tR@Rc.T).as_rotvec(); pe=np.linalg.norm(perr); we=np.linalg.norm(werr)
        if pe<5e-4 and we<0.012: break
        jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,jr,site)
        J=np.concatenate([jp[:,AD],jr[:,AD]],0); err=np.concatenate([perr,werr])
        dq=J.T@np.linalg.solve(J@J.T+0.05**2*np.eye(6),err); q=np.clip(q+np.clip(dq,-0.15,0.15),JL[:,0],JL[:,1])
    d.qpos[:]=sq; d.qvel[:]=sv; mujoco.mj_forward(m,d); return q,pe,we
def contacts(q,grip=0.04):
    sq=d.qpos.copy(); setq(q,grip); mujoco.mj_forward(m,d); n=0
    for ci in range(d.ncon):
        c=d.contact[ci]
        if c.dist>-0.003: continue
        n1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom1]) or ""; n2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom2]) or ""
        if ("gripper" in n1 or "robot0" in n1 or "gripper" in n2 or "robot0" in n2) and ("cabinet" in n1 or "cabinet" in n2): n+=1
    d.qpos[:]=sq; mujoco.mj_forward(m,d); return n
Kp=np.array([300,300,300,300,150,90,50.]); Kd=2*np.sqrt(Kp); FMAX=np.array([80,80,80,80,80,12,12.])
def qnow(): return np.array([d.qpos[a] for a in AQ])
def goto(qg,steps,grip,settle=300):
    q0=qnow()
    for s in range(steps):
        a=(s+1)/steps; qt=q0+(qg-q0)*a; q=qnow(); qd=np.array([d.qvel[AD[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qt-q)-Kd*qd+d.qfrc_bias[AD],-FMAX,FMAX); d.ctrl[7]=grip; d.ctrl[8]=-grip; mujoco.mj_step(m,d)
    for _ in range(settle):
        q=qnow(); qd=np.array([d.qvel[AD[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qg-q)-Kd*qd+d.qfrc_bias[AD],-FMAX,FMAX); d.ctrl[7]=grip; d.ctrl[8]=-grip; mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d); return d.site_xpos[site].copy()
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
Rgd=Rdown@R.from_rotvec([0,0,np.pi/2]).as_matrix()  # gripper-down, finger-sep -> world-Y (front/back)
qhome=qnow()
print("-- gripper-DOWN front/back pinch, grip_site in front of bar (back finger in gap) --")
best=None
for dy in [0.02,0.025,0.03,0.035,0.04,0.05]:
    q,e,w=ik(HANDLE+np.array([0,dy,0.0]),Rgd,qhome); c=contacts(q)
    # finger world y positions (open)
    setq(q,0.04); mujoco.mj_forward(m,d)
    f1=d.geom_xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"gripper0_finger1_pad_collision")].copy()
    f2=d.geom_xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"gripper0_finger2_pad_collision")].copy()
    setq(qhome); mujoco.mj_forward(m,d)
    straddle = min(f1[1],f2[1])< HANDLE[1] <max(f1[1],f2[1])
    print(f"  dy={dy:.3f} grip_site_y={HANDLE[1]+dy:+.3f} ik_err={e:.4f} contacts={c} padY=({f1[1]:.3f},{f2[1]:.3f}) bar_between={straddle}")
    if best is None and straddle and abs(dy-0.03)<1e-6: best=(q,dy)
if best is None: print("no straddle config"); env.close(); raise SystemExit
qg,dy=best; print("chosen dy",dy)
p=goto(qg,900,0.04,settle=400)
print("reached eef",np.round(d.site_xpos[site],4),"(bar",HANDLE,") padcontacts_now",sum(1 for ci in range(d.ncon) if d.contact[ci].dist<-0.001))
goto(qnow(),350,0.0,settle=0)
print("close grip width",round(d.qpos[gf[0]]-d.qpos[gf[1]],4)," (bar~0.015 if grabbed)")
# pull +y incremental, gripper closed
cur=d.site_xpos[site].copy()
for k in range(1,11):
    qk,_,_=ik(cur+np.array([0,0.02*k,0.0]),Rgd,qnow()); goto(qk,120,0.0,settle=20)
print("FINAL",JOINTNAME,"qpos",round(float(d.qpos[jadr]),4)," CHECK_SUCCESS:",env.check_success())
env.close()
