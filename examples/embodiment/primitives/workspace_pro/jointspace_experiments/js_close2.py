"""Engineer a RELIABLE scripted drawer-CLOSE for white_cabinet bottom drawer (task_t3).
Push glances off the thin handle (stalls ~qpos-0.10). Instead GRIP the handle bar (gripper-DOWN
rolled so fingers straddle the bar front/back in y) and TRANSLATE +y to drag it closed. Offline,
verify qpos -> 0. The handle is pulled out into open space so gripper-down has clearance above."""
import os, numpy as np, mujoco, imageio
os.environ.setdefault("MUJOCO_GL","egl"); os.environ["LIBERO_TYPE"]="pro"
import liberopro.liberopro.benchmark as bench
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation as R
b=bench.get_benchmark("libero_10_task")()
env=OffScreenRenderEnv(bddl_file_name=b.get_task_bddl_file_path(3),camera_heights=128,camera_widths=128)
env.seed(0); env.reset(); env.set_init_state(b.get_task_init_states(3)[0])
for _ in range(5): env.step(np.zeros(7))
m,d=env.sim.model._model, env.sim.data._data
site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"gripper0_grip_site")
AQ=[env.sim.model.get_joint_qpos_addr(f"robot0_joint{i}") for i in range(1,8)]
AD=[m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)]
JL=np.array([m.jnt_range[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)])
jadr=env.sim.model.get_joint_qpos_addr("white_cabinet_1_bottom_level")
gf=[env.sim.model.get_joint_qpos_addr(f"gripper0_finger_joint{i}") for i in (1,2)]
hg=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"white_cabinet_1_g40")
hg2=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"white_cabinet_1_g41")
p1g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"gripper0_finger1_pad_collision")
p2g=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"gripper0_finger2_pad_collision")
def setq(q,grip=0.04):
    for jj,a in enumerate(AQ): d.qpos[a]=q[jj]
    d.qpos[gf[0]]=grip; d.qpos[gf[1]]=-grip
def ik(tp,tR,q0,it=1200):
    sq=d.qpos.copy(); sv=d.qvel.copy(); q=q0.copy()
    for _ in range(it):
        setq(q); mujoco.mj_forward(m,d)
        perr=tp-d.site_xpos[site].copy(); werr=R.from_matrix(tR@d.site_xmat[site].reshape(3,3).T).as_rotvec()
        if np.linalg.norm(perr)<4e-4 and np.linalg.norm(werr)<0.02: break
        jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,jr,site)
        J=np.concatenate([jp[:,AD],jr[:,AD]],0); dq=J.T@np.linalg.solve(J@J.T+0.04**2*np.eye(6),np.concatenate([perr,werr]))
        q=np.clip(q+np.clip(dq,-0.1,0.1),JL[:,0],JL[:,1])
    d.qpos[:]=sq; d.qvel[:]=sv; mujoco.mj_forward(m,d); return q
Kp=np.array([400,400,400,400,220,140,80.]); Kd=2*np.sqrt(Kp); FMAX=np.array([90,90,90,90,85,14,14.])
def qnow(): return np.array([d.qpos[a] for a in AQ])
def goto(qg,steps,grip,settle=30):
    q0=qnow()
    for s in range(steps):
        a=(s+1)/steps; qt=q0+(qg-q0)*a; q=qnow(); qd=np.array([d.qvel[AD[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qt-q)-Kd*qd+d.qfrc_bias[AD],-FMAX,FMAX); d.ctrl[7]=grip; d.ctrl[8]=-grip; mujoco.mj_step(m,d)
    for _ in range(settle):
        q=qnow(); qd=np.array([d.qvel[AD[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qg-q)-Kd*qd+d.qfrc_bias[AD],-FMAX,FMAX); d.ctrl[7]=grip; d.ctrl[8]=-grip; mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d)
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
H=(d.geom_xpos[hg]+d.geom_xpos[hg2])/2; print("handle center:",np.round(H,3),"start qpos:",round(float(d.qpos[jadr]),4))
# Close needs qpos>0.005 (fully seated). Push the drawer FACE LOW (z below cabinet top) with the
# closed gripper so it stays in contact down to flush+, TRACKING the drawer front each short push.
qh=np.array([d.qpos[a] for a in AQ])
for PUSHZ in [1.0,0.99,1.01]:
    setq(qh); mujoco.mj_forward(m,d)
    goto(ik(np.array([H[0],-0.05,PUSHZ+0.05]),Rdown,qh),320,0.0,settle=15)
    goto(ik(np.array([H[0],-0.01,PUSHZ]),Rdown,qnow()),260,0.0,settle=15)   # in front of upper face, above handle/below middle drawer
    cur=d.site_xpos[site].copy(); ok=False
    for k in range(1,26):
        goto(ik(cur+np.array([0,0.012*k,0.0]),Rdown,qnow()),65,0.0,settle=10)
        qp=float(d.qpos[jadr])
        if k%4==0: print(f"  z={PUSHZ} push {k}: qpos={qp:.4f} eef_y={d.site_xpos[site][1]:.3f}")
        if qp>0.004: ok=True; break
    print(f"  z={PUSHZ} -> qpos={float(d.qpos[jadr]):.4f}")
    if ok: break
print("FINAL qpos:",round(float(d.qpos[jadr]),4),"check_success:",env.check_success())
env.close()
