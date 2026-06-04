import os
os.environ.setdefault("MUJOCO_GL","egl"); os.environ["LIBERO_TYPE"]="pro"
import numpy as np, mujoco
import liberopro.liberopro.benchmark as bench
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation as R
b=bench.get_benchmark("libero_goal_swap")()
env=OffScreenRenderEnv(bddl_file_name=b.get_task_bddl_file_path(0),camera_heights=128,camera_widths=128)
env.seed(0); env.reset(); env.set_init_state(b.get_task_init_states(0)[0])
for _ in range(5): env.step(np.zeros(7))
m,d=env.sim.model._model, env.sim.data._data
site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"gripper0_grip_site")
arm_q=[env.sim.model.get_joint_qpos_addr(f"robot0_joint{i}") for i in range(1,8)]
arm_dof=[m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)]
jlim=[m.jnt_range[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)]
jadr=env.sim.model.get_joint_qpos_addr("wooden_cabinet_1_middle_level")
gf=[env.sim.model.get_joint_qpos_addr(f"gripper0_finger_joint{i}") for i in (1,2)]
def ik(tp,tR,q0,iters=600):
    sq=d.qpos.copy(); sv=d.qvel.copy(); q=q0.copy()
    for it in range(iters):
        for jj,a in enumerate(arm_q): d.qpos[a]=q[jj]
        mujoco.mj_forward(m,d)
        p=d.site_xpos[site].copy(); Rc=d.site_xmat[site].reshape(3,3).copy()
        perr=tp-p; werr=R.from_matrix(tR@Rc.T).as_rotvec()
        if np.linalg.norm(perr)<5e-4 and np.linalg.norm(werr)<0.012: break
        jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,jr,site)
        J=np.concatenate([jp[:,arm_dof],jr[:,arm_dof]],0); err=np.concatenate([perr,werr])
        dq=J.T@np.linalg.solve(J@J.T+0.05**2*np.eye(6),err); q=q+np.clip(dq,-0.15,0.15)
        for jj in range(7): q[jj]=np.clip(q[jj],jlim[jj][0],jlim[jj][1])
    d.qpos[:]=sq; d.qvel[:]=sv; mujoco.mj_forward(m,d); return q
Kp=np.array([300,300,300,300,150,90,50.]); Kd=2*np.sqrt(Kp)*1.0; fmax=np.array([80,80,80,80,80,12,12.])
def qnow(): return np.array([d.qpos[a] for a in arm_q])
def track(qgoal, steps, grip, settle=250):
    q0=qnow()
    for s in range(steps):
        a=(s+1)/steps; qt=q0+(qgoal-q0)*a; q=qnow(); qd=np.array([d.qvel[arm_dof[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qt-q)-Kd*qd+d.qfrc_bias[arm_dof],-fmax,fmax); d.ctrl[7]=grip[0]; d.ctrl[8]=grip[1]; mujoco.mj_step(m,d)
    for s in range(settle):
        q=qnow(); qd=np.array([d.qvel[arm_dof[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qgoal-q)-Kd*qd+d.qfrc_bias[arm_dof],-fmax,fmax); d.ctrl[7]=grip[0]; d.ctrl[8]=grip[1]; mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d); return d.site_xpos[site].copy()
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
Rgrab=R.from_rotvec([-0.45,0,0]).as_matrix()@Rdown@R.from_rotvec([0,0,np.pi/2]).as_matrix()
# verify finger-sep axis at grab orientation
_sq=d.qpos.copy(); _sv=d.qvel.copy()
qchk=ik(np.array([-0.247,-0.152,1.01]),Rgrab,qnow())
for jj,a in enumerate(arm_q): d.qpos[a]=qchk[jj]
d.qpos[gf[0]]=0.04; d.qpos[gf[1]]=-0.04; mujoco.mj_forward(m,d)
p1=d.geom_xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"gripper0_finger1_pad_collision")]
p2=d.geom_xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"gripper0_finger2_pad_collision")]
print("finger-sep vector (want along world-Y):",np.round(p1-p2,3))
d.qpos[:]=_sq; d.qvel[:]=_sv; mujoco.mj_forward(m,d)
handle=np.array([-0.247,-0.152,1.015]); OPEN=(0.04,-0.04); CLOSE=(0.0,0.0)
q_app=ik(handle+np.array([0,0.09,0.05]),Rgrab,qnow())
q_eng=ik(handle+np.array([0,0.0,0.0]),Rgrab,qnow())
print("approach eef",np.round(track(q_app,500,OPEN),4))
print("engage   eef",np.round(track(q_eng,400,OPEN),4))
track(qnow(),300,CLOSE,settle=0)
print("close grip width",round(d.qpos[gf[0]]-d.qpos[gf[1]],4),"middle qpos",round(float(d.qpos[jadr]),4))
q_pull=ik(handle+np.array([0,0.22,0.05]),Rgrab,qnow())
track(q_pull,700,CLOSE)
# settle: hold final config
for _ in range(300):
    q=qnow(); qd=np.array([d.qvel[arm_dof[i]] for i in range(7)])
    d.ctrl[0:7]=np.clip(Kp*(q_pull-q)-Kd*qd+d.qfrc_bias[arm_dof],-fmax,fmax); d.ctrl[7]=CLOSE[0]; d.ctrl[8]=CLOSE[1]; mujoco.mj_step(m,d)
mujoco.mj_forward(m,d)
ta=env.sim.model.get_joint_qpos_addr("wooden_cabinet_1_top_level")
ba=env.sim.model.get_joint_qpos_addr("wooden_cabinet_1_bottom_level")
print("FINAL drawer qpos: top",round(float(d.qpos[ta]),4),"middle",round(float(d.qpos[jadr]),4),"bottom",round(float(d.qpos[ba]),4),"(Open: <-0.14)")
print("eef",np.round(d.site_xpos[site],4))
print("CHECK_SUCCESS (after drive):",env.check_success())
# sanity: does check work on THIS data object at a clean known-open qpos?
d.qpos[jadr]=-0.15; mujoco.mj_forward(m,d)
print("CHECK after forcing middle=-0.15 + forward:",env.check_success(),"middle qpos",round(float(d.qpos[jadr]),4))
# and re-set to the driven value, forward, check
d.qpos[jadr]=-0.159; mujoco.mj_forward(m,d)
print("CHECK after setting middle=-0.159 + forward:",env.check_success())
# manual predicate eval
try:
    print("obj_of_interest:",env.obj_of_interest)
except Exception as e: print("ooi err",e)
env.close()
