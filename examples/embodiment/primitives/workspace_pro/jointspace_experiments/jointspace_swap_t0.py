"""Joint-space IK + PD torque control to open the relocated middle drawer (goal_swap_t0),
bypassing the OSC operational-space singularity. Physics-only (real torques, real contact)."""
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

def ik(target_pos, target_R, q0, iters=400, tol=8e-4):
    save_q=d.qpos.copy(); save_v=d.qvel.copy()
    q=q0.copy(); pe=we=9
    for it in range(iters):
        for jj,a in enumerate(arm_q): d.qpos[a]=q[jj]
        mujoco.mj_forward(m,d)
        p=d.site_xpos[site].copy(); Rc=d.site_xmat[site].reshape(3,3).copy()
        perr=target_pos-p; werr=R.from_matrix(target_R@Rc.T).as_rotvec()
        pe=np.linalg.norm(perr); we=np.linalg.norm(werr)
        if pe<tol and we<0.04: break
        jacp=np.zeros((3,m.nv)); jacr=np.zeros((3,m.nv))
        mujoco.mj_jacSite(m,d,jacp,jacr,site)
        J=np.concatenate([jacp[:,arm_dof],jacr[:,arm_dof]],0)
        err=np.concatenate([perr,werr*0.5]); lam=0.08
        dq=J.T@np.linalg.solve(J@J.T+lam**2*np.eye(6),err)
        q=q+np.clip(dq,-0.2,0.2)
        for jj in range(7): q[jj]=np.clip(q[jj],jlim[jj][0],jlim[jj][1])
    d.qpos[:]=save_q; d.qvel[:]=save_v; mujoco.mj_forward(m,d)
    return q,pe,we

Kp=np.array([150,150,150,150,70,50,25.]); Kd=2*np.sqrt(Kp)*0.7
fmax=np.array([80,80,80,80,80,12,12.])
def drive_to(qt, steps, grip):
    for s in range(steps):
        q=np.array([d.qpos[a] for a in arm_q]); qd=np.array([d.qvel[arm_dof[i]] for i in range(7)])
        tau=np.clip(Kp*(qt-q)-Kd*qd+d.qfrc_bias[arm_dof],-fmax,fmax)
        d.ctrl[0:7]=tau; d.ctrl[7]=grip[0]; d.ctrl[8]=grip[1]
        mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d); return d.site_xpos[site].copy()

OPEN=(0.04,-0.04); CLOSE=(0.0,0.0)
handle=np.array([-0.247,-0.152,1.015])
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
Rpitch=R.from_euler('x',-0.5).as_matrix()@Rdown
q_home=np.array([d.qpos[a] for a in arm_q])

# Phase 1: pre-grab (above+front of handle)
pre=handle+np.array([0,0.04,0.05])
q1,e1,_=ik(pre,Rpitch,q_home); print("IK pre  err=%.4f"%e1)
p=drive_to(q1,500,OPEN); print(" pre  eef",np.round(p,4),"target",np.round(pre,4))
# Phase 2: grab pose (at handle)
q2,e2,_=ik(handle,Rpitch,np.array([d.qpos[a] for a in arm_q])); print("IK grab err=%.4f"%e2)
p=drive_to(q2,500,OPEN); print(" grab eef",np.round(p,4)," err=%.4f"%np.linalg.norm(p-handle))
# Phase 3: close
p=drive_to(np.array([d.qpos[a] for a in arm_q]),250,CLOSE)
g=[d.qpos[env.sim.model.get_joint_qpos_addr(f"gripper0_finger_joint{i}")] for i in (1,2)]
print(" after close: grip=",np.round(g,4)," middle qpos=",round(float(d.qpos[jadr]),4))
# Phase 4: pull +y
pull=handle+np.array([0,0.20,0.02])
q3,e3,_=ik(pull,Rpitch,np.array([d.qpos[a] for a in arm_q])); print("IK pull err=%.4f"%e3)
p=drive_to(q3,700,CLOSE); print(" pull eef",np.round(p,4))
print(" middle qpos=",round(float(d.qpos[jadr]),4)," (Open needs < -0.14)")
print(" check_success:", env.check_success())
env.close()
