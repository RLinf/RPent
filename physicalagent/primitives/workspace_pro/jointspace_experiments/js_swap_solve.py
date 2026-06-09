"""swap_t0 determined solver: knock the bottle away (verified, retry), then push the gripper
DEEP so the fingertips go BEHIND the handle bar, close, and PULL +y (hook). Iterate many times
over depth/height until the middle drawer opens (qpos<-0.14). Renders the run."""
import os, numpy as np, mujoco, imageio
os.environ.setdefault("MUJOCO_GL","egl"); os.environ["LIBERO_TYPE"]="pro"
import liberopro.liberopro.benchmark as bench
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation as R
OUT="/mnt/public/jxqiu/physicalagent/physicalagent/primitives/result_paper/goal_fail_renders"
b=bench.get_benchmark("libero_goal_swap")()
env=OffScreenRenderEnv(bddl_file_name=b.get_task_bddl_file_path(0),camera_heights=128,camera_widths=128)
env.seed(0); env.reset(); env.set_init_state(b.get_task_init_states(0)[0])
for _ in range(5): env.step(np.zeros(7))
m,d=env.sim.model._model, env.sim.data._data
site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"gripper0_grip_site")
AQ=[env.sim.model.get_joint_qpos_addr(f"robot0_joint{i}") for i in range(1,8)]
AD=[m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)]
JL=np.array([m.jnt_range[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)])
jadr=env.sim.model.get_joint_qpos_addr("wooden_cabinet_1_middle_level")
gf=[env.sim.model.get_joint_qpos_addr(f"gripper0_finger_joint{i}") for i in (1,2)]
tip1=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"gripper0_finger_joint1_tip")
tip2=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"gripper0_finger_joint2_tip")
bottleB=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"wine_bottle_1_main")
HANDLE=np.array([-0.247,-0.155,1.015])
rend=mujoco.Renderer(m,480,480); frames=[]
def snap(): rend.update_scene(d,camera="frontview"); frames.append(rend.render().copy())
def setq(q,grip=0.04):
    for jj,a in enumerate(AQ): d.qpos[a]=q[jj]
    d.qpos[gf[0]]=grip; d.qpos[gf[1]]=-grip
def ik(tp,tR,q0,it=700):
    sq=d.qpos.copy(); sv=d.qvel.copy(); q=q0.copy()
    for _ in range(it):
        setq(q); mujoco.mj_forward(m,d)
        p=d.site_xpos[site].copy(); Rc=d.site_xmat[site].reshape(3,3).copy()
        perr=tp-p; werr=R.from_matrix(tR@Rc.T).as_rotvec()
        if np.linalg.norm(perr)<5e-4 and np.linalg.norm(werr)<0.02: break
        jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,jr,site)
        J=np.concatenate([jp[:,AD],jr[:,AD]],0); err=np.concatenate([perr,werr])
        dq=J.T@np.linalg.solve(J@J.T+0.05**2*np.eye(6),err); q=np.clip(q+np.clip(dq,-0.15,0.15),JL[:,0],JL[:,1])
    d.qpos[:]=sq; d.qvel[:]=sv; mujoco.mj_forward(m,d); return q
Kp=np.array([360,360,360,360,190,120,65.]); Kd=2*np.sqrt(Kp); FMAX=np.array([80,80,80,80,80,12,12.])
def qnow(): return np.array([d.qpos[a] for a in AQ])
def goto(qg,steps,grip,settle=120,rec=True):
    q0=qnow()
    for s in range(steps):
        a=(s+1)/steps; qt=q0+(qg-q0)*a; q=qnow(); qd=np.array([d.qvel[AD[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qt-q)-Kd*qd+d.qfrc_bias[AD],-FMAX,FMAX); d.ctrl[7]=grip; d.ctrl[8]=-grip; mujoco.mj_step(m,d)
        if rec and s%7==0: snap()
    for s in range(settle):
        q=qnow(); qd=np.array([d.qvel[AD[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qg-q)-Kd*qd+d.qfrc_bias[AD],-FMAX,FMAX); d.ctrl[7]=grip; d.ctrl[8]=-grip; mujoco.mj_step(m,d)
        if rec and s%7==0: snap()
    mujoco.mj_forward(m,d); return d.site_xpos[site].copy()
def tipy(): return (d.xpos[tip1][1]+d.xpos[tip2][1])/2
Rg=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)   # gripper points -y, vertical finger-sep
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
q0=qnow()
for _ in range(8): snap()
# ===== KNOCK the bottle (verified, retry) =====
bot0=d.xpos[bottleB].copy(); print("bottle start",np.round(bot0,3))
def knocked(): return np.linalg.norm(d.xpos[bottleB][:2]-bot0[:2])>0.06 or d.xpos[bottleB][2]<bot0[2]-0.05
for kx,(zc,sweep) in enumerate([(1.00,[0,0.18]),(0.96,[0,0.18]),(1.02,[0.0,0.20]),(0.95,[0.15,0.05])]):
    goto(ik(np.array([bot0[0],bot0[1]+0.02,zc+0.06]),Rdown,qnow()),250,1.0,settle=10)  # above bottle, closed
    goto(ik(np.array([bot0[0],bot0[1],zc]),Rdown,qnow()),250,1.0,settle=10)             # onto bottle body
    goto(ik(np.array([bot0[0]+sweep[0],bot0[1]+sweep[1],zc]),Rdown,qnow()),300,1.0,settle=30)  # sweep
    print(f"  knock try {kx}: bottle now {np.round(d.xpos[bottleB],3)} knocked={knocked()}")
    if knocked(): break
    goto(ik(np.array([0.0,0.05,1.15]),Rdown,qnow()),250,1.0,settle=10)  # retreat for retry
# ===== iterate hook grab+pull =====
solved=False; att=0
for depth in [-0.085,-0.10,-0.12,-0.14,-0.16,-0.18]:
  for zc in [1.015,1.005,1.025]:
    att+=1
    goto(ik(np.array([HANDLE[0],0.0,zc]),Rg,qnow()),250,0.04,settle=15)   # in front, open
    goto(ik(np.array([HANDLE[0],depth,zc]),Rg,qnow()),650,0.04,settle=300) # push DEEP (high force)
    ty=tipy(); ey=d.site_xpos[site][1]
    goto(qnow(),280,0.0,settle=0)   # close
    gw=d.qpos[gf[0]]-d.qpos[gf[1]]
    cur=d.site_xpos[site].copy()
    for k in range(1,11):
        qk=ik(cur+np.array([0,0.015*k,0.0]),Rg,qnow()); goto(qk,65,0.0,settle=15)
    qp=float(d.qpos[jadr]); ok=env.check_success()
    print(f"att{att} depth={depth} z={zc}: tip_y={ty:.3f}(bar{HANDLE[1]}) grip={gw:.4f} -> midqpos={qp:.4f} success={ok}")
    if ok or qp<-0.14: solved=True; break
    goto(ik(np.array([-0.18,0.05,1.12]),Rg,qnow()),220,0.04,settle=10)  # retreat
  if solved: break
for _ in range(15): snap()
imageio.mimsave(f"{OUT}/swap_t0_SOLVE.mp4", frames, fps=30, quality=8); rend.close()
print(f"\n==== SOLVED={solved} FINAL middle qpos={round(float(d.qpos[jadr]),4)} CHECK_SUCCESS={env.check_success()} ({len(frames)} frames) ====")
env.close()
