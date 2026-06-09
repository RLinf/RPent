"""swap_t0: RAM the wine bottle out of the way, then push the gripper DEEP until the
fingertips are behind the handle bar, close, and PULL (+y hook). Iterate depth until it
catches and the drawer opens. Renders video. Goal = Open(middle drawer)."""
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
Kp=np.array([350,350,350,350,180,110,60.]); Kd=2*np.sqrt(Kp); FMAX=np.array([80,80,80,80,80,12,12.])
def qnow(): return np.array([d.qpos[a] for a in AQ])
def goto(qg,steps,grip,settle=150,rec=True):
    q0=qnow()
    for s in range(steps):
        a=(s+1)/steps; qt=q0+(qg-q0)*a; q=qnow(); qd=np.array([d.qvel[AD[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qt-q)-Kd*qd+d.qfrc_bias[AD],-FMAX,FMAX); d.ctrl[7]=grip; d.ctrl[8]=-grip; mujoco.mj_step(m,d)
        if rec and s%6==0: snap()
    for s in range(settle):
        q=qnow(); qd=np.array([d.qvel[AD[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qg-q)-Kd*qd+d.qfrc_bias[AD],-FMAX,FMAX); d.ctrl[7]=grip; d.ctrl[8]=-grip; mujoco.mj_step(m,d)
        if rec and s%6==0: snap()
    mujoco.mj_forward(m,d); return d.site_xpos[site].copy()
def tipy(): return (d.xpos[tip1][1]+d.xpos[tip2][1])/2
Rg=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)   # gripper points -y, vertical finger-sep
q0=qnow()
for _ in range(8): snap()
# ===== RAM the bottle out of the way (closed gripper, sweep through it sideways) =====
bot=d.xpos[bottleB].copy(); print("bottle start",np.round(bot,3))
goto(ik(np.array([bot[0]+0.07,bot[1],0.97]),Rg,qnow()),300,0.0,settle=10)   # beside bottle (+x)
goto(ik(np.array([bot[0]-0.12,bot[1]+0.02,0.97]),Rg,qnow()),300,0.0,settle=30) # sweep -x through it
print("bottle after ram",np.round(d.xpos[bottleB],3))
# ===== iterate: push DEEP so fingertips go behind the bar, close, pull =====
solved=False
for attempt,depth in enumerate([-0.085,-0.10,-0.115,-0.13]):
    goto(ik(np.array([HANDLE[0],-0.02,HANDLE[2]+0.0]),Rg,qnow()),300,0.04,settle=20)  # in front, open
    # push deep toward the bar (high force); IK target deep, drive hard
    qd_=ik(np.array([HANDLE[0],depth,HANDLE[2]]),Rg,qnow())
    goto(qd_,700,0.04,settle=350)
    ty=tipy(); ey=d.site_xpos[site][1]
    print(f"attempt{attempt} depth_target={depth}: eef_y={ey:.3f} fingertip_y={ty:.3f}  bar_y={HANDLE[1]} (want fingertip < bar)")
    goto(qnow(),300,0.0,settle=0)   # close
    gw=d.qpos[gf[0]]-d.qpos[gf[1]]
    # pull +y slowly
    cur=d.site_xpos[site].copy()
    for k in range(1,12):
        qk=ik(cur+np.array([0,0.014*k,0.0]),Rg,qnow()); goto(qk,70,0.0,settle=20)
    qp=float(d.qpos[jadr]); ok=env.check_success()
    print(f"   grip_w={gw:.4f}  -> middle qpos={qp:.4f}  check_success={ok}")
    if ok or qp<-0.14: solved=True; break
    # retreat for next attempt
    goto(ik(np.array([-0.18,0.05,1.12]),Rg,qnow()),300,0.04,settle=20)
for _ in range(15): snap()
imageio.mimsave(f"{OUT}/swap_t0_HOOK.mp4", frames, fps=30, quality=8); rend.close()
print(f"\nSOLVED={solved}  FINAL middle qpos={round(float(d.qpos[jadr]),4)}  CHECK_SUCCESS={env.check_success()}  ({len(frames)} frames -> swap_t0_HOOK.mp4)")
env.close()
