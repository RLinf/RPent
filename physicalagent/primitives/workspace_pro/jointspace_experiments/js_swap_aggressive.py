"""Task-completion-first: clear the wine bottle, then grasp the relocated middle handle
(gripper pointing -y, vertical pinch precisely on the bar) and pull the drawer open SLOWLY
(friction-held). Goal = Open(middle drawer) only; collateral (knocked bottle) is fine. Renders video."""
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
pad1=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"gripper0_finger1_pad_collision")
pad2=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"gripper0_finger2_pad_collision")
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
Kp=np.array([320,320,320,320,160,100,55.]); Kd=2*np.sqrt(Kp); FMAX=np.array([80,80,80,80,80,12,12.])
def qnow(): return np.array([d.qpos[a] for a in AQ])
def goto(qg,steps,grip,settle=200):
    q0=qnow()
    for s in range(steps):
        a=(s+1)/steps; qt=q0+(qg-q0)*a; q=qnow(); qd=np.array([d.qvel[AD[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qt-q)-Kd*qd+d.qfrc_bias[AD],-FMAX,FMAX); d.ctrl[7]=grip; d.ctrl[8]=-grip; mujoco.mj_step(m,d)
        if s%6==0: snap()
    for s in range(settle):
        q=qnow(); qd=np.array([d.qvel[AD[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qg-q)-Kd*qd+d.qfrc_bias[AD],-FMAX,FMAX); d.ctrl[7]=grip; d.ctrl[8]=-grip; mujoco.mj_step(m,d)
        if s%6==0: snap()
    mujoco.mj_forward(m,d); return d.site_xpos[site].copy()
Rg=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)  # gripper points -y, finger-sep vertical (z)
q0=qnow()
for _ in range(10): snap()
# PHASE 0: knock the wine bottle aside (drive closed gripper through it, sweep +y/+x away from cabinet)
bottle=np.array([-0.196,-0.064,0.96])
goto(ik(bottle+np.array([0,0.0,0.06]),Rg,qnow()),350,0.0,settle=20)   # above bottle, closed
goto(ik(bottle+np.array([0.06,0.10,0.0]),Rg,qnow()),350,0.0,settle=40) # shove it +x/+y (away)
bp=d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"wine_bottle_1_main")]
print("bottle after knock:",np.round(bp,3))
# PHASE 1: find eef pose so vertical pads straddle the bar (z) and sit ON the bar (y)
best=None
for ey in [-0.10,-0.09,-0.08,-0.07]:
    q=ik(np.array([HANDLE[0],ey,HANDLE[2]]),Rg,qnow())
    setq(q,0.04); mujoco.mj_forward(m,d)
    p1=d.geom_xpos[pad1].copy(); p2=d.geom_xpos[pad2].copy()
    zstr=min(p1[2],p2[2])<HANDLE[2]<max(p1[2],p2[2]); ony=abs((p1[1]+p2[1])/2-HANDLE[1])<0.03
    if best is None and zstr and ony: best=(q,ey)
    setq(q0); mujoco.mj_forward(m,d)
if best is None: best=(ik(np.array([HANDLE[0],-0.08,HANDLE[2]]),Rg,qnow()),-0.08)
qg,ey=best; print("grasp eef_y target",ey)
goto(qg,800,0.04,settle=400)
print("reached eef",np.round(d.site_xpos[site],4))
# PHASE 2: close HARD
goto(qnow(),400,0.0,settle=0)
print("grip width",round(d.qpos[gf[0]]-d.qpos[gf[1]],4))
# PHASE 3: pull +y VERY slowly (friction-held), small increments
cur=d.site_xpos[site].copy()
for k in range(1,13):
    qk=ik(cur+np.array([0,0.015*k,0.0]),Rg,qnow()); goto(qk,90,0.0,settle=25)
    if k%4==0: print(f"  pull {k}: middle qpos={float(d.qpos[jadr]):.4f}")
for _ in range(15): snap()
ok=env.check_success(); qp=round(float(d.qpos[jadr]),4)
imageio.mimsave(f"{OUT}/swap_t0_AGGRESSIVE.mp4", frames, fps=30, quality=8)
rend.close()
print(f"\nFINAL middle qpos={qp}  CHECK_SUCCESS={ok}  ({len(frames)} frames -> swap_t0_AGGRESSIVE.mp4)")
env.close()
