"""Properly: (1) pick the wine bottle out of the way, (2) grasp the ACTUAL middle-handle bar
(verify pads on the bar, not the bottle), (3) pull slowly (friction-held). Renders video."""
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
Kp=np.array([320,320,320,320,160,100,55.]); Kd=2*np.sqrt(Kp); FMAX=np.array([80,80,80,80,80,12,12.])
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
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
Rg=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)   # gripper points -y, vertical finger-sep
q0=qnow()
for _ in range(8): snap()
# ===== PHASE 0: PICK the wine bottle and move it away =====
bot=d.xpos[bottleB].copy(); print("bottle start:",np.round(bot,3))
goto(ik(np.array([bot[0],bot[1],1.10]),Rdown,qnow()),300,0.04)       # above bottle, open
goto(ik(np.array([bot[0],bot[1],1.05]),Rdown,qnow()),300,0.04,settle=20)
goto(ik(np.array([bot[0],bot[1],1.00]),Rdown,qnow()),250,0.04,settle=30)  # descend onto body
goto(qnow(),150,1.0,settle=0)                                        # close on bottle
goto(ik(np.array([bot[0],bot[1],1.15]),Rdown,qnow()),300,1.0,settle=40)   # lift
goto(ik(np.array([0.10,0.18,1.15]),Rdown,qnow()),400,1.0,settle=40)       # carry to front-right
goto(qnow(),120,0.04,settle=60)                                      # release
print("bottle after removal:",np.round(d.xpos[bottleB],3))
if abs(d.xpos[bottleB][2]-bot[2])<0.05 and np.linalg.norm(d.xpos[bottleB][:2]-bot[:2])<0.05:
    print("  -> pick FAILED, knocking it over instead")
    goto(ik(np.array([bot[0]+0.10,bot[1],1.00]),Rdown,qnow()),300,0.0,settle=20)
    goto(ik(np.array([bot[0]-0.10,bot[1],1.00]),Rdown,qnow()),350,0.0,settle=40)
    print("  bottle after knock:",np.round(d.xpos[bottleB],3))
goto(ik(np.array([-0.18,0.05,1.15]),Rg,qnow()),300,0.04)             # retreat toward cabinet, switch orient
# ===== PHASE 1: grasp the BAR (pads on bar) — sweep eef_y, verify pad world pos =====
best=None
for ey in [-0.12,-0.11,-0.10,-0.09,-0.08]:
    q=ik(np.array([HANDLE[0],ey,HANDLE[2]]),Rg,qnow())
    setq(q,0.04); mujoco.mj_forward(m,d)
    pc=(d.geom_xpos[pad1].copy()+d.geom_xpos[pad2].copy())/2
    onbar = abs(pc[0]-HANDLE[0])<0.04 and abs(pc[1]-HANDLE[1])<0.025 and abs(pc[2]-HANDLE[2])<0.04
    setq(q0); mujoco.mj_forward(m,d)
    if best is None and onbar: best=(q,ey,pc)
    print(f"   ey_target={ey} -> pad_center={np.round(pc,3)}")
if best is None:
    q=ik(np.array([HANDLE[0],-0.10,HANDLE[2]]),Rg,qnow()); best=(q,-0.10,None)
qg,ey,pc=best; print("grasp eef_y",ey,"pad_center",None if pc is None else np.round(pc,3),"(bar",HANDLE,")")
goto(qg,800,0.04,settle=400)
setq(qnow(),0.04); mujoco.mj_forward(m,d)
print("at grasp: pad_center",np.round((d.geom_xpos[pad1]+d.geom_xpos[pad2])/2,3),"eef",np.round(d.site_xpos[site],3))
goto(qnow(),400,0.0,settle=0)   # close HARD on the bar
print("grip width on bar:",round(d.qpos[gf[0]]-d.qpos[gf[1]],4))
# ===== PHASE 2: pull +y VERY slowly =====
cur=d.site_xpos[site].copy()
for k in range(1,14):
    qk=ik(cur+np.array([0,0.012*k,0.0]),Rg,qnow()); goto(qk,80,0.0,settle=30)
    if k%4==0: print(f"  pull {k}: middle qpos={float(d.qpos[jadr]):.4f}")
for _ in range(15): snap()
ok=env.check_success(); qp=round(float(d.qpos[jadr]),4)
imageio.mimsave(f"{OUT}/swap_t0_CLEARBOTTLE.mp4", frames, fps=30, quality=8); rend.close()
print(f"\nFINAL middle qpos={qp}  CHECK_SUCCESS={ok}  ({len(frames)} frames -> swap_t0_CLEARBOTTLE.mp4)")
env.close()
