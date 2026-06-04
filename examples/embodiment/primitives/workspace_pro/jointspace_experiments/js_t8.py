"""10_swap_t8 SOLVE: 'put both moka pots on the stove'. Stove already ON (button qpos 0.96>=0.5).
Joint-space pick-and-place x2: grasp each moka body (gripper-down, close=0.0), carry, place on the
15cm cook_region (0.215,0.051,~0.93), release. Goal: On(moka1,cook)+On(moka2,cook)+TurnOn (already on)."""
import os, sys, numpy as np, mujoco, imageio
os.environ.setdefault("MUJOCO_GL","egl"); os.environ["LIBERO_TYPE"]="pro"
import liberopro.liberopro.benchmark as bench
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation as R
OUT="/mnt/public2/zhangyixian/RLinf_agentic/examples/embodiment/primitives/result_paper/goal_fail_renders"
b=bench.get_benchmark("libero_10_swap")()
env=OffScreenRenderEnv(bddl_file_name=b.get_task_bddl_file_path(8),camera_heights=128,camera_widths=128)
env.seed(0); env.reset(); env.set_init_state(b.get_task_init_states(8)[0])
for _ in range(5): env.step(np.zeros(7))
m,d=env.sim.model._model, env.sim.data._data
site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"gripper0_grip_site")
AQ=[env.sim.model.get_joint_qpos_addr(f"robot0_joint{i}") for i in range(1,8)]
AD=[m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)]
JL=np.array([m.jnt_range[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)])
gf=[env.sim.model.get_joint_qpos_addr(f"gripper0_finger_joint{i}") for i in (1,2)]
mk1=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"moka_pot_1_main")
mk2=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"moka_pot_2_main")
COOK=np.array([0.215,0.051,0.905])
rend=mujoco.Renderer(m,480,480); frames=[]
def snap(): rend.update_scene(d,camera="frontview"); frames.append(rend.render().copy())
def setq(q,grip=0.04):
    for jj,a in enumerate(AQ): d.qpos[a]=q[jj]
    d.qpos[gf[0]]=grip; d.qpos[gf[1]]=-grip
def ik(tp,tR,q0,it=800):
    sq=d.qpos.copy(); sv=d.qvel.copy(); q=q0.copy()
    for _ in range(it):
        setq(q); mujoco.mj_forward(m,d)
        p=d.site_xpos[site].copy(); Rc=d.site_xmat[site].reshape(3,3).copy()
        perr=tp-p; werr=R.from_matrix(tR@Rc.T).as_rotvec()
        if np.linalg.norm(perr)<5e-4 and np.linalg.norm(werr)<0.02: break
        jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,jr,site)
        J=np.concatenate([jp[:,AD],jr[:,AD]],0); err=np.concatenate([perr,werr])
        dq=J.T@np.linalg.solve(J@J.T+0.05**2*np.eye(6),err); q=np.clip(q+np.clip(dq,-0.12,0.12),JL[:,0],JL[:,1])
    d.qpos[:]=sq; d.qvel[:]=sv; mujoco.mj_forward(m,d); return q
Kp=np.array([360,360,360,360,190,120,65.]); Kd=2*np.sqrt(Kp); FMAX=np.array([80,80,80,80,80,12,12.])
OPEN=0.04; CLOSE=0.0
def qnow(): return np.array([d.qpos[a] for a in AQ])
def goto(qg,steps,grip,settle=100,rec=True):
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
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
def ik_pos(tp,q0,it=900):  # position-only IK (free orientation) -> reaches the far burner
    sq=d.qpos.copy(); sv=d.qvel.copy(); q=q0.copy()
    for _ in range(it):
        setq(q); mujoco.mj_forward(m,d)
        perr=tp-d.site_xpos[site].copy()
        if np.linalg.norm(perr)<5e-4: break
        jp=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,np.zeros((3,m.nv)),site)
        dq=jp[:,AD].T@np.linalg.solve(jp[:,AD]@jp[:,AD].T+0.05**2*np.eye(3),perr); q=np.clip(q+np.clip(dq,-0.12,0.12),JL[:,0],JL[:,1])
    d.qpos[:]=sq; d.qvel[:]=sv; mujoco.mj_forward(m,d); return q
def move(target,grip,n=5,steps=180,settle=30):  # chained Cartesian (robust IK to far targets)
    start=d.site_xpos[site].copy(); target=np.array(target,float)
    for i in range(1,n+1):
        goto(ik(start+(target-start)*i/n,Rdown,qnow()),steps,grip,settle if i==n else 8)
    return d.site_xpos[site].copy()
def move_pos(target,grip,n=5,steps=180,settle=40):  # chained position-only (free orientation)
    start=d.site_xpos[site].copy(); target=np.array(target,float)
    for i in range(1,n+1):
        goto(ik_pos(start+(target-start)*i/n,qnow()),steps,grip,settle if i==n else 10)
    return d.site_xpos[site].copy()
def grasp(o):  # robust: chained move to high-above (good seed), verify eef over moka, descend, close
    move((o[0],o[1],o[2]+0.20),OPEN,n=4,steps=150)
    e=d.site_xpos[site]
    if abs(e[0]-o[0])>0.04 or abs(e[1]-o[1])>0.04:   # IK diverged -> re-seed from home
        goto(ik(np.array([0.0,0.1,1.25]),Rdown,qnow()),250,OPEN,settle=10); move((o[0],o[1],o[2]+0.20),OPEN,n=4,steps=150)
    goto(ik(np.array([o[0],o[1],o[2]+0.03]),Rdown,qnow()),320,OPEN,settle=50)  # upper-body grip (reliable for BOTH mokas)
    goto(qnow(),260,CLOSE,settle=0)
    return d.qpos[gf[0]]-d.qpos[gf[1]]
def pick_place(mkbody,place_xy,label):
    o=d.xpos[mkbody].copy(); print(f"{label}: moka at {np.round(o,3)}")
    gw=grasp(o); tries=0
    while gw<0.045 and tries<2:
        move((o[0],o[1],o[2]+0.20),OPEN,n=2); o=d.xpos[mkbody].copy(); gw=grasp(o); tries+=1; print(f"   retry{tries} gw={gw:.4f}")
    goto(ik(np.array([o[0],o[1],1.10]),Rdown,qnow()),400,CLOSE,settle=40)   # lift straight up SLOW
    lifted=d.xpos[mkbody][2]-o[2]; print(f"   grip width={gw:.4f} lifted={lifted:.3f}")
    move((0.13,place_xy[1],1.08),CLOSE,n=6,steps=160,settle=40)              # gripper-down to near-burner (reachable)
    # free-orientation reach ONTO the raised burner (gripper tilts forward to reach x>0.15)
    eef=move_pos((place_xy[0],place_xy[1],0.965),CLOSE,n=5,steps=170,settle=80)
    print(f"   eef on burner={np.round(eef,3)} (target {place_xy})")
    goto(qnow(),140,OPEN,settle=220)                  # release onto burner + settle (topple OK if center in region)
    print(f"   after release moka={np.round(d.xpos[mkbody],3)}")
    move_pos((place_xy[0]-0.03,place_xy[1],1.10),OPEN,n=3,settle=20)  # retreat up-and-back (clear the moka)
    print(f"   placed -> moka now {np.round(d.xpos[mkbody],3)}")
for _ in range(8): snap()
print("init check_success:",env.check_success(),"button qpos:",round(float(d.qpos[env.sim.model.get_joint_qpos_addr('flat_stove_1_button')]),3))
# place well inside the 15cm region (x[0.14,0.29] y[-0.024,0.126]); release drifts ~-0.04 in x so
# target +x interior. moka_1 back-right, moka_2 front-left, separated so the 2nd doesn't knock the 1st.
# moka_2 FIRST at controlled FAR spot (x0.21); moka_1 LAST at front-left (CLOSER to robot than moka_2,
# so moka_1's tilted reach never sweeps over moka_2, and being placed last it stays put).
pick_place(mk2,(0.21,0.09),"moka_2")
goto(ik(np.array([0.0,0.12,1.25]),Rdown,qnow()),280,OPEN,settle=20)  # neutral high seed
pick_place(mk1,(0.155,0.0),"moka_1")
goto(ik(np.array([-0.12,0.14,1.22]),Rdown,qnow()),300,OPEN,settle=30)  # park away high-LEFT (clear of region)
def inreg(p): return 0.14<=p[0]<=0.29 and -0.024<=p[1]<=0.126
print(f"moka1 inreg={inreg(d.xpos[mk1])} moka2 inreg={inreg(d.xpos[mk2])}")
for _ in range(20): snap()
ok=env.check_success()
imageio.mimsave(f"{OUT}/swap_t8_SOLVE.mp4", frames, fps=30, quality=8); rend.close()
print(f"\n==== moka1={np.round(d.xpos[mk1],3)} moka2={np.round(d.xpos[mk2],3)} cook={COOK} CHECK_SUCCESS={ok} ({len(frames)} frames) ====")
env.close()
