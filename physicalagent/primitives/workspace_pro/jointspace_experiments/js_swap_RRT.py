"""swap_t0 SOLVE via motion planning: (1) PICK the wine bottle out of the way (PD, close=0.0,
y-comp). (2) The bar IS reachable collision-free (verified) but the straight-line drive sweeps the
arm through the cabinet -> RRT-Connect a collision-free joint path to the at-bar config (rolled
orientation: pads straddle the bar in z). (3) drive path, close (pinch bar), pull +y -> Open(drawer)."""
import os, sys, time, numpy as np, mujoco, imageio
os.environ.setdefault("MUJOCO_GL","egl"); os.environ["LIBERO_TYPE"]="pro"
import liberopro.liberopro.benchmark as bench
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation as R
np.random.seed(0)
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
bottleB=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"wine_bottle_1_main")
HANDLE=np.array([-0.247,-0.155,1.015])
rend=mujoco.Renderer(m,480,480); frames=[]
def snap(): rend.update_scene(d,camera="frontview"); frames.append(rend.render().copy())
def setq(q,grip=0.04):
    for jj,a in enumerate(AQ): d.qpos[a]=q[jj]
    d.qpos[gf[0]]=grip; d.qpos[gf[1]]=-grip
PEN=-0.004
def cabinet_contacts():
    n=0
    for ci in range(d.ncon):
        c=d.contact[ci]
        if c.dist>PEN: continue
        n1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom1]) or ""
        n2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom2]) or ""
        rob=("gripper" in n1 or "robot0" in n1 or "gripper" in n2 or "robot0" in n2)
        obs=("cabinet" in n1 or "cabinet" in n2 or "wine_bottle" in n1 or "wine_bottle" in n2)
        if rob and obs: n+=1
    return n
def collision_free(q,grip=0.04):
    sq=d.qpos.copy(); sv=d.qvel.copy()
    setq(q,grip); mujoco.mj_forward(m,d); ok=cabinet_contacts()==0
    d.qpos[:]=sq; d.qvel[:]=sv; mujoco.mj_forward(m,d); return ok
def ik(tp,tR,q0,it=1500):
    sq=d.qpos.copy(); sv=d.qvel.copy(); q=q0.copy(); pe=we=9
    for _ in range(it):
        setq(q); mujoco.mj_forward(m,d)
        p=d.site_xpos[site].copy(); Rc=d.site_xmat[site].reshape(3,3).copy()
        perr=tp-p; werr=R.from_matrix(tR@Rc.T).as_rotvec(); pe=np.linalg.norm(perr); we=np.linalg.norm(werr)
        if pe<3e-4 and we<0.01: break
        jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,jr,site)
        J=np.concatenate([jp[:,AD],jr[:,AD]],0); err=np.concatenate([perr,werr])
        dq=J.T@np.linalg.solve(J@J.T+0.04**2*np.eye(6),err); q=np.clip(q+np.clip(dq,-0.1,0.1),JL[:,0],JL[:,1])
    d.qpos[:]=sq; d.qvel[:]=sv; mujoco.mj_forward(m,d); return q,pe,we
def edge_free(qa,qb,res=0.04,grip=0.04):
    n=int(np.ceil(np.linalg.norm(qb-qa)/res))
    for i in range(1,n+1):
        if not collision_free(qa+(qb-qa)*i/n,grip): return False
    return True
def rrt(qs,qg,grip=0.04,iters=5000,step=0.5,goal_bias=0.2):
    if edge_free(qs,qg,grip=grip): return [qs,qg]
    Ta=[qs]; Tb=[qg]; Pa={0:-1}; Pb={0:-1}
    def nearest(T,q): return int(np.argmin([np.linalg.norm(n-q) for n in T]))
    def extend(T,P,q):
        i=nearest(T,q); dirn=q-T[i]; dist=np.linalg.norm(dirn)
        if dist<1e-6: return i,False
        qnew=np.clip(T[i]+dirn/dist*min(step,dist),JL[:,0],JL[:,1])
        if collision_free(qnew,grip) and edge_free(T[i],qnew,grip=grip):
            T.append(qnew); P[len(T)-1]=i; return len(T)-1,np.linalg.norm(qnew-q)<1e-6
        return i,False
    for it in range(iters):
        qr= qg if np.random.rand()<goal_bias else np.random.uniform(JL[:,0],JL[:,1])
        ia,_=extend(Ta,Pa,qr); qtarget=Ta[ia]
        while True:
            ib,reached=extend(Tb,Pb,qtarget)
            if reached or ib==nearest(Tb,qtarget): break
        jb=nearest(Tb,qtarget)
        if np.linalg.norm(Tb[jb]-qtarget)<1e-6 or edge_free(Ta[ia],Tb[jb],grip=grip):
            pa=[]; i=ia
            while i!=-1: pa.append(Ta[i]); i=Pa[i]
            pb=[]; i=jb
            while i!=-1: pb.append(Tb[i]); i=Pb[i]
            return pa[::-1]+pb
        Ta,Tb,Pa,Pb=Tb,Ta,Pb,Pa
    return None
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
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
Rg=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)@R.from_rotvec([0,0,np.pi/2]).as_matrix()  # rolled: pads straddle bar in z
for _ in range(8): snap()
# compute the at-bar config from the HOME seed (good IK basin) BEFORE moving — pure kinematics
qhome=qnow(); qsol=qhome.copy()
for ty in [0.10,0.05,0.0,-0.05,-0.10,-0.13,-0.155]:
    qsol,e,w=ik(np.array([HANDLE[0],ty,HANDLE[2]]),Rg,qsol)
print(f"IK-at-bar (home seed) err={e:.4f} collision_free={collision_free(qsol)}")
# ===== PHASE 0: PICK the bottle out of the way (PD) =====
bot=d.xpos[bottleB].copy(); print("bottle start",np.round(bot,3))
goto(ik(np.array([bot[0],bot[1]-0.02,1.12]),Rdown,qnow())[0],300,0.04)
goto(ik(np.array([bot[0],bot[1]-0.02,1.02]),Rdown,qnow())[0],300,0.04,settle=40)
goto(qnow(),200,0.0,settle=0)
goto(ik(np.array([bot[0],bot[1]-0.02,1.18]),Rdown,qnow())[0],300,0.0,settle=20)
goto(ik(np.array([0.18,0.22,1.18]),Rdown,qnow())[0],450,0.0,settle=20)
goto(qnow(),150,0.04,settle=60)
print("bottle after removal",np.round(d.xpos[bottleB],3))
# ===== PHASE 1: RRT from current config to the precomputed collision-free at-bar config =====
qstart=qnow()
t0=time.time(); path=rrt(qstart,qsol,grip=0.04)
print(f"RRT: {'FOUND len '+str(len(path)) if path else 'FAILED'} in {time.time()-t0:.1f}s")
if not path:
    imageio.mimsave(f"{OUT}/swap_t0_SOLVE.mp4", frames, fps=30, quality=8); rend.close(); print("RRT FAILED"); env.close(); sys.exit()
for wp in path: goto(wp,500,0.04,settle=120)
print("reached eef:",np.round(d.site_xpos[site],3),"(bar",HANDLE,")")
# ===== PHASE 2: close (pinch bar) + pull +y =====
goto(qnow(),300,0.0,settle=0)
print("grip on bar:",round(d.qpos[gf[0]]-d.qpos[gf[1]],4))
cur=d.site_xpos[site].copy()
for k in range(1,14):
    qk,_,_=ik(cur+np.array([0,0.018*k,0.0]),Rg,qnow()); goto(qk,90,0.0,settle=20)
    if k%3==0: print(f"  pull {k}: eef_y={d.site_xpos[site][1]:.3f} mid_qpos={float(d.qpos[jadr]):.4f}")
for _ in range(20): snap()
ok=env.check_success(); qp=round(float(d.qpos[jadr]),4)
imageio.mimsave(f"{OUT}/swap_t0_SOLVE.mp4", frames, fps=30, quality=8); rend.close()
print(f"\n==== FINAL middle qpos={qp}  CHECK_SUCCESS={ok}  ({len(frames)} frames) ====")
env.close()
