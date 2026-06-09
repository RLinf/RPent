"""task_t0 v3 (open BOTTOM drawer; clear plate+bowl, relaxed oracle, elbow-up search, default cabinet) SOLVE. The low bar + bowl/plate clutter make
the home-seed IK pick a colliding arm branch. Fix: (1) proper collision oracle = ANY robot-vs-env
deep penetration except finger-pad<->cabinet (the grasp); (2) multi-seed search for a collision-free
at-bar config; (3) RRT-Connect around the bowl/plate; (4) close (pinch bar) + pull +y -> Open(drawer)."""
import os, sys, time, numpy as np, mujoco, imageio
os.environ.setdefault("MUJOCO_GL","egl"); os.environ["LIBERO_TYPE"]="pro"
import liberopro.liberopro.benchmark as bench
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation as R
np.random.seed(0)
OUT="/mnt/public/jxqiu/physicalagent/physicalagent/primitives/result_paper/goal_fail_renders"
b=bench.get_benchmark("libero_goal_task")()
env=OffScreenRenderEnv(bddl_file_name=b.get_task_bddl_file_path(0),camera_heights=128,camera_widths=128)
env.seed(0); env.reset(); env.set_init_state(b.get_task_init_states(0)[0])
for _ in range(5): env.step(np.zeros(7))
m,d=env.sim.model._model, env.sim.data._data
site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"gripper0_grip_site")
AQ=[env.sim.model.get_joint_qpos_addr(f"robot0_joint{i}") for i in range(1,8)]
AD=[m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)]
JL=np.array([m.jnt_range[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)])
jadr=env.sim.model.get_joint_qpos_addr("wooden_cabinet_1_bottom_level")
gf=[env.sim.model.get_joint_qpos_addr(f"gripper0_finger_joint{i}") for i in (1,2)]
HANDLE=np.array([0.043,-0.151,0.946])
rend=mujoco.Renderer(m,480,480); frames=[]
def snap(): rend.update_scene(d,camera="frontview"); frames.append(rend.render().copy())
def setq(q,grip=0.04):
    for jj,a in enumerate(AQ): d.qpos[a]=q[jj]
    d.qpos[gf[0]]=grip; d.qpos[gf[1]]=-grip
def gn(g): return mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or str(g)
PEN=-0.006
def is_rob(n): return "robot0" in n or "gripper" in n
def env_clear(q,grip=0.04):
    """Arm links + hand must clear EVERYTHING; fingers/pads must clear table/bowl/plate;
    fingers/pads MAY touch the cabinet (that is the grasp at the bar)."""
    sq=d.qpos.copy(); sv=d.qvel.copy(); setq(q,grip); mujoco.mj_forward(m,d); ok=True
    for c in range(d.ncon):
        if d.contact[c].dist>PEN: continue
        n1=gn(d.contact[c].geom1); n2=gn(d.contact[c].geom2); r1=is_rob(n1); r2=is_rob(n2)
        if r1==r2: continue
        rob=n1 if r1 else n2; ev=n2 if r1 else n1
        is_grip=("finger" in rob or "pad" in rob)
        if is_grip and "cabinet" in ev: continue              # gripper touching the bar: allowed
        ok=False; break
    d.qpos[:]=sq; d.qvel[:]=sv; mujoco.mj_forward(m,d); return ok
def ik(tp,tR,q0,it=1500):
    q=q0.copy(); pe=we=9
    for _ in range(it):
        setq(q); mujoco.mj_forward(m,d)
        p=d.site_xpos[site].copy(); Rc=d.site_xmat[site].reshape(3,3).copy()
        perr=tp-p; werr=R.from_matrix(tR@Rc.T).as_rotvec(); pe=np.linalg.norm(perr); we=np.linalg.norm(werr)
        if pe<3e-4 and we<0.01: break
        jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,jr,site)
        J=np.concatenate([jp[:,AD],jr[:,AD]],0); err=np.concatenate([perr,werr])
        dq=J.T@np.linalg.solve(J@J.T+0.04**2*np.eye(6),err); q=np.clip(q+np.clip(dq,-0.1,0.1),JL[:,0],JL[:,1])
    return q,pe,we
def chain_ik(seed,x,z,ys=(0.12,0.06,0.0,-0.06,-0.12,-0.151)):
    q=seed.copy()
    for ty in ys: q,e,w=ik(np.array([x,ty,z]),Rg,q)
    return q,e,w
def edge_clear(qa,qb,res=0.04,grip=0.04):
    n=int(np.ceil(np.linalg.norm(qb-qa)/res))
    for i in range(1,n+1):
        if not env_clear(qa+(qb-qa)*i/n,grip): return False
    return True
def rrt(qs,qg,grip=0.04,iters=6000,step=0.5,goal_bias=0.2):
    if edge_clear(qs,qg,grip=grip): return [qs,qg]
    Ta=[qs]; Tb=[qg]; Pa={0:-1}; Pb={0:-1}
    def nearest(T,q): return int(np.argmin([np.linalg.norm(n-q) for n in T]))
    def extend(T,P,q):
        i=nearest(T,q); dirn=q-T[i]; dist=np.linalg.norm(dirn)
        if dist<1e-6: return i,False
        qnew=np.clip(T[i]+dirn/dist*min(step,dist),JL[:,0],JL[:,1])
        if env_clear(qnew,grip) and edge_clear(T[i],qnew,grip=grip):
            T.append(qnew); P[len(T)-1]=i; return len(T)-1,np.linalg.norm(qnew-q)<1e-6
        return i,False
    for it in range(iters):
        qr= qg if np.random.rand()<goal_bias else np.random.uniform(JL[:,0],JL[:,1])
        ia,_=extend(Ta,Pa,qr); qtarget=Ta[ia]
        while True:
            ib,reached=extend(Tb,Pb,qtarget)
            if reached or ib==nearest(Tb,qtarget): break
        jb=nearest(Tb,qtarget)
        if np.linalg.norm(Tb[jb]-qtarget)<1e-6 or edge_clear(Ta[ia],Tb[jb],grip=grip):
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
Rg=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)@R.from_rotvec([0,0,np.pi/2]).as_matrix()  # horizontal rolled: pads straddle bar in z
for _ in range(8): snap()
# ---- clear the plate and bowl out of the way (PD pick, close=0.0, y-drift comp) ----
def pd_goto(qg,steps,grip,settle=80):
    q0=qnow()
    for sps in range(steps):
        a=(sps+1)/steps; qt=q0+(qg-q0)*a; q=qnow(); qd=np.array([d.qvel[AD[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qt-q)-Kd*qd+d.qfrc_bias[AD],-FMAX,FMAX); d.ctrl[7]=grip; d.ctrl[8]=-grip; mujoco.mj_step(m,d)
        if sps%7==0: snap()
    for _ in range(settle):
        q=qnow(); qd=np.array([d.qvel[AD[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qg-q)-Kd*qd+d.qfrc_bias[AD],-FMAX,FMAX); d.ctrl[7]=grip; d.ctrl[8]=-grip; mujoco.mj_step(m,d)
        if _%7==0: snap()
    mujoco.mj_forward(m,d)
for objn,drop in [("plate_1_main",(0.30,0.18)),("akita_black_bowl_1_main",(0.30,0.30))]:
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,objn); o=d.xpos[bid].copy()
    pd_goto(ik(np.array([o[0],o[1]-0.02,o[2]+0.18]),Rdown,qnow())[0],250,0.04)
    pd_goto(ik(np.array([o[0],o[1]-0.02,o[2]+0.06]),Rdown,qnow())[0],250,0.04,settle=30)
    pd_goto(qnow(),180,0.0,settle=0)
    pd_goto(ik(np.array([o[0],o[1]-0.02,o[2]+0.28]),Rdown,qnow())[0],250,0.0,settle=20)
    pd_goto(ik(np.array([drop[0],drop[1],1.10]),Rdown,qnow())[0],350,0.0,settle=20)
    pd_goto(qnow(),120,0.04,settle=40)
    print(f"cleared {objn} -> now {np.round(d.xpos[bid],3)}")
pd_goto(ik(np.array([0.0,0.15,1.15]),Rdown,qnow())[0],250,0.04,settle=20)  # park center-up
qhome=qnow()
# ---- search a collision-free at-bar config: vary seed + grasp height ----
print("searching collision-free bottom-bar grasp config...")
qsol=None
seeds=[qhome]+[np.clip(qhome+np.random.uniform(-1.0,1.0,7),JL[:,0],JL[:,1]) for _ in range(120)]
for hz in [0.946,0.950,0.942,0.955,0.960]:
    for sd in seeds:
        q,e,w=chain_ik(sd,HANDLE[0],hz)
        if e<0.002 and w<0.02 and env_clear(q):
            qsol=q; print(f"  FOUND clear config: hz={hz} ik_err={e:.4f}")
            break
    if qsol is not None: break
if qsol is None:
    print("NO collision-free bottom-bar grasp config found in search")
    imageio.mimsave(f"{OUT}/task_t0_V3_SOLVE.mp4", frames, fps=30, quality=8); rend.close(); env.close(); sys.exit()
setq(qsol,0.04); mujoco.mj_forward(m,d); print("grasp config eef:",np.round(d.site_xpos[site],3),"(bar",HANDLE,")")
setq(qhome); mujoco.mj_forward(m,d)
# ---- RRT around the bowl/plate to the grasp config ----
t0=time.time(); path=rrt(qhome,qsol,grip=0.04)
print(f"RRT: {'FOUND len '+str(len(path)) if path else 'FAILED'} in {time.time()-t0:.1f}s")
if not path:
    imageio.mimsave(f"{OUT}/task_t0_V3_SOLVE.mp4", frames, fps=30, quality=8); rend.close(); print("RRT FAILED"); env.close(); sys.exit()
for wp in path: goto(wp,500,0.04,settle=120)
print("reached eef:",np.round(d.site_xpos[site],3))
goto(qnow(),300,0.0,settle=0); print("grip on bar:",round(d.qpos[gf[0]]-d.qpos[gf[1]],4))
cur=d.site_xpos[site].copy()
for k in range(1,14):
    qk,_,_=ik(cur+np.array([0,0.018*k,0.0]),Rg,qnow()); goto(qk,90,0.0,settle=20)
    if k%3==0: print(f"  pull {k}: eef_y={d.site_xpos[site][1]:.3f} bottom_qpos={float(d.qpos[jadr]):.4f}")
for _ in range(20): snap()
ok=env.check_success(); qp=round(float(d.qpos[jadr]),4)
imageio.mimsave(f"{OUT}/task_t0_V3_SOLVE.mp4", frames, fps=30, quality=8); rend.close()
print(f"\n==== FINAL bottom qpos={qp}  CHECK_SUCCESS={ok}  ({len(frames)} frames) ====")
env.close()
