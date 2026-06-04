"""Joint-space RRT motion planner (MuJoCo collision oracle) + IK + PD control to
open a relocated/stacked cabinet drawer where the OSC operational-space servo walls.
Non-parametric: damped-LS IK, RRT-Connect in C-space, PD torque tracking. Physics-only."""
import os, sys, time
os.environ.setdefault("MUJOCO_GL","egl"); os.environ["LIBERO_TYPE"]="pro"
import numpy as np, mujoco
import liberopro.liberopro.benchmark as bench
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation as R

SUITE=os.environ.get("SUITE","libero_goal_swap"); TASK=int(os.environ.get("TASK","0"))
HANDLE=np.array([float(x) for x in os.environ.get("HANDLE","-0.247,-0.152,1.015").split(",")])
JOINTNAME=os.environ.get("JOINTNAME","wooden_cabinet_1_middle_level")
np.random.seed(0)

b=bench.get_benchmark(SUITE)()
env=OffScreenRenderEnv(bddl_file_name=b.get_task_bddl_file_path(TASK),camera_heights=128,camera_widths=128)
env.seed(0); env.reset(); env.set_init_state(b.get_task_init_states(TASK)[0])
for _ in range(5): env.step(np.zeros(7))
m,d=env.sim.model._model, env.sim.data._data   # bind AFTER reset
site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"gripper0_grip_site")
AQ=[env.sim.model.get_joint_qpos_addr(f"robot0_joint{i}") for i in range(1,8)]
AD=[m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)]
JL=np.array([m.jnt_range[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)])
jadr=env.sim.model.get_joint_qpos_addr(JOINTNAME)
gf=[env.sim.model.get_joint_qpos_addr(f"gripper0_finger_joint{i}") for i in (1,2)]

def setq(q,grip=0.04):
    for jj,a in enumerate(AQ): d.qpos[a]=q[jj]
    d.qpos[gf[0]]=grip; d.qpos[gf[1]]=-grip
PEN=-0.004
def cabinet_contacts():
    n=0
    for ci in range(d.ncon):
        c=d.contact[ci]
        if c.dist>PEN: continue   # count only deep penetration
        n1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom1]) or ""
        n2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom2]) or ""
        rob=("gripper" in n1 or "robot0" in n1 or "gripper" in n2 or "robot0" in n2)
        obs=("cabinet" in n1 or "cabinet" in n2)
        if rob and obs: n+=1
    return n
def collision_free(q,grip=0.04):
    sq=d.qpos.copy(); sv=d.qvel.copy()
    setq(q,grip); mujoco.mj_forward(m,d); ok=cabinet_contacts()==0
    d.qpos[:]=sq; d.qvel[:]=sv; mujoco.mj_forward(m,d); return ok
def ik(tp,tR,q0,iters=600):
    sq=d.qpos.copy(); sv=d.qvel.copy(); q=q0.copy(); pe=we=9
    for it in range(iters):
        setq(q); mujoco.mj_forward(m,d)
        p=d.site_xpos[site].copy(); Rc=d.site_xmat[site].reshape(3,3).copy()
        perr=tp-p; werr=R.from_matrix(tR@Rc.T).as_rotvec(); pe=np.linalg.norm(perr); we=np.linalg.norm(werr)
        if pe<5e-4 and we<0.012: break
        jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,jr,site)
        J=np.concatenate([jp[:,AD],jr[:,AD]],0); err=np.concatenate([perr,werr])
        dq=J.T@np.linalg.solve(J@J.T+0.05**2*np.eye(6),err); q=q+np.clip(dq,-0.15,0.15)
        q=np.clip(q,JL[:,0],JL[:,1])
    d.qpos[:]=sq; d.qvel[:]=sv; mujoco.mj_forward(m,d); return q,pe,we

# ---- RRT-Connect in joint space ----
def edge_free(qa,qb,res=0.04,grip=0.04):
    n=int(np.ceil(np.linalg.norm(qb-qa)/res))
    for i in range(1,n+1):
        if not collision_free(qa+(qb-qa)*i/n,grip): return False
    return True
def rrt(qs,qg,grip=0.04,iters=4000,step=0.5,goal_bias=0.2):
    if edge_free(qs,qg,grip=grip): return [qs,qg]
    Ta=[qs]; Tb=[qg]; Pa={0:-1}; Pb={0:-1}
    def nearest(T,q): return int(np.argmin([np.linalg.norm(n-q) for n in T]))
    def extend(T,P,q):
        i=nearest(T,q); dirn=q-T[i]; dist=np.linalg.norm(dirn)
        if dist<1e-6: return i,False
        qnew=T[i]+dirn/dist*min(step,dist)
        qnew=np.clip(qnew,JL[:,0],JL[:,1])
        if collision_free(qnew,grip) and edge_free(T[i],qnew,grip=grip):
            T.append(qnew); P[len(T)-1]=i; return len(T)-1,np.linalg.norm(qnew-q)<1e-6
        return i,False
    for it in range(iters):
        qr = qg if np.random.rand()<goal_bias else np.random.uniform(JL[:,0],JL[:,1])
        ia,_=extend(Ta,Pa,qr)
        # try connect Tb to the new node of Ta
        qtarget=Ta[ia];
        while True:
            ib,reached=extend(Tb,Pb,qtarget)
            if reached or ib==nearest(Tb,qtarget):
                break
        if np.linalg.norm(Tb[nearest(Tb,qtarget)]-qtarget)<1e-6 or edge_free(Ta[ia],Tb[nearest(Tb,qtarget)],grip=grip):
            # build path
            pa=[]; i=ia
            while i!=-1: pa.append(Ta[i]); i=Pa[i]
            pa=pa[::-1]
            jb=nearest(Tb,qtarget); pb=[]; i=jb
            while i!=-1: pb.append(Tb[i]); i=Pb[i]
            return pa+pb
        Ta,Tb,Pa,Pb=Tb,Ta,Pb,Pa  # swap trees
    return None

# ---- PD control ----
Kp=np.array([300,300,300,300,150,90,50.]); Kd=2*np.sqrt(Kp)*1.0; FMAX=np.array([80,80,80,80,80,12,12.])
def qnow(): return np.array([d.qpos[a] for a in AQ])
def goto(qg,steps,grip,settle=120):
    q0=qnow()
    for s in range(steps):
        a=(s+1)/steps; qt=q0+(qg-q0)*a; q=qnow(); qd=np.array([d.qvel[AD[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qt-q)-Kd*qd+d.qfrc_bias[AD],-FMAX,FMAX); d.ctrl[7]=grip; d.ctrl[8]=-grip; mujoco.mj_step(m,d)
    for s in range(settle):
        q=qnow(); qd=np.array([d.qvel[AD[i]] for i in range(7)])
        d.ctrl[0:7]=np.clip(Kp*(qg-q)-Kd*qd+d.qfrc_bias[AD],-FMAX,FMAX); d.ctrl[7]=grip; d.ctrl[8]=-grip; mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d); return d.site_xpos[site].copy()

Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
Rgrab=R.from_rotvec([-0.45,0,0]).as_matrix()@np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)@R.from_rotvec([0,0,np.pi/2]).as_matrix()  # pitched, finger-sep world-y (front/back) -> pulls
qhome=qnow()
# find a collision-free grasp config (gripper OPEN) with grip_site near the bar
print("-- grasp-config search (gripper open, rolled orientation) --")
qgrasp=None
def detail(q,grip=0.04):
    sq=d.qpos.copy(); sv=d.qvel.copy(); setq(q,grip); mujoco.mj_forward(m,d)
    cnt=0; mind=0.0
    for ci in range(d.ncon):
        c=d.contact[ci]; n1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom1]) or ""; n2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom2]) or ""
        if ("gripper" in n1 or "robot0" in n1 or "gripper" in n2 or "robot0" in n2) and ("cabinet" in n1 or "cabinet" in n2):
            cnt+=1; mind=min(mind,c.dist)
    d.qpos[:]=sq; d.qvel[:]=sv; mujoco.mj_forward(m,d); return cnt,mind
for grip in [0.04,0.03]:
    for dy in [0.06,0.07,0.08,0.09,0.10]:
        for dz in [0.04,0.05]:
            q,e,w=ik(HANDLE+np.array([0,dy,dz]),Rgrab,qhome)
            cnt,mind=detail(q,grip); cf=collision_free(q,grip)
            if e<0.001 and (cf or cnt<=2):
                print(f"  grip={grip} dy={dy:.2f} dz={dz:.2f} ik_err={e:.4f} contacts={cnt} min_pen={mind:.4f} freeDeep={cf}")
                if qgrasp is None and cf: qgrasp=q; gdy=dy; ggrip=grip
if qgrasp is None: print("NO collision-free grasp config found"); env.close(); sys.exit()
print(f"chosen grasp dy={gdy}")
t0=time.time()
path=rrt(qhome,qgrasp,grip=0.04)
print(f"RRT: {'FOUND path len '+str(len(path)) if path else 'FAILED'} in {time.time()-t0:.1f}s")
if not path: env.close(); sys.exit()
for wp in path: goto(wp,700,0.04,settle=200)
print("reached pre-grasp eef:",np.round(d.site_xpos[site],4))
# forceful engage to the bar (front/back fingers straddle the bar), accept light contact
q_eng,_,_=ik(HANDLE+np.array([0,0.0,0.0]),Rgrab,qnow())
goto(q_eng,600,0.04,settle=300)
print("engage eef:",np.round(d.site_xpos[site],4),"(bar",HANDLE,")")
goto(qnow(),350,0.0,settle=0)  # close front/back on the bar
print("close grip width",round(d.qpos[gf[0]]-d.qpos[gf[1]],4),JOINTNAME,"qpos",round(float(d.qpos[jadr]),4))
# pull: incremental +y drag, gripper held closed, IK each small step (keeps the grip at the bar)
print("grip before pull",round(d.qpos[gf[0]]-d.qpos[gf[1]],4))
cur=d.site_xpos[site].copy()
for k in range(1,13):
    tgt=cur+np.array([0,0.02*k,0.0])
    qk,_,_=ik(tgt,Rgrab,qnow()); goto(qk,120,0.0,settle=20)
    if k%3==0: print(f"  pull step {k}: eef_y={d.site_xpos[site][1]:.3f} grip={d.qpos[gf[0]]-d.qpos[gf[1]]:.4f} {JOINTNAME}_qpos={float(d.qpos[jadr]):.4f}")
print("FINAL",JOINTNAME,"qpos",round(float(d.qpos[jadr]),4),"  CHECK_SUCCESS:",env.check_success())
env.close()
