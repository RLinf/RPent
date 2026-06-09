"""10_task_t3 SOLVE: 'put the bottle in the bottom drawer of the cabinet and close it'.
Bottom drawer is ALREADY OPEN (qpos -0.144). Joint-space: grasp the upright bottle, lift, ROTATE to
horizontal (long axis along +y = drawer depth) and lay it FLAT in the drawer, release, then PUSH the
drawer closed (+y). Goal: In(wine_bottle, bottom_region) AND Close(bottom_region)."""
import os, sys, numpy as np, mujoco, imageio
os.environ.setdefault("MUJOCO_GL","egl"); os.environ["LIBERO_TYPE"]="pro"
import liberopro.liberopro.benchmark as bench
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation as R
OUT="/mnt/public/jxqiu/physicalagent/physicalagent/primitives/result_paper/goal_fail_renders"
b=bench.get_benchmark("libero_10_task")()
env=OffScreenRenderEnv(bddl_file_name=b.get_task_bddl_file_path(3),camera_heights=128,camera_widths=128)
env.seed(0); env.reset(); env.set_init_state(b.get_task_init_states(3)[0])
for _ in range(5): env.step(np.zeros(7))
m,d=env.sim.model._model, env.sim.data._data
site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"gripper0_grip_site")
AQ=[env.sim.model.get_joint_qpos_addr(f"robot0_joint{i}") for i in range(1,8)]
AD=[m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)]
JL=np.array([m.jnt_range[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f"robot0_joint{i}")] for i in range(1,8)])
jadr=env.sim.model.get_joint_qpos_addr("white_cabinet_1_bottom_level")
gf=[env.sim.model.get_joint_qpos_addr(f"gripper0_finger_joint{i}") for i in (1,2)]
bottle=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"wine_bottle_1_main")
REG=np.array([-0.006,0.174,0.952])
rend=mujoco.Renderer(m,480,480); frames=[]
def snap(): rend.update_scene(d,camera="frontview"); frames.append(rend.render().copy())
def setq(q,grip=0.04):
    for jj,a in enumerate(AQ): d.qpos[a]=q[jj]
    d.qpos[gf[0]]=grip; d.qpos[gf[1]]=-grip
def ik(tp,tR,q0,it=900):
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
def goto(qg,steps,grip,settle=80,rec=True):
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
def imove(tp,tR,grip,n=4,steps=170,settle=40):  # chained Cartesian+orientation move
    p0=d.site_xpos[site].copy(); R0=R.from_matrix(d.site_xmat[site].reshape(3,3))
    R1=R.from_matrix(tR); tp=np.array(tp,float)
    from scipy.spatial.transform import Slerp
    sl=Slerp([0,1],R.concatenate([R0,R1]))
    for i in range(1,n+1):
        a=i/n; goto(ik(p0+(tp-p0)*a, sl([a])[0].as_matrix(), qnow()), steps, grip, settle if i==n else 8)
    return d.site_xpos[site].copy()
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],float)
Rlay=R.from_rotvec([np.pi/2,0,0]).as_matrix()@Rdown   # approach -> +y: rotate held bottle to flat along +y
for _ in range(8): snap()
print("init In/Close check:",env.check_success(),"bottom qpos:",round(float(d.qpos[jadr]),4))
o=d.xpos[bottle].copy(); print("bottle:",np.round(o,3))
# ---- grasp the UPRIGHT bottle at its body center (g0 mesh world pos) ----
g0=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,"wine_bottle_1_g0"); c=d.geom_xpos[g0].copy()
print("body-center g0:",np.round(c,3))
goto(ik(np.array([c[0],c[1],c[2]+0.18]),Rdown,qnow()),300,OPEN)
goto(ik(np.array([c[0],c[1],c[2]+0.0]),Rdown,qnow()),300,OPEN,settle=40)
goto(qnow(),240,CLOSE,settle=0); gw=d.qpos[gf[0]]-d.qpos[gf[1]]
goto(ik(np.array([c[0],c[1],1.20]),Rdown,qnow()),350,CLOSE,settle=30)
print(f"grasp gw={gw:.4f} bottle z={d.xpos[bottle][2]:.3f}")
# ROTATE to horizontal (Rlay) over the drawer, lay the bottle FLAT along +y, lower in, release
imove(np.array([REG[0],0.05,1.18]),Rdown,CLOSE,n=4)
imove(np.array([REG[0],0.13,1.06]),Rlay,CLOSE,n=6,steps=180,settle=50)    # rotate to flat + position over drawer
imove(np.array([REG[0],0.15,0.99]),Rlay,CLOSE,n=3,steps=170,settle=70)    # lower flat into drawer
print("over-drawer eef:",np.round(d.site_xpos[site],3),"bottle:",np.round(d.xpos[bottle],3))
goto(qnow(),150,OPEN,settle=140)   # release
print("after release bottle:",np.round(d.xpos[bottle],3))
goto(ik(np.array([REG[0]-0.02,-0.02,1.18]),Rdown,qnow()),300,CLOSE,settle=20)  # retreat up-front, gripper closed
inreg = (abs(d.xpos[bottle][0]-REG[0])<0.10 and abs(d.xpos[bottle][1]-REG[1])<0.12)
print(f"bottle in drawer region(approx)={inreg} In before close={env.check_success()}")
# ===== CLOSE the drawer: push the front face +y with the closed gripper =====
goto(ik(np.array([-0.006,-0.03,0.96]),Rdown,qnow()),300,CLOSE,settle=20)   # in front of the open drawer face
cur=d.site_xpos[site].copy()
for k in range(1,10):
    goto(ik(np.array([-0.006,cur[1]+0.03*k,0.96]),Rdown,qnow()),90,CLOSE,settle=15)  # push +y to close
    qp=float(d.qpos[jadr])
    if k%3==0: print(f"  close push {k}: drawer qpos={qp:.4f}")
    if qp>-0.02: break
print(f"after close: drawer qpos={float(d.qpos[jadr]):.4f} bottle={np.round(d.xpos[bottle],3)} success={env.check_success()}")
for _ in range(15): snap()
imageio.mimsave(f"{OUT}/task_t3_SOLVE.mp4", frames, fps=30, quality=8); rend.close()
print(f"\n==== bottle={np.round(d.xpos[bottle],3)} bottom qpos={round(float(d.qpos[jadr]),4)} CHECK_SUCCESS={env.check_success()} ====")
env.close()
