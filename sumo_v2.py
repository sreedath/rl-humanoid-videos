"""
AI Sumo Wrestling v2 — Two Humanoids on an Elevated Disc
=========================================================
Two colored humanoids (blue vs red) on a circular elevated platform.
When pushed off the edge, they fall due to gravity.
Self-play PPO: one policy controls both agents.

Key improvements over v1:
  - Positional backend (GPU-accelerated, fast training)
  - ALL body parts have collision physics (not just feet)
  - Stronger leg gears for stable movement
  - Static camera centered on the disc
  - Post-fall recording (shows full gravity fall)
  - Vizuara branding, YouTube Shorts format
"""

import os, time, functools
from dataclasses import dataclass
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as brax_mjcf, model as brax_model
from brax.training.agents.ppo import train as ppo
import mujoco as mj_lib
import imageio
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Config:
    width: int = 1080
    height: int = 1920
    fps: int = 50
    num_timesteps: int = 30_000_000
    num_evals: int = 20
    num_envs: int = 512
    episode_length: int = 500
    ring_radius: float = 3.0
    ring_height: float = 2.5
    # Static camera
    cam_distance: float = 9.0
    cam_elevation: float = -25.0
    cam_azimuth: float = 90.0
    cam_lookat_z: float = 3.0
    post_done_frames: int = 60  # show full fall
    min_episode_frames: int = 90
    safe_top: int = 300
    safe_bottom: int = 1248
    safe_left: int = 100
    safe_right: int = 980
    output_dir: str = "/workspace/rl_walking/output"


@dataclass
class Checkpoint:
    step: int
    reward: float
    params: Any


# ---------------------------------------------------------------------------
# MJCF: Two humanoids on elevated disc, ALL parts have collision
# ---------------------------------------------------------------------------
def _humanoid_body(prefix, pos_x, pos_y, rgba, ring_height):
    z = ring_height + 1.5
    # ALL geoms have contype="1" conaffinity="1" for full body collision
    return f"""
        <body name="{prefix}torso" pos="{pos_x} {pos_y} {z}">
            <joint armature="0" damping="0" limited="false" name="{prefix}root" pos="0 0 0" stiffness="0" type="free"/>
            <geom fromto="0 -.07 0 0 .07 0" name="{prefix}torso1" size="0.07" type="capsule" contype="1" conaffinity="1" rgba="{rgba}"/>
            <geom name="{prefix}head" pos="0 0 .19" size=".09" type="sphere" user="258" contype="1" conaffinity="1" rgba="{rgba}"/>
            <geom fromto="-.01 -.06 -.12 -.01 .06 -.12" name="{prefix}uwaist" size="0.06" type="capsule" contype="1" conaffinity="1" rgba="{rgba}"/>
            <body name="{prefix}lwaist" pos="-.01 0 -0.260" quat="1.000 0 -0.002 0">
                <geom fromto="0 -.06 0 0 .06 0" name="{prefix}lwaist_g" size="0.06" type="capsule" contype="1" conaffinity="1" rgba="{rgba}"/>
                <joint armature="0.02" axis="0 0 1" damping="5" name="{prefix}abdomen_z" pos="0 0 0.065" range="-45 45" stiffness="20" type="hinge"/>
                <joint armature="0.02" axis="0 1 0" damping="5" name="{prefix}abdomen_y" pos="0 0 0.065" range="-75 30" stiffness="10" type="hinge"/>
                <body name="{prefix}pelvis" pos="0 0 -0.165" quat="1.000 0 -0.002 0">
                    <joint armature="0.02" axis="1 0 0" damping="5" name="{prefix}abdomen_x" pos="0 0 0.1" range="-35 35" stiffness="10" type="hinge"/>
                    <geom fromto="-.02 -.07 0 -.02 .07 0" name="{prefix}butt" size="0.09" type="capsule" contype="1" conaffinity="1" rgba="{rgba}"/>
                    <body name="{prefix}right_thigh" pos="0 -0.1 -0.04">
                        <joint armature="0.01" axis="1 0 0" damping="5" name="{prefix}right_hip_x" range="-25 5" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 0 1" damping="5" name="{prefix}right_hip_z" range="-60 35" stiffness="10" type="hinge"/>
                        <joint armature="0.008" axis="0 1 0" damping="5" name="{prefix}right_hip_y" range="-110 20" stiffness="20" type="hinge"/>
                        <geom fromto="0 0 0 0 0.01 -.34" name="{prefix}right_thigh1" size="0.06" type="capsule" contype="1" conaffinity="1" rgba="{rgba}"/>
                        <body name="{prefix}right_shin" pos="0 0.01 -0.403">
                            <joint armature="0.006" axis="0 -1 0" name="{prefix}right_knee" pos="0 0 .02" range="-160 -2" type="hinge"/>
                            <geom fromto="0 0 0 0 0 -.3" name="{prefix}right_shin1" size="0.049" type="capsule" contype="1" conaffinity="1" rgba="{rgba}"/>
                            <body name="{prefix}right_foot" pos="0 0 -0.45">
                                <geom contype="1" conaffinity="1" name="{prefix}right_foot" pos="0 0 0.1" size="0.075" type="sphere" rgba="{rgba}"/>
                            </body>
                        </body>
                    </body>
                    <body name="{prefix}left_thigh" pos="0 0.1 -0.04">
                        <joint armature="0.01" axis="-1 0 0" damping="5" name="{prefix}left_hip_x" range="-25 5" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 0 -1" damping="5" name="{prefix}left_hip_z" range="-60 35" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 1 0" damping="5" name="{prefix}left_hip_y" range="-110 20" stiffness="20" type="hinge"/>
                        <geom fromto="0 0 0 0 -0.01 -.34" name="{prefix}left_thigh1" size="0.06" type="capsule" contype="1" conaffinity="1" rgba="{rgba}"/>
                        <body name="{prefix}left_shin" pos="0 -0.01 -0.403">
                            <joint armature="0.006" axis="0 -1 0" name="{prefix}left_knee" pos="0 0 .02" range="-160 -2" stiffness="1" type="hinge"/>
                            <geom fromto="0 0 0 0 0 -.3" name="{prefix}left_shin1" size="0.049" type="capsule" contype="1" conaffinity="1" rgba="{rgba}"/>
                            <body name="{prefix}left_foot" pos="0 0 -0.45">
                                <geom contype="1" conaffinity="1" name="{prefix}left_foot" type="sphere" size="0.075" pos="0 0 0.1" rgba="{rgba}"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
            <body name="{prefix}right_upper_arm" pos="0 -0.17 0.06">
                <joint armature="0.0068" axis="2 1 1" name="{prefix}right_shoulder1" range="-85 60" stiffness="1" type="hinge"/>
                <joint armature="0.0051" axis="0 -1 1" name="{prefix}right_shoulder2" range="-85 60" stiffness="1" type="hinge"/>
                <geom fromto="0 0 0 .16 -.16 -.16" name="{prefix}right_uarm1" size="0.04 0.16" type="capsule" contype="1" conaffinity="1" rgba="{rgba}"/>
                <body name="{prefix}right_lower_arm" pos=".18 -.18 -.18">
                    <joint armature="0.0028" axis="0 -1 1" name="{prefix}right_elbow" range="-90 50" stiffness="0" type="hinge"/>
                    <geom fromto="0.01 0.01 0.01 .17 .17 .17" name="{prefix}right_larm" size="0.031" type="capsule" contype="1" conaffinity="1" rgba="{rgba}"/>
                    <geom name="{prefix}right_hand" pos=".18 .18 .18" size="0.04" type="sphere" contype="1" conaffinity="1" rgba="{rgba}"/>
                </body>
            </body>
            <body name="{prefix}left_upper_arm" pos="0 0.17 0.06">
                <joint armature="0.0068" axis="2 -1 1" name="{prefix}left_shoulder1" range="-60 85" stiffness="1" type="hinge"/>
                <joint armature="0.0051" axis="0 1 1" name="{prefix}left_shoulder2" range="-60 85" stiffness="1" type="hinge"/>
                <geom fromto="0 0 0 .16 .16 -.16" name="{prefix}left_uarm1" size="0.04 0.16" type="capsule" contype="1" conaffinity="1" rgba="{rgba}"/>
                <body name="{prefix}left_lower_arm" pos=".18 .18 -.18">
                    <joint armature="0.0028" axis="0 -1 -1" name="{prefix}left_elbow" range="-90 50" stiffness="0" type="hinge"/>
                    <geom fromto="0.01 -0.01 0.01 .17 -.17 .17" name="{prefix}left_larm" size="0.031" type="capsule" contype="1" conaffinity="1" rgba="{rgba}"/>
                    <geom name="{prefix}left_hand" pos=".18 -.18 .18" size="0.04" type="sphere" contype="1" conaffinity="1" rgba="{rgba}"/>
                </body>
            </body>
        </body>"""

def _actuators(prefix):
    joints = [("abdomen_y",100),("abdomen_z",100),("abdomen_x",100),
              ("right_hip_x",100),("right_hip_z",100),("right_hip_y",300),("right_knee",200),
              ("left_hip_x",100),("left_hip_z",100),("left_hip_y",300),("left_knee",200),
              ("right_shoulder1",25),("right_shoulder2",25),("right_elbow",25),
              ("left_shoulder1",25),("left_shoulder2",25),("left_elbow",25)]
    return "\n".join(f'        <motor gear="{g}" joint="{prefix}{j}" name="{prefix}{j}"/>' for j,g in joints)

def build_xml(config):
    r = config.ring_radius; h = config.ring_height
    a0 = _humanoid_body("a0_", -1.5, 0.0, "0.15 0.47 0.82 1.0", h)  # blue
    a1 = _humanoid_body("a1_",  1.5, 0.0, "0.85 0.22 0.18 1.0", h)  # red
    act0 = _actuators("a0_"); act1 = _actuators("a1_")
    return f"""<mujoco model="sumo_v2">
    <compiler angle="degree" inertiafromgeom="true"/>
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom condim="3" friction="1.5 .1 .1" contype="1" conaffinity="1" rgba="0.5 0.5 0.5 1.0"/>
        <motor ctrllimited="true" ctrlrange="-.4 .4"/>
    </default>
    <option iterations="8" timestep="0.003" gravity="0 0 -9.81"/>
    <custom>
        <numeric data="2500" name="constraint_limit_stiffness"/>
        <numeric data="27000" name="constraint_stiffness"/>
        <numeric data="30" name="constraint_ang_damping"/>
        <numeric data="80" name="constraint_vel_damping"/>
        <numeric data="-0.05" name="ang_damping"/>
        <numeric data="0.5" name="joint_scale_pos"/>
        <numeric data="0.1" name="joint_scale_ang"/>
        <numeric data="0" name="spring_mass_scale"/>
        <numeric data="1" name="spring_inertia_scale"/>
        <numeric data="20" name="matrix_inv_iterations"/>
        <numeric data="15" name="solver_maxls"/>
    </custom>
    <size nkey="5" nuser_geom="1"/>
    <visual>
        <quality shadowsize="4096"/>
        <map fogend="12" fogstart="8"/>
        <global offwidth="{config.width}" offheight="{config.height}"/>
    </visual>
    <asset>
        <texture builtin="gradient" height="100" rgb1="0.65 0.78 0.95" rgb2="0.90 0.92 1.0" type="skybox" width="100"/>
    </asset>
    <worldbody>
        <light cutoff="100" diffuse="0.95 0.92 0.85" dir="-0.3 -0.6 -1.3" directional="true" pos="0 0 8" specular="0.35 0.35 0.35"/>
        <light diffuse="0.35 0.38 0.45" dir="0.5 0.8 -0.5" directional="true" pos="0 -3 7"/>
        <!-- Ground below the ring -->
        <geom name="ground" type="plane" size="20 20 0.1" pos="0 0 0" rgba="0.5 0.48 0.45 1.0"
              conaffinity="1" contype="1" condim="3" friction="1 .1 .1"/>
        <!-- Elevated disc platform -->
        <body name="ring" pos="0 0 {h}">
            <geom name="ring_floor" type="box" size="{r} {r} 0.08"
                  rgba="0.72 0.68 0.62 1.0" conaffinity="1" contype="1"
                  condim="3" friction="1.5 .1 .1" mass="10000"/>
        </body>
{a0}
{a1}
    </worldbody>
    <actuator>
{act0}
{act1}
    </actuator>
</mujoco>"""


# ---------------------------------------------------------------------------
# Sumo Environment
# ---------------------------------------------------------------------------
class SumoEnv(PipelineEnv):
    def __init__(self, config=None, backend="positional", **kwargs):
        if config is None: config = Config()
        self._ring_r = config.ring_radius
        self._ring_h = config.ring_height
        xml = build_xml(config)
        sys = brax_mjcf.loads(xml)
        n_frames = 5
        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.0015})
            n_frames = 10
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng):
        qpos = self.sys.init_q; qvel = jnp.zeros(self.sys.qd_size())
        rng, rn = random.split(rng)
        qpos = qpos + random.uniform(rn, (self.sys.q_size(),), minval=-0.005, maxval=0.005)
        ps = self.pipeline_init(qpos, qvel)
        return State(ps, self._obs(ps), jnp.float32(0), jnp.float32(0), {})

    def step(self, state, action):
        ps = self.pipeline_step(state.pipeline_state, action)
        # Body indices: 0=ring, 1=a0_torso, 15=a1_torso (from XML order)
        a0 = ps.x.pos[1]; a1 = ps.x.pos[15]
        a0_dist = jnp.maximum(jnp.abs(a0[0]), jnp.abs(a0[1]))
        a1_dist = jnp.maximum(jnp.abs(a1[0]), jnp.abs(a1[1]))
        a0_h = a0[2]; a1_h = a1[2]
        rr = self._ring_r; rh = self._ring_h

        a0_off = jnp.logical_or(a0_dist > rr - 0.1, a0_h < rh - 0.5)
        a1_off = jnp.logical_or(a1_dist > rr - 0.1, a1_h < rh - 0.5)

        survive = jnp.float32(0.5)
        push = (a1_dist - a0_dist) * 0.3
        win = jnp.where(a1_off, jnp.float32(10.0), jnp.float32(0.0))
        lose = jnp.where(a0_off, jnp.float32(-10.0), jnp.float32(0.0))
        dist_between = jnp.sqrt(jnp.sum((a0[:2] - a1[:2])**2))
        approach = -dist_between * 0.1
        ctrl = -0.01 * jnp.sum(action**2)
        reward = survive + push + win + lose + approach + ctrl

        done = jnp.logical_or(a0_off, a1_off).astype(jnp.float32)
        done = jnp.maximum(done,
            jnp.logical_or(a0_h < rh + 0.3, a1_h < rh + 0.3).astype(jnp.float32))
        return state.replace(pipeline_state=ps, obs=self._obs(ps), reward=reward, done=done)

    def _obs(self, ps):
        return jnp.concatenate([ps.q, ps.qd])

    @property
    def observation_size(self): return self.sys.q_size() + self.sys.qd_size()
    @property
    def action_size(self): return 17 * 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

def train_env(config, env):
    ckpts, pm = [], {}; t0 = time.time()
    def pfn(ns, m):
        r = float(m.get("eval/episode_reward", 0)); pm[ns] = r
        print(f"  step={ns:>12,}  reward={r:>8.1f}  time={time.time()-t0:.0f}s")
    def ppfn(s, mp, p):
        ckpts.append(Checkpoint(s, 0.0, jax.device_get(p)))
        print(f"  [ckpt {len(ckpts)}] step={s:,}")
    tf = functools.partial(ppo.train, num_timesteps=config.num_timesteps,
        num_evals=config.num_evals, reward_scaling=0.1,
        episode_length=config.episode_length, normalize_observations=True,
        action_repeat=1, unroll_length=5, num_minibatches=32,
        num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4,
        entropy_cost=1e-2, num_envs=config.num_envs, batch_size=512, seed=0)
    print("\nTraining (JIT first)...\n")
    mif, fp, _ = tf(environment=env, progress_fn=pfn, policy_params_fn=ppfn)
    tt = time.time() - t0; print(f"\nDone in {tt:.0f}s ({len(ckpts)} ckpts)")
    for c in ckpts:
        cl = min(pm.keys(), key=lambda s: abs(s-c.step), default=0)
        if cl: c.reward = pm[cl]
    return ckpts, mif, fp, tt

def narrative(ckpts):
    n = len(ckpts)
    if n < 6: return [(c, f"Step {c.step:,}", 300) for c in ckpts]
    idx = [0, n//8, n//4, n//2, 3*n//4, n-1]
    lbl = ["First Encounters","Early Pushing","Learning to Fight",
           "Mid Training","Getting Aggressive","Trained Fighters"]
    stp = [200,200,300,300,400,500]
    return [(ckpts[i],l,s) for i,l,s in zip(idx,lbl,stp)]

def record(env, mif, params, num_steps, config, seed=42):
    inf = mif(params); ji = jax.jit(inf)
    jr = jax.jit(env.reset); js = jax.jit(env.step)
    key = random.PRNGKey(seed); st = jr(rng=key)
    states = [st.pipeline_state]; tr = 0.0; done_at = None
    for i in range(num_steps):
        key, sk = random.split(key); a = ji(st.obs, sk)
        act = a[0] if isinstance(a, tuple) else a
        st = js(st, act); states.append(st.pipeline_state); tr += float(st.reward)
        if float(st.done) and done_at is None: done_at = i
        if done_at is not None and (i - done_at) >= config.post_done_frames: break
    while len(states) < config.min_episode_frames: states.append(states[-1])
    return states, tr

def setup_renderer(config):
    xml = build_xml(config)
    mm = mj_lib.MjModel.from_xml_string(xml); md = mj_lib.MjData(mm)
    mm.vis.global_.offwidth = max(config.width, mm.vis.global_.offwidth)
    mm.vis.global_.offheight = max(config.height, mm.vis.global_.offheight)
    mm.vis.quality.shadowsize = 4096
    return mm, md, mj_lib.Renderer(mm, height=config.height, width=config.width)

def render_frame(mm, md, rr, ps, config):
    q = np.array(ps.q); qd = np.array(ps.qd)
    if len(q.shape)>1: q,qd = q[0],qd[0]
    md.qpos[:len(q)] = q; md.qvel[:len(qd)] = qd
    mj_lib.mj_forward(mm, md)
    cam = mj_lib.MjvCamera()
    cam.type = mj_lib.mjtCamera.mjCAMERA_FREE
    cam.lookat[0] = 0.0; cam.lookat[1] = 0.0; cam.lookat[2] = config.cam_lookat_z
    cam.distance = config.cam_distance
    cam.elevation = config.cam_elevation; cam.azimuth = config.cam_azimuth
    rr.update_scene(md, camera=cam)
    return rr.render().copy()

def compose(frame, label, info, config, progress=0.0):
    img = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0,0,0,0)); draw = ImageDraw.Draw(ov)
    fl,fs = _get_font(36),_get_font(28)
    bb = draw.textbbox((0,0),label,font=fl); tw,th = bb[2]-bb[0],bb[3]-bb[1]
    x = max(config.safe_left,(config.safe_left+config.safe_right-tw)//2); y = config.safe_top+16
    draw.rounded_rectangle([x-16,y-8,x+tw+16,y+th+8],radius=12,fill=(0,0,0,170))
    draw.text((x,y),label[:28],fill=(255,255,255,255),font=fl)
    bb2 = draw.textbbox((0,0),info,font=fs); tw2,th2 = bb2[2]-bb2[0],bb2[3]-bb2[1]
    x2 = max(config.safe_left,(config.safe_left+config.safe_right-tw2)//2); y2 = y+th+24
    draw.rounded_rectangle([x2-12,y2-6,x2+tw2+12,y2+th2+6],radius=10,fill=(0,0,0,140))
    draw.text((x2,y2),info[:35],fill=(210,210,210,255),font=fs)
    if progress>0:
        by=config.safe_top+4; bw=int((config.safe_right-config.safe_left)*min(progress,1.0))
        draw.rectangle([config.safe_left,by,config.safe_left+bw,by+3],fill=(80,200,120,200))
    return np.array(Image.alpha_composite(img,ov).convert("RGB"))

def title_card(lines, config, dur=2.5, fsizes=None, bg=(18,18,22)):
    n = int(dur*config.fps)
    card = np.full((config.height,config.width,3),bg,dtype=np.uint8)
    img = Image.fromarray(card); draw = ImageDraw.Draw(img)
    if fsizes is None: fsizes = [48]+[34]*(len(lines)-1)
    sp=28; ld=[]; th=0
    for t,fs in zip(lines,fsizes):
        f=_get_font(fs); bb=draw.textbbox((0,0),t,font=f)
        tw,h=bb[2]-bb[0],bb[3]-bb[1]; ld.append((t,f,tw,h)); th+=h
    th+=sp*(len(lines)-1); sy=(config.safe_top+config.safe_bottom-th)//2
    for t,f,tw,h in ld:
        x=max(config.safe_left,(config.width-tw)//2)
        draw.text((x,sy),t,fill=(255,255,255),font=f); sy+=h+sp
    return [np.array(img)]*n

def transition(dur, config):
    return [np.full((config.height,config.width,3),15,dtype=np.uint8)]*int(dur*config.fps)


def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    print("="*65)
    print("  AI Sumo Wrestling v2 — Two Humanoids, One Ring")
    print("="*65)
    print(f"  GPU: {jax.devices()[0]}")
    print(f"  Ring: {config.ring_radius}m radius, {config.ring_height}m high")
    print(f"  Training: {config.num_timesteps:,} steps, {config.num_envs} envs")
    print(f"  Backend: positional (GPU-accelerated)")
    print("="*65)

    env = SumoEnv(config, backend="positional")
    print(f"\n  obs={env.observation_size}, act={env.action_size}")
    envs.register_environment("sumo2", lambda **kw: SumoEnv(config, backend="positional", **kw))

    ckpts, mif, fp, tt = train_env(config, env)
    brax_model.save_params(os.path.join(config.output_dir, "sumo_v2_params"), fp)

    nar = narrative(ckpts)
    print(f"\nNarrative ({len(nar)} episodes):")
    for c,l,s in nar: print(f"  {l:25s} step={c.step:>10,} reward={c.reward:>8.1f}")

    out = os.path.join(config.output_dir, "sumo_v2.mp4")
    w = imageio.get_writer(out, fps=config.fps, codec="libx264", quality=9,
        pixelformat="yuv420p", macro_block_size=8,
        output_params=["-preset","slow","-crf","18"])
    mm, md, rr = setup_renderer(config); tf = 0

    for f in title_card(["AI Sumo Wrestling","","Two Humanoids","One Ring","","Vizuara | Brax PPO"],
        config, dur=3.0, fsizes=[52,20,42,42,20,28]):
        w.append_data(f); tf+=1

    for si,(ckpt,label,ms) in enumerate(nar):
        print(f"\n--- {si+1}/{len(nar)}: {label} ---")
        for f in transition(0.3,config): w.append_data(f); tf+=1
        states, ep_r = record(env, mif, ckpt.params, ms, config, seed=si*7+42)
        print(f"  {len(states)} steps, reward={ep_r:.1f}")
        prog = ckpt.step / config.num_timesteps
        info = f"Step {ckpt.step:,} | Reward {ckpt.reward:.0f}"
        fc=0; print(f"  Rendering {len(states)} frames...")
        for ps in states:
            raw = render_frame(mm, md, rr, ps, config)
            comp = compose(raw, label, info, config, prog)
            w.append_data(comp); fc+=1
        tf+=fc; print(f"  Wrote {fc} frames ({fc/config.fps:.1f}s)")
        del states

    rr.close()
    for f in transition(0.3,config): w.append_data(f); tf+=1
    for f in title_card([f"Trained in {tt:.0f}s",f"Reward: {ckpts[-1].reward:.0f}","",
        "30M steps | Self-Play PPO","Vizuara"],
        config, dur=3.0, fsizes=[38,44,20,28,28]):
        w.append_data(f); tf+=1
    for f in title_card(["GPU-Accelerated","Reinforcement Learning","","Vizuara"],
        config, dur=2.5, fsizes=[40,40,20,36]):
        w.append_data(f); tf+=1
    w.close()
    dur=tf/config.fps; sz=os.path.getsize(out)/(1024**2)
    print(f"\n{'='*65}\n  Video: {out}\n  Duration: {dur:.1f}s | Size: {sz:.1f}MB\n  Training: {tt:.0f}s\n{'='*65}")

if __name__ == "__main__":
    main()
