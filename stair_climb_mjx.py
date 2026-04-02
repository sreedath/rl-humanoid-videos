"""
Humanoid Walk — MuJoCo Playground + Custom Shorts Rendering
=============================================================
Step 1: Train with Playground's own script (returns make_inference_fn + params)
Step 2: Custom rendering with camera tracking + text overlays + Shorts format

This script patches the Playground training to capture checkpoints in-memory,
then renders the YouTube Shorts video.
"""

import os
import sys
import time
import functools
import warnings

import numpy as np

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jp
from jax import random
import mujoco
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from PIL import Image, ImageDraw, ImageFont
import imageio

# --- Config ---
WIDTH, HEIGHT, FPS = 1080, 1920, 50
CAM_DISTANCE = 8.0
CAM_ELEVATION = -10.0
CAM_AZIMUTH = 90.0
CAM_LOOKAT_Z = 1.2
OUTPUT_DIR = "/workspace/output"
TRAIN_STEPS = 30_000_000
NUM_EVALS = 20
ENV_NAME = "HumanoidWalk"


def _get_font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  Humanoid Walk — MuJoCo Playground (MJX)")
    print("=" * 65)
    print(f"  GPU: {jax.devices()[0]}")
    print(f"  Steps: {TRAIN_STEPS:,}")
    print("=" * 65)

    # --- Setup env (Playground's way) ---
    from mujoco_playground.config import dm_control_suite_params
    env_cfg = registry.get_default_config(ENV_NAME)
    env = registry.load(ENV_NAME, config=env_cfg)

    # --- PPO params (from Playground's dm_control config) ---
    ppo_cfg = dm_control_suite_params.brax_ppo_config(ENV_NAME)

    # Checkpoint capture
    checkpoints = []
    rewards = {}
    t0 = time.time()

    def progress_fn(num_steps, metrics):
        r = float(metrics.get("eval/episode_reward", 0))
        rewards[num_steps] = r
        print(f"  step={num_steps:>12,}  reward={r:>8.1f}  time={time.time()-t0:.0f}s")

    def policy_params_fn(step, make_policy, params):
        checkpoints.append({"step": step, "params": jax.device_get(params)})
        print(f"  [ckpt {len(checkpoints)}] step={step:,}")

    # --- Train ---
    print("\nTraining...\n")
    # Override config with our values
    ppo_cfg["num_timesteps"] = TRAIN_STEPS
    ppo_cfg["num_evals"] = NUM_EVALS
    ppo_cfg["seed"] = 1
    # Use wrap_env_fn like Playground does (avoids PRNG issues)
    make_inference_fn, final_params, _ = ppo.train(
        environment=env,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        **ppo_cfg,
    )
    # Wrap env for inference rollouts
    brax_env = wrapper.wrap_for_brax_training(
        env, episode_length=env_cfg.episode_length,
        action_repeat=env_cfg.get("action_repeat", 1),
    )
    train_time = time.time() - t0
    print(f"\nDone in {train_time:.0f}s ({len(checkpoints)} ckpts)")

    # Backfill rewards
    for c in checkpoints:
        cl = min(rewards.keys(), key=lambda s: abs(s - c["step"]), default=0)
        if cl: c["reward"] = rewards[cl]
        else: c["reward"] = 0.0

    # --- Select narrative ---
    n = len(checkpoints)
    if n >= 6:
        idx = [0, n//6, n//3, n//2, 2*n//3, n-1]
        lbl = ["Episode 1", "Early Training", "Learning to Walk",
               "Mid Training", "Getting Better", "Trained Walker"]
    else:
        idx = list(range(n))
        lbl = [f"Checkpoint {i}" for i in range(n)]
    narrative = [(checkpoints[i], lbl[j]) for j, i in enumerate(idx)]

    # --- Rollouts ---
    print(f"\n{'='*65}\n  Running rollouts\n{'='*65}")
    rollouts = []
    for ckpt, label in narrative:
        print(f"\n--- {label} (step {ckpt['step']:,}) ---")
        inf_fn = make_inference_fn(ckpt["params"], deterministic=True)
        jit_inf = jax.jit(inf_fn)
        jit_step = jax.jit(brax_env.step)

        # Reset with batched key (batch size 1) then unbatch
        rng = random.split(random.PRNGKey(42), 1)  # shape (1, 2)
        state = jax.jit(brax_env.reset)(rng)

        qpos_list = [np.array(state.data.qpos)]
        qvel_list = [np.array(state.data.qvel)]
        total_r = 0.0; done_at = None

        for i in range(600):
            rng, sk = random.split(rng[0])
            rng = rng.reshape(1, -1)
            act_key = random.split(sk, 1)
            act = jax.vmap(jit_inf)(state.obs, act_key)[0]
            state = jit_step(state, act)
            qpos_list.append(np.array(state.data.qpos))
            qvel_list.append(np.array(state.data.qvel))
            total_r += float(state.reward.sum())
            if float(state.done.sum()) > 0 and done_at is None: done_at = i
            if done_at is not None and (i - done_at) >= 40: break

        while len(qpos_list) < 100:
            qpos_list.append(qpos_list[-1])
            qvel_list.append(qvel_list[-1])

        rollouts.append({"qpos": qpos_list, "qvel": qvel_list,
                         "label": label, "step": ckpt["step"],
                         "reward": total_r})
        print(f"  {len(qpos_list)} frames, reward={total_r:.1f}")

    # --- Render YouTube Shorts ---
    print(f"\n{'='*65}\n  Rendering video\n{'='*65}")
    mj_model = env.mj_model; mj_data = mujoco.MjData(mj_model)
    mj_model.vis.global_.offwidth = max(WIDTH, mj_model.vis.global_.offwidth)
    mj_model.vis.global_.offheight = max(HEIGHT, mj_model.vis.global_.offheight)
    mj_model.vis.quality.shadowsize = 4096
    for i in range(mj_model.ngeom):
        nm = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if nm and "floor" in nm.lower(): mj_model.geom_matid[i] = -1

    rr = mujoco.Renderer(mj_model, height=HEIGHT, width=WIDTH)
    out = f"{OUTPUT_DIR}/humanoid_walk_shorts.mp4"
    w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=9,
        pixelformat="yuv420p", macro_block_size=8,
        output_params=["-preset","slow","-crf","18"])
    tf = 0

    # Intro
    for f in _tc(["Can an AI Learn","to WALK?","","MuJoCo MJX | PPO","","Vizuara"]):
        w.append_data(f); tf+=1

    for ro in rollouts:
        print(f"  Rendering: {ro['label']} ({len(ro['qpos'])} frames)")
        for f in _tr(0.3): w.append_data(f); tf+=1
        info = f"Step {ro['step']:,} | Reward {ro['reward']:.0f}"
        prog = ro["step"] / TRAIN_STEPS
        for qi in range(len(ro["qpos"])):
            qp = np.array(ro["qpos"][qi]).flatten()
            qv = np.array(ro["qvel"][qi]).flatten()
            # Handle batched (1, nq) -> (nq,)
            if len(qp) > mj_model.nq: qp = qp[:mj_model.nq]
            if len(qv) > mj_model.nv: qv = qv[:mj_model.nv]
            mj_data.qpos[:len(qp)]=qp; mj_data.qvel[:len(qv)]=qv
            mujoco.mj_forward(mj_model, mj_data)
            torso = mj_data.body("torso").xpos
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[0]=float(torso[0]); cam.lookat[1]=float(torso[1])
            cam.lookat[2]=CAM_LOOKAT_Z; cam.distance=CAM_DISTANCE
            cam.elevation=CAM_ELEVATION; cam.azimuth=CAM_AZIMUTH
            rr.update_scene(mj_data, camera=cam)
            frame = rr.render().copy()
            frame = _co(frame, ro["label"], info, prog)
            w.append_data(frame); tf+=1

    rr.close()
    for f in _tr(0.3): w.append_data(f); tf+=1
    fr = rollouts[-1]["reward"]
    for f in _tc([f"Trained in {train_time:.0f}s",f"Reward: {fr:.0f}","",
                  f"{TRAIN_STEPS//1_000_000}M steps | MuJoCo MJX","Vizuara"],
                 fsizes=[38,44,20,28,28]):
        w.append_data(f); tf+=1
    for f in _tc(["GPU-Accelerated","Reinforcement Learning","","Vizuara"],
                 fsizes=[40,40,20,36]):
        w.append_data(f); tf+=1
    w.close()
    dur=tf/FPS; sz=os.path.getsize(out)/(1024**2)
    print(f"\n{'='*65}\n  Video: {out}\n  Duration: {dur:.1f}s | Size: {sz:.1f}MB\n  Training: {train_time:.0f}s\n{'='*65}")


def _co(frame, label, info, progress=0.0):
    st,sb,sl,sr = 300,1248,100,980
    img = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA",img.size,(0,0,0,0)); draw=ImageDraw.Draw(ov)
    fl,fs = _get_font(36),_get_font(28)
    bb=draw.textbbox((0,0),label,font=fl); tw,th=bb[2]-bb[0],bb[3]-bb[1]
    x=max(sl,(sl+sr-tw)//2); y=st+16
    draw.rounded_rectangle([x-16,y-8,x+tw+16,y+th+8],radius=12,fill=(0,0,0,170))
    draw.text((x,y),label[:28],fill=(255,255,255,255),font=fl)
    bb2=draw.textbbox((0,0),info,font=fs); tw2=bb2[2]-bb2[0]; th2=bb2[3]-bb2[1]
    x2=max(sl,(sl+sr-tw2)//2); y2=y+th+24
    draw.rounded_rectangle([x2-12,y2-6,x2+tw2+12,y2+th2+6],radius=10,fill=(0,0,0,140))
    draw.text((x2,y2),info[:35],fill=(210,210,210,255),font=fs)
    if progress>0:
        by=st+4; bw=int((sr-sl)*min(progress,1.0))
        draw.rectangle([sl,by,sl+bw,by+3],fill=(80,200,120,200))
    return np.array(Image.alpha_composite(img,ov).convert("RGB"))

def _tc(lines, dur=3.0, fsizes=None, bg=(18,18,22)):
    n=int(dur*FPS); card=np.full((HEIGHT,WIDTH,3),bg,dtype=np.uint8)
    img=Image.fromarray(card); draw=ImageDraw.Draw(img)
    if fsizes is None: fsizes=[44,52,20,30,20,36]
    sp=28; ld=[]; th=0
    for t,fs in zip(lines,fsizes):
        f=_get_font(fs); bb=draw.textbbox((0,0),t,font=f)
        tw,h=bb[2]-bb[0],bb[3]-bb[1]; ld.append((t,f,tw,h)); th+=h
    th+=sp*(len(lines)-1); sy=(300+1248-th)//2
    for t,f,tw,h in ld:
        x=max(100,(WIDTH-tw)//2); draw.text((x,sy),t,fill=(255,255,255),font=f); sy+=h+sp
    return [np.array(img)]*n

def _tr(dur):
    return [np.full((HEIGHT,WIDTH,3),15,dtype=np.uint8)]*int(dur*FPS)


if __name__ == "__main__":
    main()
