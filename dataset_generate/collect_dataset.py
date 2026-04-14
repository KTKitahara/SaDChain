"""collect_dataset.py

ONLY generate universal step-only dataset (NO TRAINING).

Output layout:
  <out_root>/<split>/ep_<seed>/step_XXXX.txt

Each step_XXXX.txt is written by env.step() using the universal format:
  line1: N
  for each node i=0..N-1, 5 lines:
    id
    x
    y
    computing_capacity
    transmit_power

This script is dedicated to simulating environment evolution and generating datasets.
Training must be done separately by offline_dqn_train.py (replaying ONLY step files).

Key knobs (explicitly exposed):
  - number of episodes (train/test)
  - steps per episode
  - number of nodes
  - max_shards (K upper bound)
  - Pt write unit switch:
      pt_write_flag=0  -> write Pt in W to txt
      pt_write_flag=1  -> write Pt in dBm to txt
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

from BC_env_Mal_new_norm import BCnetenv

# =========================
# IDE 运行参数区
# =========================
USE_IDE_DEFAULTS = True  # True: 忽略命令行参数，直接用下面的配置；False: 用命令行 argparse

IDE_CFG = dict(
    out_root="dataset_200",
    train_episodes=1,
    test_episodes=10,
    steps_per_episode=10,

    nb_nodes=200,
    max_shards=40,
    min_nodes_per_shard=5,

    pt_write_flag=0,  # 0: 写W；1: 写dBm
    seed_base=1,      # 可选：生成 episode seed 的起始偏移
)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


@dataclass
class CollectConfig:
    out_root: str

    # ===== explicitly exposed knobs =====
    train_episodes: int
    test_episodes: int
    steps_per_episode: int
    nb_nodes: int

    min_nodes_per_shard: int
    max_shards: int

    # 0 -> write W, 1 -> write dBm
    pt_write_flag: int

    # seeds
    train_base_seed: int
    test_base_seed: int

    # behavior policy randomness
    behavior_seed: int


def _make_env(cfg: CollectConfig) -> BCnetenv:
    pt_mode: Literal["w", "dbm"] = "dbm" if int(cfg.pt_write_flag) == 1 else "w"

    # Prefer passing max_shards + pt_write_mode if supported.
    try:
        return BCnetenv(
            nb_nodes=int(cfg.nb_nodes),
            min_nodes_per_shard=int(cfg.min_nodes_per_shard),
            max_shards=int(cfg.max_shards),
            pt_write_mode=pt_mode,
        )
    except TypeError:
        # Fallback for older env signatures.
        return BCnetenv(nb_nodes=int(cfg.nb_nodes), min_nodes_per_shard=int(cfg.min_nodes_per_shard))


def _collect_split(split: str, n_episodes: int, base_seed: int, cfg: CollectConfig) -> None:
    env = _make_env(cfg)

    # One RNG controlling behavior-policy randomness (actions) for this split.

    for ep in range(int(n_episodes)):
        scenario_seed = int(base_seed) + int(ep)

        # reset() creates ep directory; step() writes step_XXXX.txt
        _ = env.reset(scenario_seed=scenario_seed, export_root=cfg.out_root, split=split)

        # Derive a per-episode RNG so behavior is reproducible per (behavior_seed, scenario_seed)
        local_seed = (int(cfg.behavior_seed) * 1315423911 + int(scenario_seed)) & 0xFFFFFFFF
        local_rng = np.random.default_rng(int(local_seed))

        for _t in range(int(cfg.steps_per_episode)):
            # Sample only valid shard-count actions: action in [0, max_shards-1] => K in [1, max_shards]
            if int(cfg.max_shards) <= 0:
                raise ValueError("max_shards must be > 0")
            action = int(local_rng.integers(0, int(cfg.max_shards)))

            _next_state, _reward, _done, _const, _info = env.step(action)

        ep_dir = os.path.join(cfg.out_root, split, f"ep_{scenario_seed:05d}")
        print(f"[collect] {split} ep_{scenario_seed:05d}: steps={cfg.steps_per_episode} -> {ep_dir}")


def collect_all(cfg: CollectConfig) -> None:
    ensure_dir(os.path.join(cfg.out_root, "train"))
    ensure_dir(os.path.join(cfg.out_root, "test"))

    # Print the explicit knobs once (easy to verify)
    print("\n=== collect_dataset.py (NO TRAINING) ===")
    print(f"out_root            : {cfg.out_root}")
    print(f"train_episodes      : {cfg.train_episodes} (base_seed={cfg.train_base_seed})")
    print(f"test_episodes       : {cfg.test_episodes} (base_seed={cfg.test_base_seed})")
    print(f"steps_per_episode   : {cfg.steps_per_episode}")
    print(f"nb_nodes            : {cfg.nb_nodes}")
    print(f"min_nodes_per_shard : {cfg.min_nodes_per_shard}")
    print(f"max_shards          : {cfg.max_shards}")
    print(f"pt_write_flag       : {cfg.pt_write_flag}  (0->W, 1->dBm)")
    print(f"behavior_seed       : {cfg.behavior_seed}\n")

    _collect_split("train", cfg.train_episodes, cfg.train_base_seed, cfg)
    _collect_split("test", cfg.test_episodes, cfg.test_base_seed, cfg)


def main() -> None:
    ap = argparse.ArgumentParser()

    # ===== explicitly exposed knobs =====
    ap.add_argument("--train_episodes", type=int, default=100, help="# train episodes to generate")
    ap.add_argument("--test_episodes", type=int, default=20, help="# test episodes to generate")
    ap.add_argument("--steps_per_episode", type=int, default=10, help="# steps per episode")
    ap.add_argument("--nb_nodes", type=int, default=200, help="# nodes N")

    ap.add_argument("--min_nodes_per_shard", type=int, default=5)
    ap.add_argument("--max_shards", type=int, default=40, help="K upper bound; actions sampled in [0,K-1]")

    # Pt write unit switch
    ap.add_argument(
        "--pt_write_flag",
        type=int,
        default=0,
        choices=[0, 1],
        help="0: write Pt in W to txt; 1: write Pt in dBm to txt",
    )

    # output
    ap.add_argument("--out_root", default="dataset", help="dataset root folder")

    # episode seeds
    ap.add_argument("--train_base_seed", type=int, default=0, help="scenario_seed for train starts from here")
    ap.add_argument("--test_base_seed", type=int, default=0, help="scenario_seed for test starts from here")

    # behavior
    ap.add_argument("--behavior_seed", type=int, default=12345, help="seed controlling action sampling")

    args = ap.parse_args()
    if USE_IDE_DEFAULTS:
        # 覆盖 argparse 解析结果，让 IDE_CFG 生效
        for k, v in IDE_CFG.items():
            if k == "seed_base":
                continue
            if not hasattr(args, k):
                raise AttributeError(f"IDE_CFG key '{k}' not a valid argparse arg name")
            setattr(args, k, v)

        # seed_base 映射到 train/test_base_seed
        seed_base = int(IDE_CFG.get("seed_base", 0))
        args.train_base_seed = seed_base
        args.test_base_seed = 20000 + seed_base
 

    cfg = CollectConfig(
        out_root=str(args.out_root),
        train_episodes=int(args.train_episodes),
        test_episodes=int(args.test_episodes),
        steps_per_episode=int(args.steps_per_episode),
        nb_nodes=int(args.nb_nodes),
        min_nodes_per_shard=int(args.min_nodes_per_shard),
        max_shards=int(args.max_shards),
        pt_write_flag=int(args.pt_write_flag),
        train_base_seed=int(args.train_base_seed),
        test_base_seed=int(args.test_base_seed),
        behavior_seed=int(args.behavior_seed),
    )

    collect_all(cfg)


if __name__ == "__main__":
    main()
