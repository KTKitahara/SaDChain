# BC_env_Mal_new_norm.py
# Online env + universal snapshot writer.
#
# User requirement:
# - Online env generates snapshots AND computes state/reward using the SAME formulas/params as BC_env_offline.py.
# - Only difference vs offline: offline reads step_XXXX.txt; online produces & saves them.
#
# Universal step format:
#   step_XXXX.txt:
#     line1: N
#     then for each node i (i=0..N-1) write 5 lines:
#       id
#       x
#       y
#       computing_capacity
#       transmit_power
#
# DQN state is 4 channels: [R, C, H, e_prob]
# - R: Friis + Shannon directional rate matrix (bps)
# - C: compute matrix (tile of c_i)
# - H: topology matrix (1/dist)
# - e_prob: risk probability matrix (tile of Risk_i)

from __future__ import annotations

import os
import random
from typing import List

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

try:
    from scipy.spatial.distance import cdist
except Exception:
    cdist = None


def dbm_to_w(dbm: float) -> float:
    return 10.0 ** ((float(dbm) - 30.0) / 10.0)


def w_to_dbm(w: float) -> float:
    w = max(float(w), 1e-30)
    return 10.0 * np.log10(w * 1000.0)


def _truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    size,
) -> np.ndarray:
    """Simple truncated normal sampler (rejection) for small sizes."""
    out = np.empty(size, dtype=np.float64)
    it = np.nditer(out, flags=["multi_index"], op_flags=["writeonly"])
    while not it.finished:
        v = rng.normal(mean, std)
        while v < low or v > high:
            v = rng.normal(mean, std)
        it[0] = v
        it.iternext()
    return out


def deterministic_shards(episode_id: int, step_idx: int, N: int, K: int) -> List[List[int]]:
    """Reproducible balanced sharding derived ONLY from (episode_id, step_idx, K)."""
    seed = (int(episode_id) * 1000003 + int(step_idx) * 10007 + int(K) * 101) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    order = np.arange(N, dtype=np.int32)
    rng.shuffle(order)

    shards: List[List[int]] = [[] for _ in range(K)]
    for i, nid in enumerate(order):
        shards[i % K].append(int(nid))
    return shards


class BC_Env(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        nb_nodes: int = 200,
        min_nodes_per_shard: int = 5,
        max_shards: int = 40,
        pt_write_mode: str = "w",  # "w" or "dbm"
    ):
        super().__init__()

        # ===== scenario parameters =====
        self.nb_nodes = int(nb_nodes)
        self.min_nodes_per_shard = int(min_nodes_per_shard)
        self.max_shards = int(max_shards)
        self.pt_write_mode = str(pt_write_mode).lower().strip()
        if self.pt_write_mode not in ("w", "dbm"):
            self.pt_write_mode = "w"

        # ===== dataset export routing =====
        self.export_root = "dataset"
        self.split = "train"
        self.episode_id = 0
        self.ep_dir = None
        self.current_step = 0

        # ===== action space: choose number of shards (K = action+1) =====
        self.action_space = spaces.Discrete(int(self.nb_nodes))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4, self.nb_nodes, self.nb_nodes),
            dtype=np.float32,
        )

        # ===== Markov transition probability =====
        self.trans_prob = 0.1

        # ===== compute capability distribution (Hz) =====
        self.c_mu = 20e9
        self.c_sigma = 5e9
        self.c_min = 10e9
        self.c_max = 30e9
        self.c_step = 5e9

        # ===== transmit power distribution (dBm) =====
        self.pt_mu_dbm = 10.0
        self.pt_sigma_dbm = 2.0
        self.pt_min_dbm = 1.0
        self.pt_max_dbm = 24.0
        self.pt_step_dbm = 2.0

        # ===== positions (meters) =====
        self.xy_min = 0.0
        self.xy_max = 999.0
        self.xy_step = 5.0

        # ===== physics / comm (offline-aligned defaults) =====
        self.B_hz = 20e6
        self.f_c_hz = 868e6
        self.lambda_m = 3e8 / self.f_c_hz
        self.Gt = 1.0
        self.Gr = 1.0
        self.noise_w = 1e-13  # -100 dBm

        # ===== consensus / TPS (offline-aligned defaults) =====
        self.T_max = 100.0
        self.r_round = 1000
        self.SB_bytes = 8_000_000
        self.ST_bytes = 64
        self.M_batch = 3
        self.alpha = 2.0
        self.beta = 1.0

        # ===== security penalty (offline-aligned defaults) =====
        self.c_xi = 5e9
        self.lam_sec = 1.0
        self.gamma_sec = 1e6

        # gym RNG
        self.seed()

        # snapshot variables
        self.node_xy = None  # (N,2)
        self.c_vec = None    # (N,1)
        self.Pt_nodes_dbm = None  # (N,)
        self.Pt_nodes_w = None    # (N,)
        self.dist_matrix = None

        # state matrices
        self.R_mat = None
        self.C_mat = None
        self.H_mat = None
        self.E_mat = None

        self.state = None
        self.reward = 0.0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    # ---------------- state builders (same as offline env) ----------------
    def _cdist(self, xy: np.ndarray) -> np.ndarray:
        if cdist is not None:
            d = cdist(xy, xy).astype(np.float64)
        else:
            diff = xy[:, None, :] - xy[None, :, :]
            d = np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float64)
        d[d <= 1e-6] = 1e-6
        np.fill_diagonal(d, 1e-6)
        return d

    def _build_R_from_snapshot(self, dist: np.ndarray, pt_w: np.ndarray) -> np.ndarray:
        N = self.nb_nodes
        R = np.zeros((N, N), dtype=np.float64)

        lam = float(self.lambda_m)
        B = float(self.B_hz)
        N0 = max(float(self.noise_w), 1e-30)
        Gt = float(self.Gt)
        Gr = float(self.Gr)

        for i in range(N):
            pti = float(pt_w[i])
            if (not np.isfinite(pti)) or pti <= 0.0:
                pti = dbm_to_w(14.0)

            gain = (lam / (4.0 * np.pi * dist[i, :])) ** 2
            Pr = pti * Gt * Gr * gain
            snr = Pr / N0
            snr[snr < 0.0] = 0.0
            R[i, :] = B * np.log2(1.0 + snr)

        np.fill_diagonal(R, 0.0)
        R[~np.isfinite(R)] = 0.0
        return R

    def _build_C_matrix(self, c_vec: np.ndarray) -> np.ndarray:
        v = np.asarray(c_vec, dtype=np.float64).reshape(self.nb_nodes, 1)
        return np.tile(v, (1, self.nb_nodes)).astype(np.float64)

    def _build_H_matrix(self, dist: np.ndarray) -> np.ndarray:
        H = 1.0 / dist
        np.fill_diagonal(H, 0.0)
        H[~np.isfinite(H)] = 0.0
        return H.astype(np.float64)

    def _build_E_matrix(self, c_vec: np.ndarray) -> np.ndarray:
        c_i = np.asarray(c_vec, dtype=np.float64).reshape(-1)
        risk = float(self.c_xi) / (c_i + float(self.c_xi))
        risk[~np.isfinite(risk)] = 0.0
        risk = np.clip(risk, 0.0, 1.0)
        return np.tile(risk.reshape(self.nb_nodes, 1), (1, self.nb_nodes)).astype(np.float64)

    def _rebuild_state_from_snapshot(self) -> None:
        assert self.node_xy is not None and self.c_vec is not None and self.Pt_nodes_w is not None
        self.dist_matrix = self._cdist(self.node_xy)
        self.R_mat = self._build_R_from_snapshot(self.dist_matrix, self.Pt_nodes_w)
        self.C_mat = self._build_C_matrix(self.c_vec)
        self.H_mat = self._build_H_matrix(self.dist_matrix)
        self.E_mat = self._build_E_matrix(self.c_vec)
        self.state = [self.R_mat, self.C_mat, self.H_mat, self.E_mat]

    # ---------------- gym API ----------------
    def reset(self, *, scenario_seed: int = 0, export_root: str | None = None, split: str | None = None):
        """Collector interface: pre_state = env.reset(scenario_seed=seed, export_root=..., split=...)"""
        self.current_step = 0

        if export_root is not None:
            self.export_root = str(export_root)
        if split is not None:
            self.split = str(split)

        self.episode_id = int(scenario_seed)
        self.ep_dir = os.path.join(self.export_root, self.split, f"ep_{self.episode_id:05d}")
        os.makedirs(self.ep_dir, exist_ok=True)

        rng = np.random.default_rng(int(scenario_seed))

        # init xy
        self.node_xy = rng.uniform(self.xy_min, self.xy_max, size=(self.nb_nodes, 2)).astype(np.float64)

        # init c_vec (truncated normal + quantize to step for stable random walk)
        c_vec = _truncated_normal(rng, self.c_mu, self.c_sigma, self.c_min, self.c_max, size=(self.nb_nodes, 1))
        c_ghz = c_vec / 1e9
        c_ghz_q = np.round(c_ghz / (self.c_step / 1e9)) * (self.c_step / 1e9)
        c_ghz_q = np.clip(c_ghz_q, self.c_min / 1e9, self.c_max / 1e9)
        self.c_vec = (c_ghz_q * 1e9).astype(np.float64)

        # init Pt in dBm then derive W
        pt_dbm = _truncated_normal(rng, self.pt_mu_dbm, self.pt_sigma_dbm, self.pt_min_dbm, self.pt_max_dbm, size=(self.nb_nodes,))
        pt_dbm_q = np.round(pt_dbm / self.pt_step_dbm) * self.pt_step_dbm
        pt_dbm_q = np.clip(pt_dbm_q, self.pt_min_dbm, self.pt_max_dbm)
        self.Pt_nodes_dbm = pt_dbm_q.astype(np.float64)
        self.Pt_nodes_w = (10.0 ** ((self.Pt_nodes_dbm - 30.0) / 10.0)).astype(np.float64)

        # build state matrices from snapshot
        self._rebuild_state_from_snapshot()
        return self.state

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Invalid action: {action}")

        # decode action -> n_shard
        n_shard = int(action) + 1
        invalid_action = bool(n_shard < 1 or n_shard > int(self.max_shards) or n_shard > int(self.nb_nodes))

        # advance Markov snapshot (online-only; offline reads next snapshot)
        # update c_vec (random walk)
        for i in range(self.nb_nodes):
            rnum = random.random()
            v = float(self.c_vec[i, 0])

            if v <= self.c_min:
                if rnum < self.trans_prob:
                    v += self.c_step
            elif v >= self.c_max:
                if rnum < self.trans_prob:
                    v -= self.c_step
            else:
                if rnum < self.trans_prob:
                    v += self.c_step
                elif rnum > 1 - self.trans_prob:
                    v -= self.c_step

            v = float(np.clip(v, self.c_min, self.c_max))
            self.c_vec[i, 0] = v

        # update xy (random walk)
        for i in range(self.nb_nodes):
            for d in range(2):
                rnum = random.random()
                v = float(self.node_xy[i, d])

                if v <= self.xy_min:
                    if rnum < self.trans_prob:
                        v += self.xy_step
                elif v >= self.xy_max:
                    if rnum < self.trans_prob:
                        v -= self.xy_step
                else:
                    if rnum < self.trans_prob:
                        v += self.xy_step
                    elif rnum > 1 - self.trans_prob:
                        v -= self.xy_step

                self.node_xy[i, d] = float(np.clip(v, self.xy_min, self.xy_max))

        # update Pt dBm (random walk)
        for i in range(self.nb_nodes):
            rnum = random.random()
            v = float(self.Pt_nodes_dbm[i])

            if v <= self.pt_min_dbm:
                if rnum < self.trans_prob:
                    v += self.pt_step_dbm
            elif v >= self.pt_max_dbm:
                if rnum < self.trans_prob:
                    v -= self.pt_step_dbm
            else:
                if rnum < self.trans_prob:
                    v += self.pt_step_dbm
                elif rnum > 1 - self.trans_prob:
                    v -= self.pt_step_dbm

            self.Pt_nodes_dbm[i] = float(np.clip(v, self.pt_min_dbm, self.pt_max_dbm))

        self.Pt_nodes_w = (10.0 ** ((self.Pt_nodes_dbm - 30.0) / 10.0)).astype(np.float64)

        # rebuild DQN-visible state from snapshot only (offline-aligned)
        self._rebuild_state_from_snapshot()

        # step index (1-based) used in deterministic sharding
        step_idx_1based = int(self.current_step + 1)

        # shard assignment (reproducible, no extra randomness)
        NodesInShard = deterministic_shards(self.episode_id, step_idx_1based, self.nb_nodes, n_shard) if not invalid_action else [[] for _ in range(max(1, n_shard))]

        # constraints
        done_min = False
        if (not invalid_action) and self.min_nodes_per_shard > 0:
            done_min = any(len(NodesInShard[k]) < self.min_nodes_per_shard for k in range(n_shard))

        done = bool(invalid_action or done_min)

        # shard_id array
        shard_id = np.full((self.nb_nodes,), -1, dtype=np.int16)
        if not invalid_action:
            for k in range(n_shard):
                for nid in NodesInShard[k]:
                    shard_id[int(nid)] = int(k)

        # metrics default
        TPS = float("nan")
        T_round = float("nan")
        T_epoch = float("nan")
        Treco = float("nan")
        sec_penalty = float("nan")
        sec_var = float("nan")
        sec_primary = float("nan")

        # invalid action -> early terminate with reward=0
        if invalid_action:
            reward = 0.0
        else:
            # ---------------- TPS computation (same as offline env) ----------------
            SB = float(self.SB_bytes)
            ST = float(self.ST_bytes)
            r_round = float(self.r_round)
            M = float(self.M_batch)
            alpha = float(self.alpha)
            beta = float(self.beta)
            T_MAX = float(self.T_max)

            SB_bits = SB * 8.0

            def get_rate(a: int, b: int) -> float:
                r = float(self.R_mat[a, b])
                if (not np.isfinite(r)) or r <= 1e-12:
                    return 1e-12
                return r

            def estimate_Tcon_k(pmac_id: int, team: List[int]) -> float:
                nodes = [int(x) for x in team]
                followers = [n for n in nodes if n != int(pmac_id)]
                g = int(len(nodes))
                if g <= 1:
                    return 0.0

                # propagation terms
                t_pp = {nid: 0.0 for nid in nodes}
                for fid in followers:
                    t_pp[fid] += SB_bits / get_rate(int(pmac_id), int(fid))

                t_pre = {nid: 0.0 for nid in nodes}
                for s in nodes:
                    for r in nodes:
                        if r == s:
                            continue
                        t_pre[r] += SB_bits / get_rate(int(s), int(r))

                t_com = {nid: 0.0 for nid in nodes}
                for s in nodes:
                    for r in nodes:
                        if r == s:
                            continue
                        t_com[r] += SB_bits / get_rate(int(s), int(r))

                T_prop = float(max(t_pp.values()) + max(t_pre.values()) + max(t_com.values()))

                # validation terms
                C_primary = M * alpha + (2.0 * M + 4.0 * float(g - 1)) * beta
                C_replica = M * alpha + (1.0 * M + 4.0 * float(g - 1)) * beta

                cal_p = float(self.c_vec[int(pmac_id), 0])
                if (not np.isfinite(cal_p)) or cal_p <= 1.0:
                    cal_p = 1.0
                T_cal_primary = C_primary / cal_p

                T_cal_rep_max = 0.0
                for fid in followers:
                    cal_i = float(self.c_vec[int(fid), 0])
                    if (not np.isfinite(cal_i)) or cal_i <= 1.0:
                        cal_i = 1.0
                    T_cal_rep_max = max(T_cal_rep_max, C_replica / cal_i)

                T_val = float(max(T_cal_primary, T_cal_rep_max))
                return float(min(T_prop, T_MAX) + min(T_val, T_MAX))

            def estimate_Tlc(primaries: List[int]) -> float:
                if len(primaries) <= 1:
                    return 0.0
                max_t = 0.0
                for i1 in range(len(primaries)):
                    for i2 in range(len(primaries)):
                        if i1 == i2:
                            continue
                        a = int(primaries[i1])
                        b = int(primaries[i2])
                        max_t = max(max_t, SB_bits / get_rate(a, b))
                return float(max_t)

            max_Tcon = 0.0
            primaries: List[int] = []
            Tcon_list: List[float] = []

            for k in range(n_shard):
                team = NodesInShard[k]
                if len(team) == 0:
                    continue
                pmac = int(max(team, key=lambda nid: float(self.c_vec[int(nid), 0])))
                primaries.append(pmac)
                tcon = estimate_Tcon_k(pmac, team)
                Tcon_list.append(tcon)
                if tcon > max_Tcon:
                    max_Tcon = float(tcon)

            n_eff_shard = int(len(Tcon_list)) if len(Tcon_list) > 0 else int(n_shard)
            T_lc = estimate_Tlc(primaries)
            T_algo = 0.0

            Treco = 3.0 * float(T_lc) + float(max_Tcon) + float(T_algo)
            T_epoch = float(r_round) * float(max_Tcon) + float(Treco)
            if T_epoch <= 1e-12:
                T_epoch = 1e-12

            TPS = float(n_eff_shard) * (float(r_round) * (float(SB) / float(ST))) / float(T_epoch)
            T_round = float(max_Tcon)

            # ---------------- security penalty (same as offline env) ----------------
            c_i = np.asarray(self.c_vec[:, 0], dtype=np.float64).reshape(-1)
            risk = float(self.c_xi) / (c_i + float(self.c_xi))
            risk[~np.isfinite(risk)] = 0.0
            risk = np.clip(risk, 0.0, 1.0)

            r_i_list: List[float] = []
            for k in range(n_shard):
                nodes = NodesInShard[k]
                if len(nodes) == 0:
                    r_i_list.append(0.0)
                else:
                    r_i_list.append(float(np.mean(risk[np.asarray(nodes, dtype=np.int32)])))

            r_i_arr = np.asarray(r_i_list, dtype=np.float64)
            r_bar = float(np.mean(r_i_arr)) if r_i_arr.size > 0 else 0.0
            sec_var = float(np.mean((r_i_arr - r_bar) ** 2)) if r_i_arr.size > 0 else 0.0
            sec_primary = float(np.mean([risk[int(p)] for p in primaries])) if primaries else 0.0

            sec_penalty = float(sec_var + float(self.lam_sec) * float(sec_primary))

            # hard constraint: if min-nodes violated, reward=0
            if done_min:
                reward = 0.0
            else:
                reward = float(TPS) - float(self.gamma_sec) * float(sec_penalty)

        self.reward = float(reward)

        # write universal step file
        self.current_step += 1
        out_dir = self.ep_dir
        os.makedirs(out_dir, exist_ok=True)
        tag = f"{int(self.current_step):04d}"
        step_path = os.path.join(out_dir, f"step_{tag}.txt")

        with open(step_path, "w", encoding="utf-8") as f:
            f.write(f"{self.nb_nodes}\n")
            for i in range(self.nb_nodes):
                f.write(f"{i}\n")
                f.write(f"{float(self.node_xy[i, 0])}\n")
                f.write(f"{float(self.node_xy[i, 1])}\n")
                f.write(f"{float(self.c_vec[i, 0])}\n")
                if self.pt_write_mode == "dbm":
                    f.write(f"{float(self.Pt_nodes_dbm[i])}\n")
                else:
                    f.write(f"{float(self.Pt_nodes_w[i])}\n")

        # constraints output (kept compatible)
        e_p_scalar = float(np.mean(self.E_mat)) if self.E_mat is not None else 0.0
        Constraints_res = [
            float(T_round) if np.isfinite(T_round) else 0.0,
            float(self.SB_bytes),
            float(T_epoch) if np.isfinite(T_epoch) else 0.0,
            int(n_shard),
            float(100),
            bool(False),
            bool(done_min),
            float(e_p_scalar),
            int(0),
        ]

        info = {
            "invalid_action": bool(invalid_action),
            "shard_id": shard_id,
            "n_shard": int(n_shard),
            "action": int(action),
            "done_min": bool(done_min),

            "tps": float(TPS) if np.isfinite(TPS) else float("nan"),
            "t_round": float(T_round) if np.isfinite(T_round) else float("nan"),
            "t_epoch": float(T_epoch) if np.isfinite(T_epoch) else float("nan"),
            "Treco": float(Treco) if np.isfinite(Treco) else float("nan"),

            "sec_penalty": float(sec_penalty) if np.isfinite(sec_penalty) else float("nan"),
            "sec_var": float(sec_var) if np.isfinite(sec_var) else float("nan"),
            "sec_primary": float(sec_primary) if np.isfinite(sec_primary) else float("nan"),

            "reward": float(reward),

            # echo key params
            "max_shards": int(self.max_shards),
            "min_nodes_per_shard": int(self.min_nodes_per_shard),
            "c_xi": float(self.c_xi),
            "lam_sec": float(self.lam_sec),
            "gamma_sec": float(self.gamma_sec),
            "pt_write_mode": str(self.pt_write_mode),
        }

        return self.state, float(reward), bool(done), Constraints_res, info


# Collector expects BCnetenv symbol
class BCnetenv(BC_Env):
    pass
