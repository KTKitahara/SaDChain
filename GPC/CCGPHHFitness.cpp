#include "comfunc.h"
#include <string>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstring>
#include <ctime>
#include <type_traits>
#include <cstdint>
#include <limits>

#include "gpu_bridge.cuh"

#if defined(_WIN32)
#include <windows.h>
#include <direct.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

using namespace std;

/*
Dataset layout (recommended):
  dataset/train/ep_00001/step_0001.txt ...
  dataset/test /ep_00001/step_0001.txt ...
Step file format:
  first: nb_nodes
  then nb_nodes lines: id x y cal
*/

namespace {

    struct StepFrame {
        int machinesNum = 0;
        std::vector<int>    id;
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> cal;
        std::vector<double> pt;
    };

    struct Episode {
        std::vector<StepFrame> steps;
    };

    struct DatasetSplitData {
        std::string dir;
        bool loaded = false;
        std::vector<Episode> episodes;
        int steps_per_episode = 0;
    };

    static inline std::string join_path(const std::string& a, const std::string& b) {
        if (a.empty()) return b;
        if (a.back() == '/' || a.back() == '\\') return a + b;
        return a + "/" + b;
    }

    static bool has_prefix(const std::string& s, const std::string& p) {
        return s.size() >= p.size() && std::equal(p.begin(), p.end(), s.begin());
    }
    static bool has_suffix(const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() && std::equal(suf.rbegin(), suf.rend(), s.rbegin());
    }

#if defined(_WIN32)
    static bool path_is_dir(const std::string& p) {
        DWORD attr = GetFileAttributesA(p.c_str());
        return (attr != INVALID_FILE_ATTRIBUTES) && (attr & FILE_ATTRIBUTE_DIRECTORY);
    }
    static std::vector<std::string> list_entries(const std::string& dir) {
        std::vector<std::string> out;
        std::string pattern = join_path(dir, "*");
        WIN32_FIND_DATAA ffd;
        HANDLE h = FindFirstFileA(pattern.c_str(), &ffd);
        if (h == INVALID_HANDLE_VALUE) return out;
        do {
            const char* name = ffd.cFileName;
            if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) continue;
            out.push_back(name);
        } while (FindNextFileA(h, &ffd));
        FindClose(h);
        return out;
    }
#else
    static bool path_is_dir(const std::string& p) {
        struct stat st;
        if (stat(p.c_str(), &st) != 0) return false;
        return S_ISDIR(st.st_mode);
    }
    static std::vector<std::string> list_entries(const std::string& dir) {
        std::vector<std::string> out;
        DIR* dp = opendir(dir.c_str());
        if (!dp) return out;
        struct dirent* de;
        while ((de = readdir(dp)) != nullptr) {
            std::string name = de->d_name;
            if (name == "." || name == "..") continue;
            out.push_back(name);
        }
        closedir(dp);
        return out;
    }
#endif

    static bool load_step_file(const std::string& file, StepFrame& out) {
        std::ifstream in(file);
        if (!in) return false;

        int n = 0;
        in >> n;
        if (!in || n <= 0) return false;

        out.machinesNum = n;
        out.id.assign(n, 0);
        out.x.assign(n, 0.0);
        out.y.assign(n, 0.0);
        out.cal.assign(n, 0.0);
        out.pt.assign(n, 0.0);

        for (int i = 0; i < n; ++i) {
            int id; double x, y, cal,pt;
            in >> id >> x >> y >> cal>>pt;
            if (!in) return false;
            out.id[i] = id;
            out.x[i] = x;
            out.y[i] = y;
            out.cal[i] = cal;
            //out.pt[i] = pt;
            if (dbmorw == 0) {
                out.pt[i] = pt;
            }
            else {
                // dBm -> W
                out.pt[i] = pow(10.0, (pt - 30.0) / 10.0);
            }

            // 读取 step 文件中的发射功率 pt，并统一转换为 W 存入 out.pt[i]

            static int dbg_printed = 0;
            if (dbg_printed < 1 && i == 0) { // 只打印 1 次: 第 1 个 step 的第 0 个节点
                double pt_raw = pt;         // step 文件中的原始 pt 值（单位由 dbmorw 指示）
                double pt_w = out.pt[i];    // 内部统一使用的 W 值

                // 将 W 换算回 dBm 便于核对（pt_w>0 才有意义）
                double pt_dbm = (pt_w > 0) ? (10.0 * log10(pt_w * 1000.0)) : -1e9;

                std::cout << "[DBG] stepfile pt_raw=" << pt_raw
                    << "  pt_W=" << pt_w
                    << "  pt_dBm(from_W)=" << pt_dbm
                    << "  dbmorw=" << dbmorw
                    << std::endl;

                dbg_printed++;
            }


        }
        return true;
    }

    static bool dir_has_ep_structure(const std::string& dir) {
        auto entries = list_entries(dir);
        for (auto& name : entries) {
            if (!has_prefix(name, "ep_")) continue;
            if (path_is_dir(join_path(dir, name))) return true;
        }
        return false;
    }

    static void load_dataset_ep_steps(DatasetSplitData& ds) {
        ds.episodes.clear();
        ds.steps_per_episode = 0;

        if (!dir_has_ep_structure(ds.dir)) {
            // fallback: treat step_*.txt as one episode
            Episode ep;
            for (int k = 1; ; ++k) {
                std::ostringstream oss;
                oss << ds.dir << "/step_" << k << ".txt";
                StepFrame f;
                if (!load_step_file(oss.str(), f)) break;
                ep.steps.push_back(std::move(f));
            }
            if (!ep.steps.empty()) {
                ds.steps_per_episode = (int)ep.steps.size();
                ds.episodes.push_back(std::move(ep));
            }
            ds.loaded = true;
            return;
        }

        std::vector<std::string> entries = list_entries(ds.dir);
        std::vector<std::string> ep_dirs;
        for (auto& name : entries) {
            if (!has_prefix(name, "ep_")) continue;
            std::string full = join_path(ds.dir, name);
            if (path_is_dir(full)) ep_dirs.push_back(name);
        }
        std::sort(ep_dirs.begin(), ep_dirs.end());

        for (auto& ep_name : ep_dirs) {
            std::string ep_path = join_path(ds.dir, ep_name);
            std::vector<std::string> ep_entries = list_entries(ep_path);

            std::vector<std::string> step_files;
            for (auto& fn : ep_entries) {
                if (has_prefix(fn, "step_") && has_suffix(fn, ".txt")) step_files.push_back(fn);
            }
            std::sort(step_files.begin(), step_files.end());

            Episode ep;
            ep.steps.reserve(step_files.size());
            for (auto& sf : step_files) {
                StepFrame f;
                std::string fp = join_path(ep_path, sf);
                if (!load_step_file(fp, f)) continue;
                ep.steps.push_back(std::move(f));
            }

            if (!ep.steps.empty()) {
                if (ds.steps_per_episode == 0) ds.steps_per_episode = (int)ep.steps.size();
                ds.episodes.push_back(std::move(ep));
            }
        }

        ds.loaded = true;
    }

    static inline void ensure_dir(const std::string& dir) {
#if defined(_WIN32)
        _mkdir(dir.c_str());
#else
        mkdir(dir.c_str(), 0755);
#endif
    }

    static inline bool file_need_header(const char* path) {
        std::ifstream in(path, std::ios::in);
        if (!in.good()) return true;
        return (in.peek() == std::ifstream::traits_type::eof());
    }

} // namespace

// ===================== Global dataset state =====================
static std::string g_train_dir = "../dataset/train";
static std::string g_test_dir = "../dataset/test";
static int g_split = 0; // 0=train, 1=test

static DatasetSplitData g_train_ds;
static DatasetSplitData g_test_ds;

static DatasetSplitData& active_dataset() {
    DatasetSplitData& ds = (g_split == 0) ? g_train_ds : g_test_ds;
    if (!ds.loaded) {
        ds.dir = (g_split == 0) ? g_train_dir : g_test_dir;
        load_dataset_ep_steps(ds);
        if (ds.episodes.empty()) {
            std::cerr << "[Dataset] no episodes loaded from: " << ds.dir << std::endl;
        }
    }
    return ds;
}

extern "C" void dataset_set_dirs(const char* train_dir, const char* test_dir) {
    if (train_dir && train_dir[0]) g_train_dir = train_dir;
    if (test_dir && test_dir[0])  g_test_dir = test_dir;
    g_train_ds = DatasetSplitData();
    g_test_ds = DatasetSplitData();
}

extern "C" void dataset_set_split(int split01) {
    g_split = (split01 != 0) ? 1 : 0;
}

extern "C" int dataset_get_total_episodes(int split01) {
    DatasetSplitData& ds = (split01 == 0) ? g_train_ds : g_test_ds;
    if (!ds.loaded) {
        ds.dir = (split01 == 0) ? g_train_dir : g_test_dir;
        load_dataset_ep_steps(ds);
    }
    return (int)ds.episodes.size();
}

extern "C" int dataset_get_steps_per_episode(int split01) {
    DatasetSplitData& ds = (split01 == 0) ? g_train_ds : g_test_ds;
    if (!ds.loaded) {
        ds.dir = (split01 == 0) ? g_train_dir : g_test_dir;
        load_dataset_ep_steps(ds);
    }
    return ds.steps_per_episode;
}

// ===================== Episode window =====================
static int g_eval_ep_start = 0;   // 0-based
static int g_eval_ep_count = 0;   // 0 => all

extern "C" void gp_set_eval_episode_window(int start_idx, int count) {
    g_eval_ep_start = start_idx;
    g_eval_ep_count = count;
}
// ===================== Last-eval TPS cache =====================
// Cache the "average TPS (no penalty)" of the last call to CCGPHHFitness().
// This is useful for logging to gp_convergence.csv while still returning fitness (reward).
static double g_last_eval_avg_tps = 0.0;

extern "C" double gp_get_last_eval_avg_tps() {
    return g_last_eval_avg_tps;
}

static double g_last_eval_avg_pen_sec = 0.0;

extern "C" double gp_get_last_eval_avg_pen_sec() {
    return g_last_eval_avg_pen_sec;
}


// ===================== Last-batch mean metrics cache =====================
// These are set by the most recent population-wise batch evaluation call.
// They are meant for logging population mean metrics (reward/TPS/penalty) without
// having to re-run per-individual evaluations.
static double g_last_batch_mean_reward = 0.0;
static double g_last_batch_mean_tps = 0.0;
static double g_last_batch_mean_pen_sec = 0.0;

extern "C" double gp_get_last_batch_mean_reward() {
    return g_last_batch_mean_reward;
}
extern "C" double gp_get_last_batch_mean_tps() {
    return g_last_batch_mean_tps;
}
extern "C" double gp_get_last_batch_mean_pen_sec() {
    return g_last_batch_mean_pen_sec;
}


// ===================== Best-pair recording control =====================
static int  g_record_on = 0;
static int  g_record_gen = 0;
static int  g_record_split = 0; // 0 train, 1 test

extern "C" void gp_record_begin(int gen, int split01) {
    g_record_on = 1;
    g_record_gen = gen;
    g_record_split = (split01 != 0) ? 1 : 0;
    
    //ensure_dir("gp_best");
    ensure_dir("result");
    ensure_dir("result/gp_best");
}

extern "C" void gp_record_end() {
    g_record_on = 0;
}

// ===================== GPU context (reuse GPU buffers) =====================
static GpuContext* g_gpu_ctx = nullptr;
static bool        g_gpu_ctx_inited = false;

static inline void ensure_gpu_ctx(int maxN, int maxP) {
    if (g_gpu_ctx_inited) return;
    g_gpu_ctx_inited = true;

    // Try to init GPU once; ignore failure (bridge has CPU fallback)
    gpu_init_and_smoketest(/*silent=*/true);

    // Create context for buffer reuse; may return nullptr, which is OK (we will fall back)
    g_gpu_ctx = gpu_ctx_create(maxN, maxP, /*use_async=*/0);
}

// ===================== Step distance cache =====================
// Purpose: during GP evaluation, the same (split, episode, step) is evaluated many times for different trees.
// The node coordinates are identical, so pairwise distances + sorting are identical.
// We cache machines[i].dis[0..N) per step to avoid re-running gpu_update_and_sort_distances() repeatedly.
//
// NOTE: This cache is per (current split, current episode). It will be reset automatically when epi/split changes.
using DistEntry = std::remove_cv_t<std::remove_reference_t<decltype(((Machines*)nullptr)->dis[0])>>;

struct StepDistCache {
    int split = -1;
    int epi = -1;
    int N = 0;
    int steps = 0;
    std::vector<uint8_t> ready;     // ready[st] == 1 means data for that step exists
    std::vector<DistEntry> data;    // flattened: [st][i][j]
};

static StepDistCache g_step_dist_cache;

static inline void step_cache_reset_if_needed(int split, int epi, int N, int steps) {
    if (g_step_dist_cache.split == split &&
        g_step_dist_cache.epi == epi &&
        g_step_dist_cache.N == N &&
        g_step_dist_cache.steps == steps) {
        return;
    }
    g_step_dist_cache.split = split;
    g_step_dist_cache.epi = epi;
    g_step_dist_cache.N = N;
    g_step_dist_cache.steps = steps;
    g_step_dist_cache.ready.assign((size_t)steps, 0);
    g_step_dist_cache.data.assign((size_t)steps * (size_t)N * (size_t)N, DistEntry{});
}

static inline bool step_cache_has(int st) {
    return (st >= 0 && st < g_step_dist_cache.steps && g_step_dist_cache.ready[(size_t)st] != 0);
}

static inline void step_cache_restore(Machines* machines, int N, int st) {
    const size_t base = (size_t)st * (size_t)N * (size_t)N;
    const DistEntry* src = g_step_dist_cache.data.data() + base;
    for (int i = 0; i < N; ++i) {
        std::memcpy(machines[i].dis, src + (size_t)i * (size_t)N, (size_t)N * sizeof(DistEntry));
    }
}

static inline void step_cache_store(const Machines* machines, int N, int st) {
    const size_t base = (size_t)st * (size_t)N * (size_t)N;
    DistEntry* dst = g_step_dist_cache.data.data() + base;
    for (int i = 0; i < N; ++i) {
        std::memcpy(dst + (size_t)i * (size_t)N, machines[i].dis, (size_t)N * sizeof(DistEntry));
    }
    g_step_dist_cache.ready[(size_t)st] = 1;
}


//// ===================== Distance helper =====================
//static double calcDistance(const Machines& a, const Machines& b) {
//    double dx = a.xloc - b.xloc;
//    double dy = a.yloc - b.yloc;
//    return std::sqrt(dx * dx + dy * dy);
//}

// ===================== Consensus delay (paper-aligned) =====================

// Keep this symbol name/signature because gpu_kernel.cu may call it as CPU fallback.
double estimate_Tcon_k(int pmac_id, const Machines machines[]) {
    const int followers = machines[pmac_id].owned;
    const int shard_size = followers + 1;
    const int primary = pmac_id;

    auto get_rate = [&](int a, int b) -> double {
        double dist = 1.0;
        for (int i = 0; i < machineNum; ++i) {
            if (machines[a].dis[i].id == b) { dist = machines[a].dis[i].distance; break; }
        }
        const double d = (std::max)(dist, 1.0);
        // Use per-node transmit power (Pt) from the sender node 'a' if available; fall back to default Pt.
        double Pt_eff = machines[a].pt;
        if (!(Pt_eff > 0.0) || !std::isfinite(Pt_eff)) Pt_eff = Pt;
        double Pr = Pt_eff * Gt * Gr * pow(lambda / (4.0 * M_PI * d), 2.0);

        //double Pr = Pt * Gt * Gr * pow(lambda / (4.0 * M_PI * d), 2.0);
        return B * log2(1.0 + Pr / N);
        };

    std::vector<int> nodes;
    nodes.reserve(shard_size);
    nodes.push_back(primary);
    for (int i = 0; i < followers; ++i) nodes.push_back(machines[primary].team[i]);

    std::unordered_map<int, double> t_pp, t_pre, t_com;
    for (int id : nodes) { t_pp[id] = 0.0; t_pre[id] = 0.0; t_com[id] = 0.0; }

    // pre-prepare (primary -> each follower)
    for (int i = 0; i < followers; ++i) {
        int fid = machines[primary].team[i];
        double R = get_rate(primary, fid);
        t_pp[fid] += SB_bits / R;
    }

    // prepare (all-to-all except self)
    for (int s = 0; s < (int)nodes.size(); ++s) {
        int sender = nodes[s];
        for (int r = 0; r < (int)nodes.size(); ++r) {
            if (r == s) continue;
            int recv = nodes[r];
            double R = get_rate(sender, recv);
            t_pre[recv] += SB_bits / R;
        }
    }

    // commit (all-to-all except self)
    for (int s = 0; s < (int)nodes.size(); ++s) {
        int sender = nodes[s];
        for (int r = 0; r < (int)nodes.size(); ++r) {
            if (r == s) continue;
            int recv = nodes[r];
            double R = get_rate(sender, recv);
            t_com[recv] += SB_bits / R;
        }
    }

    double max_pp = 0.0, max_pre = 0.0, max_com = 0.0;
    for (int id : nodes) {
        if (t_pp[id] > max_pp)  max_pp = t_pp[id];
        if (t_pre[id] > max_pre) max_pre = t_pre[id];
        if (t_com[id] > max_com) max_com = t_com[id];
    }
    double T_prop = max_pp + max_pre + max_com;

    // compute delay (your model)
    const double C_primary = (double)M * alpha + (2.0 * M + 4.0 * (shard_size - 1)) * beta;
    const double C_replica = (double)M * alpha + (1.0 * M + 4.0 * (shard_size - 1)) * beta;

    double T_cal_primary = C_primary / (std::max)(1e-12, (double)machines[primary].cal);
    double T_cal_max_rep = 0.0;
    for (int i = 0; i < followers; ++i) {
        int fid = machines[primary].team[i];
        double t_cal = C_replica / (std::max)(1e-12, (double)machines[fid].cal);
        if (t_cal > T_cal_max_rep) T_cal_max_rep = t_cal;
    }
    double T_val = (std::max)(T_cal_primary, T_cal_max_rep);

    // Eq.(5): T_i^con = min(T_i^com, T_max) + min(T_i^cal, T_max)
    double Tcon = (std::min)(T_prop, T_max) + (std::min)(T_val, T_max);
    return Tcon;
}

// no base-station: LC delay among primaries
static double estimate_Tlc_con(const Machines machines[], const int pmacs[], int pnum) {
    if (pnum <= 1) return 0.0;

    auto get_rate = [&](int a, int b) -> double {
        double dist = 1.0;
        for (int i = 0; i < machineNum; ++i) {
            if (machines[a].dis[i].id == b) { dist = machines[a].dis[i].distance; break; }
        }
        const double d = (std::max)(dist, 1.0);
        // Use per-node transmit power (Pt) from the sender node 'a' if available; fall back to default Pt.
        double Pt_eff = machines[a].pt;
        if (!(Pt_eff > 0.0) || !std::isfinite(Pt_eff)) Pt_eff = Pt;
        double Pr = Pt_eff * Gt * Gr * pow(lambda / (4.0 * M_PI * d), 2.0);

        //double Pr = Pt * Gt * Gr * pow(lambda / (4.0 * M_PI * d), 2.0);
        return B * log2(1.0 + Pr / N);
        };

    double max_t = 0.0;
    for (int i = 0; i < pnum; ++i) {
        for (int j = 0; j < pnum; ++j) {
            if (i == j) continue;
            int a = pmacs[i];
            int b = pmacs[j];
            if (a < 0 || b < 0) continue;
            double R = get_rate(a, b);
            double t = SB_bits / (std::max)(R, 1e-12);
            if (t > max_t) max_t = t;
        }
    }
    return max_t;
}

// CPU TPS (debug fallback)
double compute_total_throughput_v2(const Machines machines[], int machineNum_, int pmacs[], int pnum, double Treconfig) {
    if (pnum <= 0) return 0.0;
    double max_Tcon = 0.0;
    for (int i = 0; i < pnum; i++) {
        double Tcon = estimate_Tcon_k(pmacs[i], machines);
        if (Tcon > max_Tcon) max_Tcon = Tcon;
    }
    double Tepoch = tps_r * max_Tcon + Treconfig;
    double T_total = 0.0;
    for (int i = 0; i < pnum; i++) {
        double Tk = (tps_r * SB / ST) / Tepoch;
        T_total += Tk;
    }
    return T_total;
}

// ===================== GPU TPS wrapper (6-arg only) =====================
// gpu_bridge.cuh provides ONLY:
//   compute_total_throughput_v2_gpu(machines, machineNum, pmacs, pnum, Treconfig, T_lc_con)
//
// In this project we pass:
//   Treconfig = T_algo
//   T_lc_con  = T_lc
// NOTE: T_rec is kept in the wrapper signature to avoid touching call sites,
//       but is not used by the current GPU API.
static inline double compute_tps_gpu_compat(const Machines* machines, int machineNum_,
    int* pmacs, int pnum,
    double T_algo, double T_lc, double /*T_rec*/)
{
    if (g_gpu_ctx) {
        return (double)compute_total_throughput_v2_gpu_ctx(g_gpu_ctx, machines, machineNum_, pmacs, pnum, T_algo, T_lc);
    }
    return (double)compute_total_throughput_v2_gpu(machines, machineNum_, pmacs, pnum, T_algo, T_lc);
}

// ===================== Plan string =====================
static std::string build_plan_string(const Machines machines[], const int pmacs[], int pnum) {
    std::ostringstream oss;
    for (int i = 0; i < pnum; ++i) {
        int p = pmacs[i];
        if (p < 0) continue;
        oss << "P" << machines[p].id << ":[";
        for (int k = 0; k < machines[p].owned; ++k) {
            int fid = machines[p].team[k];
            if (fid < 0) continue;
            oss << machines[fid].id;
            if (k + 1 < machines[p].owned) oss << ",";
        }
        oss << "]";
        if (i + 1 < pnum) oss << ";";
    }
    return oss.str();
}

// ===================== Main fitness =====================
double CCGPHHFitness(Machines* machines, const tree& rout, const tree& seq) {

    // reset last-eval TPS cache for this evaluation call
    g_last_eval_avg_tps = 0.0;
    g_last_eval_avg_pen_sec = 0.0;
    DatasetSplitData& ds = active_dataset();
    if (ds.episodes.empty()) return 0.0;

    const int total_eps = (int)ds.episodes.size();

    int start = g_eval_ep_start;
    int count = g_eval_ep_count;
    if (count <= 0) { start = 0; count = total_eps; }

    if (start < 0) start = 0;
    if (total_eps > 0 && start >= total_eps) start = start % total_eps;
    int end = start + count;
    if (end > total_eps) end = total_eps;

    const int pnum_max = (int)(threshold * machineNum);

    // init GPU context once (buffer reuse)
    ensure_gpu_ctx(/*maxN=*/machineNum, /*maxP=*/(pnum_max > 0 ? pnum_max : 1));

    // Reuse pmacs buffer to avoid per-step new/delete
    std::vector<int> pmacs_buf((pnum_max > 0) ? pnum_max : 1);

    double fitness_ep_sum = 0.0;
    double tps_ep_sum = 0.0;   // accumulate episode-average TPS (no penalty)
    double pen_ep_sum = 0.0;   // accumulate episode-average pen_sec (raw, no gamma)
    int ep_cnt = 0;

    std::string split_name = (g_record_split == 0) ? "train" : "test";
    std::string best_path = "result/gp_best/gen_" + std::to_string(g_record_gen) + "_" + split_name + "_best.txt";
    std::string avg_path = "result/gp_best/episode_avg.csv";

    std::ofstream fbest;
    std::ofstream favg;
    if (g_record_on) {
        fbest.open(best_path, std::ios::app);
        favg.open(avg_path, std::ios::app);
        if (file_need_header(avg_path.c_str())) {
            favg << "gen,split,episode,avg_tps,avg_pen_sec,avg_reward\n";
            favg.flush();
        }
    }

    for (int epi = start; epi < end; ++epi) {
        const Episode& ep = ds.episodes[epi];
        // prepare distance cache for this episode
        step_cache_reset_if_needed(g_split, epi, machineNum, (int)ep.steps.size());
        if (ep.steps.empty()) continue;

        double step_reward_sum = 0.0;
        double step_tps_sum = 0.0;
        double step_pen_sec_sum = 0.0;
        int step_cnt = 0;

        double best_step_reward = -1e100;
        double best_step_tps = 0.0;
        int    best_step_idx = -1;
        std::string best_plan;

        for (int st = 0; st < (int)ep.steps.size(); ++st) {
            const StepFrame& frame = ep.steps[st];
            int machinesNum = frame.machinesNum;
            if (machinesNum <= 0) continue;
            if (machinesNum != machineNum) {
                // dataset inconsistency: this implementation assumes machinesNum == machineNum
                continue;
            }

            // refresh from dataset
            for (int ii = 0; ii < machinesNum; ++ii) {
                machines[ii].id = frame.id[ii];
                machines[ii].xloc = frame.x[ii];
                machines[ii].yloc = frame.y[ii];
                machines[ii].cal = frame.cal[ii];
                machines[ii].pt = frame.pt[ii];
                
                machines[ii].atkprob = Ris_from_cal((double)machines[ii].cal);

                machines[ii].priority = 0.0;
                machines[ii].fprime = 0;
                machines[ii].owned = 0;
                // NOTE: do NOT clear team[0..machineNum) here; owned=0 is sufficient and avoids O(N^2) per step.
            }
            if (testallflag != 2) {
                testallflag++;
                for (int ii = 0; ii < 1; ++ii) {
                    printf("now testallflag = %d\n", testallflag);
                    printf("id: %d\n", machines[ii].id);
					printf(" xloc: %f\n", machines[ii].xloc);
					printf(" yloc: %f\n", machines[ii].yloc);
					printf(" cal: %f\n", machines[ii].cal);
					printf(" pt: %f\n", machines[ii].pt);
					printf(" atkprob: %f\n", machines[ii].atkprob);
					printf("priority: %f\n", machines[ii].priority);
					printf(" fprime: %d\n", machines[ii].fprime);
                    printf("owned: %d\n", machines[ii].owned);
					
                }
            }

            // Distances: cached per (split, episode, step) to avoid redundant GPU work across many tree evaluations
            if (step_cache_has(st)) {
                step_cache_restore(machines, machineNum, st);
            } else {
                if (g_gpu_ctx) gpu_update_and_sort_distances_ctx(g_gpu_ctx, machines, machineNum);
                else           gpu_update_and_sort_distances(machines, machineNum);
                step_cache_store(machines, machineNum, st);
            }

            // ============ PNSR ============
            int pnum = pnum_max;
            int now_p = 0;
            int* pmacs = pmacs_buf.data();
            std::fill(pmacs_buf.begin(), pmacs_buf.end(), -1);

            double last_min_priority = -1e9;
            double last_last_min_priority = -2e9;
            double eta = 1.0;

            clock_t start_talgo = clock();

            while (now_p < pnum) {
                double max_priority = -99999990.0;
                double ne_pri = -999999999.0;
                int i_maxp = -1;

                for (int i = 0; i < machineNum; i++) {
                    if (!machines[i].fprime) {
                        double val = decode(machines, i, pmacs, now_p, rout.root);
                        machines[i].priority = (val > -9999999.001 ? val : -9999999.0);
                        if (max_priority < machines[i].priority) {
                            ne_pri = max_priority;
                            max_priority = machines[i].priority;
                            i_maxp = i;
                        }
                    }
                }

                if (now_p > 1 &&
                    (i_maxp == -1 ||
                        (pnum - now_p) * std::abs(max_priority - ne_pri) / (std::max)(std::abs(max_priority), eta) <
                        priority_drop_threshold * std::abs((last_min_priority - last_last_min_priority)) /
                        (std::max)(std::abs(last_min_priority), eta))) {
                    break;
                }

                machines[i_maxp].fprime = 1;
                pmacs[now_p] = i_maxp;

                if (max_priority < last_min_priority) {
                    last_last_min_priority = last_min_priority;
                    last_min_priority = max_priority;
                }
                now_p++;

                if (now_p == 1) {
                    last_min_priority = max_priority;
                }
                else if (now_p == 2) {
                    last_last_min_priority = last_min_priority;
                    last_min_priority = max_priority;
                }
            }

            pnum = now_p; // actual

            // ============ RNAR ============
            int cnode = machineNum - pnum;
            int assigned = 0;

            while (assigned < cnode) {
                int pi = 0;
                while (assigned < cnode && pi < pnum) {
                    int pidx = pmacs[pi];
                    if (pidx < 0) { pi++; continue; }
                    if (machines[pidx].owned > limitednode) { pi++; continue; }

                    double best = -99999999.0;
                    int best_j = -1;

                    for (int j = 0; j < machineNum; j++) {
                        if (!machines[j].fprime) {
                            double val = decode1(machines, j, pmacs, pi, seq.root);
                            machines[j].priority = (val > -9999999.001 ? val : -9999999.0);
                            if (best < machines[j].priority) {
                                best = machines[j].priority;
                                best_j = j;
                            }
                        }
                    }

                    if (best_j == -1) break;

                    machines[best_j].fprime = 2;
                    machines[pidx].team[machines[pidx].owned] = best_j;
                    machines[pidx].owned++;
                    assigned++;
                    pi++;
                }
                if (pnum <= 0) break;
            }

            clock_t end_talgo = clock();
            double T_algo = (double)(end_talgo - start_talgo) / CLOCKS_PER_SEC;

            // ============ T_rec (paper) ============
            double max_Tcon = 0.0;
            for (int i = 0; i < pnum; ++i) {
                double Tcon = estimate_Tcon_k(pmacs[i], machines);
                if (Tcon > max_Tcon) max_Tcon = Tcon;
            }
            double T_lc = estimate_Tlc_con(machines, pmacs, pnum);
            double T_rec = 3.0 * T_lc + max_Tcon + T_algo;

            // ============ TPS (GPU, compat) ============
            // if GPU has 6 args => uses (T_algo, T_lc) internally to form T_rec
            // else GPU has 5 args => Treconfig = T_rec
            double tps = compute_tps_gpu_compat(machines, machineNum, pmacs, pnum, T_algo, T_lc, T_rec);

            // ============ reward ============
            double pen_sec = Pen_sec_from_shards(machines, pmacs, pnum);
            double reward = tps - gamma_sec * pen_sec;

            step_tps_sum += tps;
            step_pen_sec_sum += pen_sec;
            step_reward_sum += reward;
            step_cnt++;

            if (g_record_on) {
                if (reward > best_step_reward) {
                    best_step_reward = reward;
                    best_step_tps = tps;
                    best_step_idx = st;
                    best_plan = build_plan_string(machines, pmacs, pnum);
                }
            }
        } // steps

        if (step_cnt > 0) {
            double ep_avg_reward = step_reward_sum / (double)step_cnt;
            double ep_avg_tps = step_tps_sum / (double)step_cnt;
            double ep_avg_pen_sec = step_pen_sec_sum / (double)step_cnt;

            fitness_ep_sum += ep_avg_reward;
            tps_ep_sum += ep_avg_tps;
            pen_ep_sum += ep_avg_pen_sec;
            ep_cnt++;

            if (g_record_on) {
                favg << g_record_gen << ","
                    << split_name << ","
                    << (epi + 1) << ","
                    << ep_avg_tps << ","
                    << ep_avg_pen_sec << ","
                    << ep_avg_reward << "\n";
                favg.flush();

                fbest << "gen=" << g_record_gen
                    << " split=" << split_name
                    << " episode=" << (epi + 1)
                    << " best_step=" << (best_step_idx + 1)
                    << " tps=" << best_step_tps
                    << " reward=" << best_step_reward
                    << " plan=" << best_plan
                    << "\n";
                fbest.flush();
            }
        }
    } // episodes

    if (ep_cnt <= 0) {
        g_last_eval_avg_tps = 0.0;
        return 0.0;
    }

    // average TPS across the evaluated episodes (usually 1 episode in your setup)
    g_last_eval_avg_tps = tps_ep_sum / (double)ep_cnt;
    g_last_eval_avg_pen_sec = pen_ep_sum / (double)ep_cnt;

    return fitness_ep_sum / (double)ep_cnt;

}


// ===================== Batch evaluation (population-wise) =====================
// Goal: avoid repeating the expensive distance computation (gpu_update_and_sort_distances)
// for every individual when evaluating the SAME (split, episode, step) across a population.
//
// These APIs intentionally do NOT write gp_best logs and do NOT update g_last_eval_avg_tps.
// They only return fitness (reward) for each individual.
//
// Usage pattern from GPHH.cpp (recommended):
//   dataset_set_split(split);
//   gp_set_eval_episode_window(ep_idx, 1);
//   CCGPHHFitness_batch_fixed_seq(machines, routing_pop, SIZE, &CVs, out_f);
//   CCGPHHFitness_batch_fixed_rout(machines, sequencing_pop, SIZE, &CVr, out_f);

static inline void reset_eval_state_for_one_plan(Machines* machines, int machinesNum) {
    for (int ii = 0; ii < machinesNum; ++ii) {
        machines[ii].priority = 0.0;
        machines[ii].fprime = 0;
        machines[ii].owned = 0;
        // NOTE: do NOT clear team[]; owned=0 ensures we won't read stale entries.
    }
}

static inline double eval_one_plan_on_current_step(
    Machines* machines,
    node* rout_root,
    node* seq_root,
    int pnum_max,
    std::vector<int>& pmacs_buf,
    double& out_tps,
    double& out_pen_sec)
{
    out_tps = 0.0;
    out_pen_sec = 0.0;
    if (pnum_max <= 0) return 0.0;

    int pnum = pnum_max;
    int now_p = 0;
    int* pmacs = pmacs_buf.data();
    std::fill(pmacs_buf.begin(), pmacs_buf.end(), -1);

    double last_min_priority = -1e9;
    double last_last_min_priority = -2e9;
    double eta = 1.0;

    clock_t start_talgo = clock();

    // ============ PNSR (routing) ============
    while (now_p < pnum) {
        double max_priority = -99999990.0;
        double ne_pri = -999999999.0;
        int i_maxp = -1;

        for (int i = 0; i < machineNum; i++) {
            if (!machines[i].fprime) {
                double val = decode(machines, i, pmacs, now_p, rout_root);
                machines[i].priority = (val > -9999999.001 ? val : -9999999.0);
                if (max_priority < machines[i].priority) {
                    ne_pri = max_priority;
                    max_priority = machines[i].priority;
                    i_maxp = i;
                }
            }
        }

        if (now_p > 1 &&
            (i_maxp == -1 ||
                (pnum - now_p) * std::abs(max_priority - ne_pri) / (std::max)(std::abs(max_priority), eta) <
                priority_drop_threshold * std::abs((last_min_priority - last_last_min_priority)) /
                (std::max)(std::abs(last_min_priority), eta))) {
            break;
        }

        if (i_maxp < 0) break;

        machines[i_maxp].fprime = 1;
        pmacs[now_p] = i_maxp;

        if (max_priority < last_min_priority) {
            last_last_min_priority = last_min_priority;
            last_min_priority = max_priority;
        }
        now_p++;

        if (now_p == 1) {
            last_min_priority = max_priority;
        }
        else if (now_p == 2) {
            last_last_min_priority = last_min_priority;
            last_min_priority = max_priority;
        }
    }

    pnum = now_p; // actual primary count
    if (pnum <= 0) {
        out_tps = 0.0;
        return 0.0;
    }

    // ============ RNAR (sequencing / assignment) ============
    int cnode = machineNum - pnum;
    int assigned = 0;

    while (assigned < cnode) {
        int pi = 0;
        while (assigned < cnode && pi < pnum) {
            int pidx = pmacs[pi];
            if (pidx < 0) { pi++; continue; }
            if (machines[pidx].owned > limitednode) { pi++; continue; }

            double best = -99999999.0;
            int best_j = -1;

            for (int j = 0; j < machineNum; j++) {
                if (!machines[j].fprime) {
                    double val = decode1(machines, j, pmacs, pi, seq_root);
                    machines[j].priority = (val > -9999999.001 ? val : -9999999.0);
                    if (best < machines[j].priority) {
                        best = machines[j].priority;
                        best_j = j;
                    }
                }
            }

            if (best_j == -1) break;

            machines[best_j].fprime = 2;
            machines[pidx].team[machines[pidx].owned] = best_j;
            machines[pidx].owned++;
            assigned++;
            pi++;
        }
        if (pnum <= 0) break;
    }

    clock_t end_talgo = clock();
    double T_algo = (double)(end_talgo - start_talgo) / CLOCKS_PER_SEC;

    // ============ T_rec (paper) ============
    double max_Tcon = 0.0;
    for (int i = 0; i < pnum; ++i) {
        double Tcon = estimate_Tcon_k(pmacs[i], machines);
        if (Tcon > max_Tcon) max_Tcon = Tcon;
    }
    double T_lc = estimate_Tlc_con(machines, pmacs, pnum);
    double T_rec = 3.0 * T_lc + max_Tcon + T_algo;

    // ============ TPS (GPU, compat) ============
    double tps = compute_tps_gpu_compat(machines, machineNum, pmacs, pnum, T_algo, T_lc, T_rec);

    // ============ reward ============
    double pen_sec = Pen_sec_from_shards(machines, pmacs, pnum);
    double reward = tps - gamma_sec * pen_sec;

    out_tps = tps;
    out_pen_sec = pen_sec;
    return reward;
}

extern "C" void CCGPHHFitness_batch_fixed_seq(
    Machines* machines,
    const tree* rout_pop,
    int pop_size,
    const tree* seq_fixed,
    double* out_fitness)
{
    if (!machines || !rout_pop || !seq_fixed || !out_fitness || pop_size <= 0) return;

    DatasetSplitData& ds = active_dataset();
    if (ds.episodes.empty()) {
        for (int i = 0; i < pop_size; ++i) out_fitness[i] = 0.0;
        return;
    }

    const int total_eps = (int)ds.episodes.size();

    int start = g_eval_ep_start;
    int count = g_eval_ep_count;
    if (count <= 0) { start = 0; count = total_eps; }

    if (start < 0) start = 0;
    if (total_eps > 0 && start >= total_eps) start = start % total_eps;
    int end = start + count;
    if (end > total_eps) end = total_eps;

    const int pnum_max = (int)(threshold * machineNum);
    ensure_gpu_ctx(/*maxN=*/machineNum, /*maxP=*/(pnum_max > 0 ? pnum_max : 1));

    std::vector<int> pmacs_buf((pnum_max > 0) ? pnum_max : 1);

    std::vector<double> fitness_ep_sum((size_t)pop_size, 0.0);
    std::vector<double> tps_ep_sum((size_t)pop_size, 0.0);
    std::vector<double> pen_ep_sum((size_t)pop_size, 0.0);
    std::vector<int>    ep_cnt((size_t)pop_size, 0);

    for (int epi = start; epi < end; ++epi) {
        const Episode& ep = ds.episodes[epi];
        step_cache_reset_if_needed(g_split, epi, machineNum, (int)ep.steps.size());
        if (ep.steps.empty()) continue;

        std::vector<double> step_reward_sum((size_t)pop_size, 0.0);
        std::vector<double> step_tps_sum((size_t)pop_size, 0.0);
        std::vector<double> step_pen_sum((size_t)pop_size, 0.0);
        std::vector<int>    step_cnt((size_t)pop_size, 0);

        for (int st = 0; st < (int)ep.steps.size(); ++st) {
            const StepFrame& frame = ep.steps[st];
            int machinesNum = frame.machinesNum;
            if (machinesNum <= 0) continue;
            if (machinesNum != machineNum) continue;

            // refresh from dataset ONCE for this step
            for (int ii = 0; ii < machinesNum; ++ii) {
                machines[ii].id = frame.id[ii];
                machines[ii].xloc = frame.x[ii];
                machines[ii].yloc = frame.y[ii];
                machines[ii].cal = frame.cal[ii];
                machines[ii].pt = frame.pt[ii];
                machines[ii].atkprob = Ris_from_cal((double)machines[ii].cal);
            }

            // distances ONCE per step (cached per split/episode/step)
            if (step_cache_has(st)) {
                step_cache_restore(machines, machineNum, st);
            } else {
                if (g_gpu_ctx) gpu_update_and_sort_distances_ctx(g_gpu_ctx, machines, machineNum);
                else           gpu_update_and_sort_distances(machines, machineNum);
                step_cache_store(machines, machineNum, st);
            }

            // evaluate each routing individual (fixed seq)
            for (int pi = 0; pi < pop_size; ++pi) {
                reset_eval_state_for_one_plan(machines, machinesNum);
                double tps = 0.0;
                double pen_sec = 0.0;
                double reward = eval_one_plan_on_current_step(
                    machines,
                    rout_pop[pi].root,
                    seq_fixed->root,
                    pnum_max,
                    pmacs_buf,
                    tps,
                    pen_sec);
                step_reward_sum[(size_t)pi] += reward;
                step_tps_sum[(size_t)pi] += tps;
                step_pen_sum[(size_t)pi] += pen_sec;
                step_cnt[(size_t)pi] += 1;
            }
        } // steps

        for (int pi = 0; pi < pop_size; ++pi) {
            if (step_cnt[(size_t)pi] > 0) {
                fitness_ep_sum[(size_t)pi] += step_reward_sum[(size_t)pi] / (double)step_cnt[(size_t)pi];
                tps_ep_sum[(size_t)pi] += step_tps_sum[(size_t)pi] / (double)step_cnt[(size_t)pi];
                pen_ep_sum[(size_t)pi] += step_pen_sum[(size_t)pi] / (double)step_cnt[(size_t)pi];
                ep_cnt[(size_t)pi] += 1;
            }
        }
    } // episodes

    for (int pi = 0; pi < pop_size; ++pi) {
        if (ep_cnt[(size_t)pi] > 0) out_fitness[pi] = fitness_ep_sum[(size_t)pi] / (double)ep_cnt[(size_t)pi];
        else                        out_fitness[pi] = 0.0;
    }

    // Compute population mean metrics over this batch window.
    double sum_r = 0.0, sum_t = 0.0, sum_p = 0.0;
    for (int pi = 0; pi < pop_size; ++pi) {
        sum_r += out_fitness[pi];
        if (ep_cnt[(size_t)pi] > 0) {
            sum_t += tps_ep_sum[(size_t)pi] / (double)ep_cnt[(size_t)pi];
            sum_p += pen_ep_sum[(size_t)pi] / (double)ep_cnt[(size_t)pi];
        }
    }
    g_last_batch_mean_reward = sum_r / (double)pop_size;
    g_last_batch_mean_tps = sum_t / (double)pop_size;
    g_last_batch_mean_pen_sec = sum_p / (double)pop_size;

}

extern "C" void CCGPHHFitness_batch_fixed_rout(
    Machines* machines,
    const tree* seq_pop,
    int pop_size,
    const tree* rout_fixed,
    double* out_fitness)
{
    if (!machines || !seq_pop || !rout_fixed || !out_fitness || pop_size <= 0) return;

    DatasetSplitData& ds = active_dataset();
    if (ds.episodes.empty()) {
        for (int i = 0; i < pop_size; ++i) out_fitness[i] = 0.0;
        return;
    }

    const int total_eps = (int)ds.episodes.size();

    int start = g_eval_ep_start;
    int count = g_eval_ep_count;
    if (count <= 0) { start = 0; count = total_eps; }

    if (start < 0) start = 0;
    if (total_eps > 0 && start >= total_eps) start = start % total_eps;
    int end = start + count;
    if (end > total_eps) end = total_eps;

    const int pnum_max = (int)(threshold * machineNum);
    ensure_gpu_ctx(/*maxN=*/machineNum, /*maxP=*/(pnum_max > 0 ? pnum_max : 1));

    std::vector<int> pmacs_buf((pnum_max > 0) ? pnum_max : 1);

    std::vector<double> fitness_ep_sum((size_t)pop_size, 0.0);
    std::vector<double> tps_ep_sum((size_t)pop_size, 0.0);
    std::vector<double> pen_ep_sum((size_t)pop_size, 0.0);
    std::vector<int>    ep_cnt((size_t)pop_size, 0);

    for (int epi = start; epi < end; ++epi) {
        const Episode& ep = ds.episodes[epi];
        step_cache_reset_if_needed(g_split, epi, machineNum, (int)ep.steps.size());
        if (ep.steps.empty()) continue;

        std::vector<double> step_reward_sum((size_t)pop_size, 0.0);
        std::vector<double> step_tps_sum((size_t)pop_size, 0.0);
        std::vector<double> step_pen_sum((size_t)pop_size, 0.0);
        std::vector<int>    step_cnt((size_t)pop_size, 0);

        for (int st = 0; st < (int)ep.steps.size(); ++st) {
            const StepFrame& frame = ep.steps[st];
            int machinesNum = frame.machinesNum;
            if (machinesNum <= 0) continue;
            if (machinesNum != machineNum) continue;

            // refresh from dataset ONCE for this step
            for (int ii = 0; ii < machinesNum; ++ii) {
                machines[ii].id = frame.id[ii];
                machines[ii].xloc = frame.x[ii];
                machines[ii].yloc = frame.y[ii];
                machines[ii].cal = frame.cal[ii];
                machines[ii].pt = frame.pt[ii];
                machines[ii].atkprob = Ris_from_cal((double)machines[ii].cal);
            }

            // distances ONCE per step (cached per split/episode/step)
            if (step_cache_has(st)) {
                step_cache_restore(machines, machineNum, st);
            } else {
                if (g_gpu_ctx) gpu_update_and_sort_distances_ctx(g_gpu_ctx, machines, machineNum);
                else           gpu_update_and_sort_distances(machines, machineNum);
                step_cache_store(machines, machineNum, st);
            }

            // evaluate each sequencing individual (fixed rout)
            for (int pi = 0; pi < pop_size; ++pi) {
                reset_eval_state_for_one_plan(machines, machinesNum);
                double tps = 0.0;
                double pen_sec = 0.0;
                double reward = eval_one_plan_on_current_step(
                    machines,
                    rout_fixed->root,
                    seq_pop[pi].root,
                    pnum_max,
                    pmacs_buf,
                    tps,
                    pen_sec);
                step_reward_sum[(size_t)pi] += reward;
                step_tps_sum[(size_t)pi] += tps;
                step_pen_sum[(size_t)pi] += pen_sec;
                step_cnt[(size_t)pi] += 1;
            }
        } // steps

        for (int pi = 0; pi < pop_size; ++pi) {
            if (step_cnt[(size_t)pi] > 0) {
                fitness_ep_sum[(size_t)pi] += step_reward_sum[(size_t)pi] / (double)step_cnt[(size_t)pi];
                tps_ep_sum[(size_t)pi] += step_tps_sum[(size_t)pi] / (double)step_cnt[(size_t)pi];
                pen_ep_sum[(size_t)pi] += step_pen_sum[(size_t)pi] / (double)step_cnt[(size_t)pi];
                ep_cnt[(size_t)pi] += 1;
            }
        }
    } // episodes

    for (int pi = 0; pi < pop_size; ++pi) {
        if (ep_cnt[(size_t)pi] > 0) out_fitness[pi] = fitness_ep_sum[(size_t)pi] / (double)ep_cnt[(size_t)pi];
        else                        out_fitness[pi] = 0.0;
    }

    // Compute population mean metrics over this batch window.
    double sum_r = 0.0, sum_t = 0.0, sum_p = 0.0;
    for (int pi = 0; pi < pop_size; ++pi) {
        sum_r += out_fitness[pi];
        if (ep_cnt[(size_t)pi] > 0) {
            sum_t += tps_ep_sum[(size_t)pi] / (double)ep_cnt[(size_t)pi];
            sum_p += pen_ep_sum[(size_t)pi] / (double)ep_cnt[(size_t)pi];
        }
    }
    g_last_batch_mean_reward = sum_r / (double)pop_size;
    g_last_batch_mean_tps = sum_t / (double)pop_size;
    g_last_batch_mean_pen_sec = sum_p / (double)pop_size;

}

