#include "comfunc.h"

#include <algorithm>
#include <vector>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <string>
#include <cmath>

#include <limits>
// ===== dataset control API (implemented in CCGPHHFitness.cpp) =====
extern "C" void dataset_set_split(int split01);                 // 0=train, 1=test
extern "C" int  dataset_get_total_episodes(int split01);        // number of ep_* folders
extern "C" int  dataset_get_steps_per_episode(int split01);     // inferred steps per ep

// Evaluate window control: only evaluate ONE episode per call
extern "C" void gp_set_eval_episode_window(int start_idx, int count);

// Best-per-episode recording (only enabled for the best pair)
extern "C" void gp_record_begin(int gen, int split01);
extern "C" void gp_record_end();

// Average TPS (no penalty) from the last CCGPHHFitness call
extern "C" double gp_get_last_eval_avg_tps();
// Average security penalty (raw, no gamma) from the last CCGPHHFitness call
extern "C" double gp_get_last_eval_avg_pen_sec();
// Population mean metrics from the last batch evaluation call
extern "C" double gp_get_last_batch_mean_reward();
extern "C" double gp_get_last_batch_mean_tps();
extern "C" double gp_get_last_batch_mean_pen_sec();

using namespace std;

// ----------------- utilities -----------------
static inline bool file_need_header(const char* path) {
    std::ifstream in(path, std::ios::in);
    if (!in.good()) return true;
    return (in.peek() == std::ifstream::traits_type::eof());
}

// Baker ranking selection CDF (assume sorted by fitness desc)
static inline void build_baker_cdf(int M, float* EP, double s_fixed) {
    // p_i = (2 - s)/M + 2(s - 1)/(M(M - 1)) * (M - rank)
    const double base = (2.0 - s_fixed) / (double)M;
    const double step = (M > 1) ? (2.0 * (s_fixed - 1.0) / (double)(M * (M - 1))) : 0.0;
    double csum = 0.0;
    for (int i = 0; i < M; ++i) {
        const int rank = i + 1; // 1..M, i=0 is best
        const double pi = base + step * (double)(M - rank);
        csum += pi;
        EP[i] = (float)csum;
    }
    EP[M - 1] = 1.0f;
}

static inline int baker_sample(const float* EP, int M) {
    const double r = (double)rand() / (double)RAND_MAX;
    int k = 0;
    while (k < M - 1 && r > EP[k]) ++k;
    return k;
}

// ----------------- evaluation wrapper -----------------
static inline double eval_on_split_one_episode(
    Machines* machines, const tree& r, const tree& s,
    int split01, int episode_idx_0based)
{
    // Avoid redundant context switches when evaluating many individuals on the same split/episode.
    // This saves a *lot* of tiny overhead because fitness evaluation is called very frequently.
    static int s_last_split = -1;
    static int s_last_ep = -1;
    if (split01 != s_last_split) {
        dataset_set_split(split01);
        s_last_split = split01;
        // split change implies data source changed; reset episode cache
        s_last_ep = -1;
    }
    if (episode_idx_0based != s_last_ep) {
        gp_set_eval_episode_window(episode_idx_0based, 1); // ONE episode only
        s_last_ep = episode_idx_0based;
    }
    return CCGPHHFitness(machines, r, s);
}


// ============================================================================
//  GPHH: Cooperative Co-evolutionary GP Hyper-Heuristic
//  Training mode (as you required):
//   - Initialization: evaluate only ONE TRAIN episode (episode #1) and record it as iter=0.
//   - Each generation: uses the NEXT episode (one episode per generation) to evaluate population.
//   - Best pair (CVr, CVs): for each generation, record only best plan/tps/reward for that episode.
// ============================================================================

void GPHH(Machines* machines, int SIZE, int Itr, double* bestFitness, tree& bestCVr, tree& bestCVs)
{
    srand((unsigned)time(NULL));
    init();

    // ---- GP params (keep consistent with your project defaults) ----
    int dep1 = 7, dep2 = 7;           // max depth (routing / sequencing)
    int Nrs = 0, Nre = 6;             // terminal/operator id range (routing PNSR)
    int Nss = 0, Nse = 7;             // terminal/operator id range (sequencing RNAR)
    double rate = 0.7;                // construct/mutation internal probability
    double prom = 0.15;               // mutation probability
    const double s_fixed = 1.7;       // Baker selection pressure in [1,2]

    // ---- Simulated Annealing acceptance (for mutation) ----
    // SA 初温建议: 0.20 ~ 0.50
    double sa_T0 = 0.30;     // 初温
    double sa_Tmin = 0.02;   // 最低温度/停止阈值
    double sa_decay = 0.985; // 每代衰减系数 (0.98 ~ 0.995)

    auto sa_accept = [&](double oldF, double newF, double T) -> bool {
        if (newF >= oldF) return true;
        // 用 |oldF| 归一化，避免 fitness 尺度变化导致 exp 下溢/上溢
        double scale = std::max(1.0, std::fabs(oldF));
        double denom = std::max(1e-12, T * scale);
        double p = std::exp((newF - oldF) / denom);  // (new-old)<0 p in (0,1)
        double u = (double)rand() / (double)RAND_MAX;
        return u < p;
        };
    // ---- Fitness validity helpers ----
    // NOTE: fitness can be negative due to penalty terms, so NEVER use (fitness <= 0) as "not evaluated".
    // We use NaN to mark "not evaluated / invalidated by crossover or mutation".
    auto fitness_invalidate = [](tree& t) {
        t.fitness = std::numeric_limits<double>::quiet_NaN();
    };
    auto fitness_valid = [](const tree& t) -> bool {
        return !std::isnan(t.fitness);
    };



    // (mu + lambda) evolution
    double lambda_factor = 1.5;
    int LAMBDA = (std::max)(2, (int)(lambda_factor * SIZE));
    if (LAMBDA % 2) ++LAMBDA;

    // immigration
    int imm_period = 25;
    double imm_rate = 0.10;
    int imm_cnt = (std::max)(1, (int)(imm_rate * SIZE));

    // ---- populations ----
    tree* routing = new tree[SIZE];
    tree* sequencing = new tree[SIZE];
    float* EP1 = new float[SIZE];
    float* EP2 = new float[SIZE];

    // Batch fitness buffer (reused)
    std::vector<double> batch_fit((size_t)SIZE, 0.0);

    // ---- init population (random) ----
    for (int i = 0; i < SIZE; ++i) {
        routing[i].construct(rate, Nrs, Nre, dep1, routing[i].root);
        sequencing[i].construct(rate, Nss, Nse, dep2, sequencing[i].root);
    }

    // initial elites placeholders
    tree CVr = routing[0];
    tree CVs = sequencing[0];

    // episode indices (0-based)
    int train_ep_idx = 0;
    int test_ep_idx = 0;

    int total_train_eps = dataset_get_total_episodes(0);
    int total_test_eps = dataset_get_total_episodes(1);
    if (total_train_eps <= 0) total_train_eps = 1;
    if (total_test_eps <= 0) total_test_eps = 1;

    // ---- initial fitness on TRAIN: only ONE episode (Episode 1) ----
    std::cout << "[InitEpisode] TRAIN episode " << (train_ep_idx + 1)
        << "/" << total_train_eps << " begin" << std::endl;

    
    // Batch-evaluate populations on the current TRAIN episode to avoid redundant distance computation.
    dataset_set_split(0);
    gp_set_eval_episode_window(train_ep_idx, 1);

    CCGPHHFitness_batch_fixed_seq(machines, routing, SIZE, &CVs, batch_fit.data());
    for (int i = 0; i < SIZE; ++i) routing[i].fitness = batch_fit[(size_t)i];

    // population mean metrics (routing population evaluated with fixed CVs)
    double mean_r_reward_init = gp_get_last_batch_mean_reward();
    double mean_r_tps_init = gp_get_last_batch_mean_tps();
    double mean_r_pen_sec_init = gp_get_last_batch_mean_pen_sec();

    CCGPHHFitness_batch_fixed_rout(machines, sequencing, SIZE, &CVr, batch_fit.data());
    for (int i = 0; i < SIZE; ++i) sequencing[i].fitness = batch_fit[(size_t)i];

    // population mean metrics (sequencing population evaluated with fixed CVr)
    double mean_s_reward_init = gp_get_last_batch_mean_reward();
    double mean_s_tps_init = gp_get_last_batch_mean_tps();
    double mean_s_pen_sec_init = gp_get_last_batch_mean_pen_sec();


    std::cout << "[InitEpisode] TRAIN episode " << (train_ep_idx + 1)
        << "/" << total_train_eps << " end" << std::endl;

    // ---- convergence curve output (iter=0 is init episode result) ----
    std::string curve_path_s = result_path("gp_convergence.csv");
    const char* curve_path = curve_path_s.c_str();

    bool need_header = file_need_header(curve_path);
    ofstream fcurve(curve_path, ios::app);
    if (need_header) {
        // CSV 表头: best/pair 指标 + 种群均值指标 (mean_*)
        fcurve << "iter,train_episode,test_episode,"
            << "best_r_reward,best_r_tps,best_r_pen_sec,best_s_reward,best_s_tps,best_s_pen_sec,"
            << "mean_r_reward,mean_r_tps,mean_r_pen_sec,mean_s_reward,mean_s_tps,mean_s_pen_sec,"
            << "pair_train_reward,pair_train_tps,pair_train_pen_sec,pair_test_reward,pair_test_tps,pair_test_pen_sec\n";
    }

    // 兼容旧输出: result_fitness.txt 只追加 best_r 和 best_s 的 reward/fitness
    ofstream ftg(result_path("result_fitness.txt"), ios::app);

    // ================= Init round is treated as Episode #1 and IS recorded =================
    sort(routing, routing + SIZE, [](const tree& a, const tree& b) { return a.fitness > b.fitness; });
    sort(sequencing, sequencing + SIZE, [](const tree& a, const tree& b) { return a.fitness > b.fitness; });
    CVr = routing[0];
    CVs = sequencing[0];

    // record BEST pair on TRAIN episode #1
    gp_record_begin(0, 0);
    double pair_train_reward_init = eval_on_split_one_episode(machines, CVr, CVs, 0, train_ep_idx);
    double pair_train_tps_init = gp_get_last_eval_avg_tps();
    double pair_train_pen_sec_init = gp_get_last_eval_avg_pen_sec();
    gp_record_end();

    // record BEST pair on TEST episode #1
    gp_record_begin(0, 1);
    double pair_test_reward_init = eval_on_split_one_episode(machines, CVr, CVs, 1, test_ep_idx);
    double pair_test_tps_init = gp_get_last_eval_avg_tps();
    double pair_test_pen_sec_init = gp_get_last_eval_avg_pen_sec();
    gp_record_end();

    // 协同进化中 best_r/best_s 均对应当前最优组合 (best pair) 在 TRAIN episode 上的指标
    // 因此这里用 best pair 在 TRAIN episode 上的 reward/tps 作为 best_r/best_s 指标
    double best_r_reward_init = pair_train_reward_init;
    double best_r_tps_init = pair_train_tps_init;
    double best_s_reward_init = pair_train_reward_init;
    double best_s_tps_init = pair_train_tps_init;

    fcurve << 0 << "," << (train_ep_idx + 1) << "," << (test_ep_idx + 1) << ","
        << best_r_reward_init << "," << best_r_tps_init << "," << pair_train_pen_sec_init << ","
        << best_s_reward_init << "," << best_s_tps_init << "," << pair_train_pen_sec_init << ","
        << mean_r_reward_init << "," << mean_r_tps_init << "," << mean_r_pen_sec_init << ","
        << mean_s_reward_init << "," << mean_s_tps_init << "," << mean_s_pen_sec_init << ","
        << pair_train_reward_init << "," << pair_train_tps_init << "," << pair_train_pen_sec_init << ","
        << pair_test_reward_init << "," << pair_test_tps_init << "," << pair_test_pen_sec_init << "\n";
    fcurve.flush();

    ftg << best_r_reward_init << '\n' << best_s_reward_init << '\n';

    // move to next episode for generation-1
    train_ep_idx = (train_ep_idx + 1) % total_train_eps;
    test_ep_idx = (test_ep_idx + 1) % total_test_eps;

    // ---- generations ----
    for (int gen = 0; gen < Itr; ++gen) {
        // sort by fitness desc (maximize)
        std::cout << "[GenEpisode] gen=" << (gen + 1)
            << " TRAIN episode " << (train_ep_idx + 1) << "/" << total_train_eps
            << " begin" << std::endl;

        // refresh fitness for current TRAIN episode (avoid stale fitness across episodes)
        // Batch-evaluate populations on this TRAIN episode (huge reduction in redundant distance computation).
        dataset_set_split(0);
        gp_set_eval_episode_window(train_ep_idx, 1);

        CCGPHHFitness_batch_fixed_seq(machines, routing, SIZE, &CVs, batch_fit.data());
        for (int i = 0; i < SIZE; ++i) routing[i].fitness = batch_fit[(size_t)i];

        // population mean metrics (routing population evaluated with fixed CVs)
        double mean_r_reward = gp_get_last_batch_mean_reward();
        double mean_r_tps = gp_get_last_batch_mean_tps();
        double mean_r_pen_sec = gp_get_last_batch_mean_pen_sec();

        CCGPHHFitness_batch_fixed_rout(machines, sequencing, SIZE, &CVr, batch_fit.data());
        for (int i = 0; i < SIZE; ++i) sequencing[i].fitness = batch_fit[(size_t)i];

        // population mean metrics (sequencing population evaluated with fixed CVr)
        double mean_s_reward = gp_get_last_batch_mean_reward();
        double mean_s_tps = gp_get_last_batch_mean_tps();
        double mean_s_pen_sec = gp_get_last_batch_mean_pen_sec();


        sort(routing, routing + SIZE, [](const tree& a, const tree& b) { return a.fitness > b.fitness; });
        sort(sequencing, sequencing + SIZE, [](const tree& a, const tree& b) { return a.fitness > b.fitness; });

        CVr = routing[0];
        CVs = sequencing[0];
        /*std::cout << "routing[0].root=" << (void*)routing[0].root
          << "  CVr.root=" << (void*)CVr.root << std::endl;*/
        // build Baker CDF
        build_baker_cdf(SIZE, EP1, s_fixed);
        build_baker_cdf(SIZE, EP2, s_fixed);

        // ============= evolve routing population (fixed CVs) =============
        {
            std::vector<tree> offspring;
            offspring.reserve(LAMBDA);

            for (int k = 0; k < LAMBDA / 2; ++k) {
                tree p0 = routing[baker_sample(EP1, SIZE)];
                tree p1 = routing[baker_sample(EP1, SIZE)];

                tree c0 = p0, c1 = p1;
                crossover(c0, c1, dep1);

                fitness_invalidate(c0);
                fitness_invalidate(c1);
                // mutation with SA acceptance
                if (((double)rand() / RAND_MAX) < prom) {
                    double T = std::max(sa_Tmin, sa_T0 * std::pow(sa_decay, (double)gen));

                    // c0: 交叉后、变异前，先在当前 episode 上评估其 fitness
                    c0.fitness = eval_on_split_one_episode(machines, c0, CVs, 0, train_ep_idx);

                    tree m = c0;
                    m.mutation(rate, Nrs, Nre, dep1);
                    m.fitness = eval_on_split_one_episode(machines, m, CVs, 0, train_ep_idx);

                    if (sa_accept(c0.fitness, m.fitness, T)) c0 = m;
                }

                // mutation with SA acceptance
                if (((double)rand() / RAND_MAX) < prom) {
                    double T = std::max(sa_Tmin, sa_T0 * std::pow(sa_decay, (double)gen));

                    c1.fitness = eval_on_split_one_episode(machines, c1, CVs, 0, train_ep_idx);

                    tree m = c1;
                    m.mutation(rate, Nrs, Nre, dep1);
                    m.fitness = eval_on_split_one_episode(machines, m, CVs, 0, train_ep_idx);

                    if (sa_accept(c1.fitness, m.fitness, T)) c1 = m;
                }

                // 确保 fitness 已计算（用 NaN 表示无效/未评估）
                if (!fitness_valid(c0)) c0.fitness = eval_on_split_one_episode(machines, c0, CVs, 0, train_ep_idx);
                if (!fitness_valid(c1)) c1.fitness = eval_on_split_one_episode(machines, c1, CVs, 0, train_ep_idx);
                offspring.push_back(c0);
                offspring.push_back(c1);
            }

            // (mu + lambda): merge and keep top SIZE
            std::vector<tree> pool;
            pool.reserve(SIZE + LAMBDA);
            for (int i = 0; i < SIZE; ++i) pool.push_back(routing[i]);
            for (auto& x : offspring) pool.push_back(x);

            std::sort(pool.begin(), pool.end(),
                [](const tree& a, const tree& b) { return a.fitness > b.fitness; });

            for (int i = 0; i < SIZE; ++i) routing[i] = pool[i];
            CVr = routing[0];

            // immigration
            if ((gen + 1) % imm_period == 0) {
                for (int i = 0; i < imm_cnt; ++i) {
                    int idx = SIZE - 1 - i;
                    if (idx <= 0) break;
                    routing[idx].construct(rate, Nrs, Nre, dep1, routing[idx].root);
                    routing[idx].fitness = eval_on_split_one_episode(machines, routing[idx], CVs, 0, train_ep_idx);
                }
                sort(routing, routing + SIZE, [](const tree& a, const tree& b) { return a.fitness > b.fitness; });
                CVr = routing[0];
            }
        }

        // ============= evolve sequencing population (fixed CVr) =============
        {
            std::vector<tree> offspring;
            offspring.reserve(LAMBDA);

            for (int k = 0; k < LAMBDA / 2; ++k) {
                tree p0 = sequencing[baker_sample(EP2, SIZE)];
                tree p1 = sequencing[baker_sample(EP2, SIZE)];

                tree c0 = p0, c1 = p1;
                crossover(c0, c1, dep2);

                fitness_invalidate(c0);
                fitness_invalidate(c1);
                if (((double)rand() / RAND_MAX) < prom) {
                    double T = std::max(sa_Tmin, sa_T0 * std::pow(sa_decay, (double)gen));

                    c0.fitness = eval_on_split_one_episode(machines, CVr, c0, 0, train_ep_idx);

                    tree m = c0;
                    m.mutation(rate, Nss, Nse, dep2);
                    m.fitness = eval_on_split_one_episode(machines, CVr, m, 0, train_ep_idx);

                    if (sa_accept(c0.fitness, m.fitness, T)) c0 = m;
                }

                if (((double)rand() / RAND_MAX) < prom) {
                    double T = std::max(sa_Tmin, sa_T0 * std::pow(sa_decay, (double)gen));

                    c1.fitness = eval_on_split_one_episode(machines, CVr, c1, 0, train_ep_idx);

                    tree m = c1;
                    m.mutation(rate, Nss, Nse, dep2);
                    m.fitness = eval_on_split_one_episode(machines, CVr, m, 0, train_ep_idx);

                    if (sa_accept(c1.fitness, m.fitness, T)) c1 = m;
                }

                if (!fitness_valid(c0)) c0.fitness = eval_on_split_one_episode(machines, CVr, c0, 0, train_ep_idx);
                if (!fitness_valid(c1)) c1.fitness = eval_on_split_one_episode(machines, CVr, c1, 0, train_ep_idx);
                offspring.push_back(c0);
                offspring.push_back(c1);
            }

            std::vector<tree> pool;
            pool.reserve(SIZE + LAMBDA);
            for (int i = 0; i < SIZE; ++i) pool.push_back(sequencing[i]);
            for (auto& x : offspring) pool.push_back(x);

            std::sort(pool.begin(), pool.end(),
                [](const tree& a, const tree& b) { return a.fitness > b.fitness; });

            for (int i = 0; i < SIZE; ++i) sequencing[i] = pool[i];
            CVs = sequencing[0];

            // immigration
            if ((gen + 1) % imm_period == 0) {
                for (int i = 0; i < imm_cnt; ++i) {
                    int idx = SIZE - 1 - i;
                    if (idx <= 0) break;
                    sequencing[idx].construct(rate, Nss, Nse, dep2, sequencing[idx].root);
                    sequencing[idx].fitness = eval_on_split_one_episode(machines, CVr, sequencing[idx], 0, train_ep_idx);
                }
                sort(sequencing, sequencing + SIZE, [](const tree& a, const tree& b) { return a.fitness > b.fitness; });
                CVs = sequencing[0];
            }
        }

        // ============= generation end: log convergence (best pair on TRAIN/TEST) =============
        sort(routing, routing + SIZE, [](const tree& a, const tree& b) { return a.fitness > b.fitness; });
        sort(sequencing, sequencing + SIZE, [](const tree& a, const tree& b) { return a.fitness > b.fitness; });
        CVr = routing[0];
        CVs = sequencing[0];

        // record ONLY best pair on current train/test episode
        gp_record_begin(gen + 1, 0);
        double pair_train_reward = eval_on_split_one_episode(machines, CVr, CVs, 0, train_ep_idx);
        double pair_train_tps = gp_get_last_eval_avg_tps();
        double pair_train_pen_sec = gp_get_last_eval_avg_pen_sec();
        gp_record_end();

        gp_record_begin(gen + 1, 1);
        double pair_test_reward = eval_on_split_one_episode(machines, CVr, CVs, 1, test_ep_idx);
        double pair_test_tps = gp_get_last_eval_avg_tps();
        double pair_test_pen_sec = gp_get_last_eval_avg_pen_sec();
        gp_record_end();

        // 与初始化一致: best_r / best_s 取 best pair 在 TRAIN episode 上的指标
        double best_r_reward = pair_train_reward;
        double best_r_tps = pair_train_tps;
        double best_s_reward = pair_train_reward;
        double best_s_tps = pair_train_tps;

        fcurve << (gen + 1) << "," << (train_ep_idx + 1) << "," << (test_ep_idx + 1) << ","
            << best_r_reward << "," << best_r_tps << "," << pair_train_pen_sec << ","
            << best_s_reward << "," << best_s_tps << "," << pair_train_pen_sec << ","
            << mean_r_reward << "," << mean_r_tps << "," << mean_r_pen_sec << ","
            << mean_s_reward << "," << mean_s_tps << "," << mean_s_pen_sec << ","
            << pair_train_reward << "," << pair_train_tps << "," << pair_train_pen_sec << ","
            << pair_test_reward << "," << pair_test_tps << "," << pair_test_pen_sec << "\n";
        fcurve.flush();

        // 旧版输出: 每代追加 best_r 和 best_s 的 reward/fitness
        ftg << best_r_reward << '\n' << best_s_reward << '\n';

        std::cout << "[GenEpisode] gen=" << (gen + 1)
            << " TRAIN episode " << (train_ep_idx + 1) << "/" << total_train_eps
            << " end" << std::endl;

        // Move to next episode for the next generation (wrap-around to keep training forever).
        train_ep_idx = (train_ep_idx + 1) % total_train_eps;
        test_ep_idx = (test_ep_idx + 1) % total_test_eps;
    }

    // final best
    sort(routing, routing + SIZE, [](const tree& a, const tree& b) { return a.fitness > b.fitness; });
    sort(sequencing, sequencing + SIZE, [](const tree& a, const tree& b) { return a.fitness > b.fitness; });
    bestCVr = routing[0];
    bestCVs = sequencing[0];

    if (bestFitness) {
        bestFitness[0] = (std::max)(bestCVr.fitness, bestCVs.fitness);
        bestFitness[1] = 0.0;
    }

    delete[] EP1;
    delete[] EP2;
    delete[] routing;
    delete[] sequencing;
}
