// gpu_kernels.cu   Strictly equivalent, higher parallelism (multi-block per shard, two-stage reduction)
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#include "comfunc.h"       // Machines / DistanceInfo / constants
#include "gpu_bridge.cuh"  // header extern "C"

// Eq.(5) truncation upper bound Tmax (seconds).
// Keep this consistent with the CPU-side setting in CCGPHHFitness.cpp.
#ifndef GP_TMAX_DELAY
#define GP_TMAX_DELAY 100.0
#endif

// ------------------------ helpers ------------------------
#define CUDA_CHK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::fprintf(stderr, "[CUDA] %s failed: %s\n", #call, cudaGetErrorString(_e)); \
        return false; \
    } \
} while (0)

static inline int ceil_div_host(int a, int b) { return (a + b - 1) / b; } // host 端向上取整除法

// ======================== GPU 初始化 ========================
__global__ void __gpu_smoke_kernel(int* d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 8) d[i] = i;
}

bool gpu_init_and_smoketest(bool silent) {
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
        if (!silent) std::fprintf(stderr, "[CUDA] 未检测到 CUDA 设备\n");
        return false;
    }
    CUDA_CHK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    if (!silent) {
        std::printf("[CUDA] 使用设备: %s | CC %d.%d | 全局显存 %.1f GB\n",
            prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    int h[8] = { 0 }; int* d = nullptr;
    CUDA_CHK(cudaMalloc(&d, 8 * sizeof(int)));
    __gpu_smoke_kernel << <1, 8 >> > (d);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        if (!silent) std::fprintf(stderr, "[CUDA] Smoke test kernel 同步失败\n");
        cudaFree(d);
        return false;
    }
    CUDA_CHK(cudaMemcpy(h, d, 8 * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d);
    if (!silent) std::printf("[CUDA] Smoke test: %d %d\n", h[0], h[7]);
    return (h[0] == 0 && h[7] == 7);
}

// ======================== 距离矩阵 (GPU) ========================
__global__ void pairwise_distance_kernel(const Machines* __restrict__ in,
    Machines* __restrict__ out,
    int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N) return;

    double dx = in[i].xloc - in[j].xloc;
    double dy = in[i].yloc - in[j].yloc;
    double d = sqrt(dx * dx + dy * dy);
    if (d < 1.0) {
        d = 1.0;
    }

    out[i].dis[j].id = in[j].id;
    out[i].dis[j].distance = d;
}

bool gpu_compute_pairwise_distance(Machines* machines, int N) {
    if (N <= 0) return true;

    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
        std::fprintf(stderr, "[CUDA] 未检测到 CUDA 设备\n");
        return false;
    }
    CUDA_CHK(cudaSetDevice(0));

    Machines* d_in = nullptr, * d_out = nullptr;
    size_t bytes = sizeof(Machines) * (size_t)N;

    CUDA_CHK(cudaMalloc(&d_in, bytes));
    CUDA_CHK(cudaMalloc(&d_out, bytes));
    CUDA_CHK(cudaMemcpy(d_in, machines, bytes, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_out, machines, bytes, cudaMemcpyHostToDevice)); // 写入 dis[]

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y);

    pairwise_distance_kernel << <grid, block >> > (d_in, d_out, N);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] pairwise_distance kernel 同步失败\n");
        cudaFree(d_in); cudaFree(d_out);
        return false;
    }

    CUDA_CHK(cudaMemcpy(machines, d_out, bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out);
    return true;
}

// CPU 端：按距离升序排序 dis[]
static inline void host_sort_dis(Machines* machines, int N) {
    for (int i = 0; i < N; ++i) {
        std::sort(machines[i].dis, machines[i].dis + N,
            [](const DistanceInfo& a, const DistanceInfo& b) {
                return a.distance < b.distance;
            });
    }
}

void gpu_update_and_sort_distances(Machines* machines, int N) {
    if (gpu_compute_pairwise_distance(machines, N)) {
        host_sort_dis(machines, N);
    }
    else {
        // 回退到 CPU 版本
        updateAndSortDistances(machines, N);
    }
}

// ======================== device 端辅助函数 ========================
static __device__ __forceinline__
double d_rate_from_coord(double ax, double ay, double bx, double by,
    double lambda, double Pt, double Gt, double Gr,
    double N0, double B, double four_pi) {
    double dx = ax - bx, dy = ay - by;
    double d = sqrt(dx * dx + dy * dy);
    if (d < 1.0) d = 1.0;

    double Nsafe = (N0 > 1e-30 ? N0 : 1e-30);
    double Pr = Pt * Gt * Gr * pow(lambda / (four_pi * d), 2.0);
    double SNR = Pr / Nsafe;
    if (SNR < 0.0) SNR = 0.0;
    double R = B * log2(1.0 + SNR);
    if (!isfinite(R) || R < 1e-12) R = 1e-12;
    return R;
}

// ======================== 多 block/分片 的两阶段归约 kernel ========================
// 全局 atomicMax(double) 实现：使用 CAS 循环
static __device__ inline void atomicMaxDouble(double* addr, double val) {
    unsigned long long* uaddr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *uaddr, assumed;
    while (true) {
        double old_val = __longlong_as_double(old);
        if (old_val >= val) break;
        assumed = old;
        unsigned long long desired = __double_as_longlong(val);
        old = atomicCAS(uaddr, assumed, desired);
        if (old == assumed) break;
    }
}

// kernel：每个分片使用多个 block 计算三阶段最大通信时延，并写入 stage_max[shard][3]
__global__ void tcon_stage_accumulate_kernel(const Machines* __restrict__ machines,
    const int* __restrict__ pmacs,
    int pnum,
    //
    double SB_bits, double lambda, double Pt, double Gt, double Gr,
    double N0, double B, int M, double alpha, double beta,
    double four_pi,
    int blocks_per_shard,           // 同一分片使用的 block 数
    double* __restrict__ stage_max) // 大小 pnum*3，初始为 0
{
    int global_blk = blockIdx.x;
    int shard = global_blk / blocks_per_shard;
    int part = global_blk % blocks_per_shard;
    if (shard >= pnum) return;

    const int primary = pmacs[shard];
    const int followers = machines[primary].owned;
    int Rn = followers + 1;
    if (Rn <= 0) return;
    if (Rn > 512) Rn = 512;

    // 动态共享内存：累计每个接收方的三阶段时延（pp/pre/com），最后做 block 内最大值归约
    extern __shared__ double s_mem[];
    double* t_pp = s_mem;        // [Rn]
    double* t_pre = t_pp + Rn;   // [Rn]
    double* t_com = t_pre + Rn;   // [Rn]
    for (int r = threadIdx.x; r < Rn; r += blockDim.x) {
        t_pp[r] = t_pre[r] = t_com[r] = 0.0;
    }
    __syncthreads();

    auto getNodeId = [&](int idx)->int {
        return (idx == 0) ? primary : machines[primary].team[idx - 1];
        };

    // (a) pre-prepare：r=1..Rn-1，按 blocks_per_shard 分块处理
    {
        int total_r = Rn - 1;
        int chunk = (total_r + blocks_per_shard - 1) / blocks_per_shard; // host 端分块大小
        int r_begin = 1 + part * chunk;
        int r_end = min(1 + (part + 1) * chunk, Rn);
        int sid = primary;

        for (int r = r_begin + threadIdx.x; r < r_end; r += blockDim.x) {
            int rid = getNodeId(r);
            // Use per-node transmit power from sender 'sid' if available; fall back to default Pt.
            double Pt_sender = machines[sid].pt;
            if (!(Pt_sender > 0.0) || !isfinite(Pt_sender)) Pt_sender = Pt;

            double R = d_rate_from_coord(machines[sid].xloc, machines[sid].yloc,
                machines[rid].xloc, machines[rid].yloc,
                lambda, Pt_sender, Gt, Gr, N0, B, four_pi);
            atomicAdd(&t_pp[r], SB_bits / R);
        }
        __syncthreads();
    }

    // (b)(c) prepare/commit：对所有 (s!=r) 对，按 blocks_per_shard 分块处理
    long long pairs_total = (long long)Rn * (Rn - 1);
    long long chunk = (pairs_total + blocks_per_shard - 1) / blocks_per_shard;
    long long begin = part * chunk;
    long long end = min(begin + chunk, pairs_total);

    for (long long idx = begin + threadIdx.x; idx < end; idx += blockDim.x) {
        int s = (int)(idx / (Rn - 1));
        int r = (int)(idx % (Rn - 1));
        if (r >= s) r++;

        int sid = getNodeId(s);
        int rid = getNodeId(r);
        // Use per-node transmit power from sender 'sid' if available; fall back to default Pt.
        double Pt_sender = machines[sid].pt;
        if (!(Pt_sender > 0.0) || !isfinite(Pt_sender)) Pt_sender = Pt;

        double R = d_rate_from_coord(machines[sid].xloc, machines[sid].yloc,
            machines[rid].xloc, machines[rid].yloc,
            lambda, Pt_sender, Gt, Gr, N0, B, four_pi);
        double t = SB_bits / R;

        atomicAdd(&t_pre[r], t);
        atomicAdd(&t_com[r], t);
    }
    __syncthreads();

    // block 内归约求最大值
    double max_pp = 0.0, max_pre = 0.0, max_com = 0.0;
    for (int r = threadIdx.x; r < Rn; r += blockDim.x) {
        if (t_pp[r] > max_pp)  max_pp = t_pp[r];
        if (t_pre[r] > max_pre) max_pre = t_pre[r];
        if (t_com[r] > max_com) max_com = t_com[r];
    }
    __shared__ double red_pp[256], red_pre[256], red_com[256];
    red_pp[threadIdx.x] = max_pp;
    red_pre[threadIdx.x] = max_pre;
    red_com[threadIdx.x] = max_com;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red_pp[threadIdx.x] = fmax(red_pp[threadIdx.x], red_pp[threadIdx.x + stride]);
            red_pre[threadIdx.x] = fmax(red_pre[threadIdx.x], red_pre[threadIdx.x + stride]);
            red_com[threadIdx.x] = fmax(red_com[threadIdx.x], red_com[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // 写回全局：stage_max[shard*3 + {0,1,2}]
        atomicMaxDouble(&stage_max[shard * 3 + 0], red_pp[0]);
        atomicMaxDouble(&stage_max[shard * 3 + 1], red_pre[0]);
        atomicMaxDouble(&stage_max[shard * 3 + 2], red_com[0]);
    }
}

// Finalize per-shard consensus delay (paper-aligned, Eq.(5)):
//   T_i^con = min(T_i^com, Tmax) + min(T_i^cal, Tmax)
__global__ void tcon_finalize_kernel(const Machines* __restrict__ machines,
    const int* __restrict__ pmacs,
    int pnum, int M, double alpha, double beta,
    const double* __restrict__ stage_max,
    double* __restrict__ Tcon_out)
{
    int shard = blockIdx.x * blockDim.x + threadIdx.x;
    if (shard >= pnum) return;

    int primary = pmacs[shard];
    int followers = machines[primary].owned;
    int Rn = followers + 1;
    if (Rn <= 0) { Tcon_out[shard] = 0.0; return; }

    double T_prop = stage_max[shard * 3 + 0] + stage_max[shard * 3 + 1] + stage_max[shard * 3 + 2];

    const double C_primary = (double)M * alpha + (2.0 * M + 4.0 * (Rn - 1)) * beta;
    const double C_replica = (double)M * alpha + (1.0 * M + 4.0 * (Rn - 1)) * beta;

    double cal0 = machines[primary].cal; if (!isfinite(cal0) || cal0 <= 1e-12) cal0 = 1e-12;
    double T_cal_primary = C_primary / cal0;

    double T_cal_max_rep = 0.0;
    for (int i = 0; i < followers; ++i) {
        int rid = machines[primary].team[i];
        double ci = machines[rid].cal; if (!isfinite(ci) || ci <= 1e-12) ci = 1e-12;
        double v = C_replica / ci;
        if (v > T_cal_max_rep) T_cal_max_rep = v;
    }
    double T_val = (T_cal_primary > T_cal_max_rep) ? T_cal_primary : T_cal_max_rep;

    // Eq.(5): truncate with Tmax
    const double Tmax = (double)GP_TMAX_DELAY;
    double tcom = (T_prop < Tmax ? T_prop : Tmax);
    double tcal = (T_val < Tmax ? T_val : Tmax);
    Tcon_out[shard] = tcom + tcal;
}

// ======================== 计算总吞吐量（GPU 版本） ========================
double compute_total_throughput_v2_gpu(const Machines machines[], int machineNum,
    int pmacs[], int pnum, double Treconfig, double T_lc_con) {
    if (pnum <= 0) return 0.0;

    Machines* d_machines = nullptr;
    int* d_pmacs = nullptr;
    double* d_Tcon = nullptr;
    double* d_stage_max = nullptr;

    if (cudaMalloc(&d_machines, sizeof(Machines) * machineNum) != cudaSuccess) return 0.0;
    if (cudaMalloc(&d_pmacs, sizeof(int) * pnum) != cudaSuccess) { cudaFree(d_machines); return 0.0; }
    if (cudaMalloc(&d_Tcon, sizeof(double) * pnum) != cudaSuccess) { cudaFree(d_machines); cudaFree(d_pmacs); return 0.0; }
    if (cudaMalloc(&d_stage_max, sizeof(double) * pnum * 3) != cudaSuccess) { cudaFree(d_machines); cudaFree(d_pmacs); cudaFree(d_Tcon); return 0.0; }

    if (cudaMemcpy(d_machines, machines, sizeof(Machines) * machineNum, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_machines); cudaFree(d_pmacs); cudaFree(d_Tcon); cudaFree(d_stage_max); return 0.0;
    }
    if (cudaMemcpy(d_pmacs, pmacs, sizeof(int) * pnum, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_machines); cudaFree(d_pmacs); cudaFree(d_Tcon); cudaFree(d_stage_max); return 0.0;
    }
    cudaMemset(d_stage_max, 0, sizeof(double) * pnum * 3);

    // 计算 Rn_max 以确定动态共享内存大小
    int Rn_max = 1;
    for (int i = 0; i < pnum; ++i) {
        int primary = pmacs[i];
        int followers = machines[primary].owned;
        int Rn = followers + 1;
        if (Rn > Rn_max) Rn_max = Rn;
    }
    if (Rn_max > 512) Rn_max = 512;

    // blocks_per_shard：保证 GPU 侧有足够并行度（目标 ~24 个 block 总量）
    int blocks_per_shard = (pnum >= 24) ? 1 : (std::min)(16, ceil_div_host(24, (std::max)(1, pnum)));
    dim3 grid1(pnum * blocks_per_shard), block1(256);
    size_t shBytes = sizeof(double) * (size_t)Rn_max * 3;
    double four_pi = 4.0 * (double)M_PI;

    // kernel：累计三阶段时延并取最大
    tcon_stage_accumulate_kernel << <grid1, block1, shBytes >> > (
        d_machines, d_pmacs, pnum,
        SB_bits, lambda, Pt, Gt, Gr, N, B, M, alpha, beta,
        four_pi, blocks_per_shard,
        d_stage_max
        );

    // kernel：生成每个分片的 Tcon
    dim3 grid2(ceil_div_host(pnum, 128)), block2(128);
    tcon_finalize_kernel << <grid2, block2 >> > (
        d_machines, d_pmacs, pnum, M, alpha, beta,
        d_stage_max, d_Tcon
        );

    if (cudaDeviceSynchronize() != cudaSuccess) {
        // 回退到 CPU
        cudaFree(d_machines); cudaFree(d_pmacs); cudaFree(d_Tcon); cudaFree(d_stage_max);
        double max_Tcon = 0.0;
        for (int i = 0; i < pnum; ++i) {
            double Tcon = estimate_Tcon_k(pmacs[i], machines);
            if (Tcon > max_Tcon) max_Tcon = Tcon;
        }
        double T_rec = 3.0 * T_lc_con + max_Tcon + Treconfig;
        double Tepoch = tps_r * max_Tcon + T_rec;
        double T_total = 0.0;
        for (int i = 0; i < pnum; ++i) T_total += (tps_r * SB / ST) / Tepoch;
        return T_total;
    }

    std::vector<double> hTcon(pnum, 0.0);
    cudaMemcpy(hTcon.data(), d_Tcon, sizeof(double) * pnum, cudaMemcpyDeviceToHost);

    cudaFree(d_machines); cudaFree(d_pmacs); cudaFree(d_Tcon); cudaFree(d_stage_max);

    for (int i = 0; i < pnum; ++i) {
        if (!(hTcon[i] > 0.0) || !std::isfinite(hTcon[i])) {
            hTcon[i] = estimate_Tcon_k(pmacs[i], machines); // 回退到 CPU 估算
        }
    }
    double max_Tcon = 0.0;
    for (int i = 0; i < pnum; ++i) if (hTcon[i] > max_Tcon) max_Tcon = hTcon[i];

    double T_rec = 3.0 * T_lc_con + max_Tcon + Treconfig;
    double Tepoch = tps_r * max_Tcon + T_rec;
    double T_total = 0.0;
    for (int i = 0; i < pnum; ++i) T_total += (tps_r * SB / ST) / Tepoch;
    return T_total;
}

// ================== GPU 上下文（缓存/复用） ==================
struct GpuContext {
    Machines* d_machines_in = nullptr; // 输入 machines（用于 Tcon 计算）
    Machines* d_machines_out = nullptr; // 输出 machines（写回 dis[]）
    int* d_pmacs = nullptr; // 分片主节点列表
    double* d_Tcon = nullptr; // 每分片 Tcon
    double* d_stage_max = nullptr; // 三阶段最大值缓存：pnum*3
    int capN = 0;
    int capP = 0;
    int use_async = 0;
    cudaStream_t stream = nullptr;
};

static inline bool gpu_alloc_ctx_bufs(GpuContext* ctx, int maxN, int maxP) {
    ctx->capN = maxN; ctx->capP = maxP;
    if (cudaMalloc(&ctx->d_machines_in, sizeof(Machines) * maxN) != cudaSuccess) return false;
    if (cudaMalloc(&ctx->d_machines_out, sizeof(Machines) * maxN) != cudaSuccess) return false;
    if (cudaMalloc(&ctx->d_pmacs, sizeof(int) * maxP) != cudaSuccess) return false;
    if (cudaMalloc(&ctx->d_Tcon, sizeof(double) * maxP) != cudaSuccess) return false;
    if (cudaMalloc(&ctx->d_stage_max, sizeof(double) * maxP * 3) != cudaSuccess) return false;
    return true;
}

GpuContext* gpu_ctx_create(int maxN, int maxP, int use_async) {
    if (maxN <= 0 || maxP <= 0) return nullptr;
    cudaSetDevice(0);
    GpuContext* ctx = new GpuContext();
    ctx->use_async = use_async ? 1 : 0;
    if (ctx->use_async) cudaStreamCreate(&ctx->stream);
    if (!gpu_alloc_ctx_bufs(ctx, maxN, maxP)) {
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        if (ctx->d_machines_in)  cudaFree(ctx->d_machines_in);
        if (ctx->d_machines_out) cudaFree(ctx->d_machines_out);
        if (ctx->d_pmacs)        cudaFree(ctx->d_pmacs);
        if (ctx->d_Tcon)         cudaFree(ctx->d_Tcon);
        if (ctx->d_stage_max)    cudaFree(ctx->d_stage_max);
        delete ctx;
        return nullptr;
    }
    return ctx;
}

void gpu_ctx_destroy(GpuContext* ctx) {
    if (!ctx) return;
    if (ctx->d_machines_in)  cudaFree(ctx->d_machines_in);
    if (ctx->d_machines_out) cudaFree(ctx->d_machines_out);
    if (ctx->d_pmacs)        cudaFree(ctx->d_pmacs);
    if (ctx->d_Tcon)         cudaFree(ctx->d_Tcon);
    if (ctx->d_stage_max)    cudaFree(ctx->d_stage_max);
    if (ctx->stream)         cudaStreamDestroy(ctx->stream);
    delete ctx;
}

void gpu_update_and_sort_distances_ctx(GpuContext* ctx, Machines* machines, int N) {
    if (!ctx || N <= 0 || N > ctx->capN) { updateAndSortDistances(machines, N); return; }
    cudaSetDevice(0);
    auto st = ctx->use_async
        ? cudaMemcpyAsync(ctx->d_machines_in, machines, sizeof(Machines) * N, cudaMemcpyHostToDevice, ctx->stream)
        : cudaMemcpy(ctx->d_machines_in, machines, sizeof(Machines) * N, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) { updateAndSortDistances(machines, N); return; }

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    pairwise_distance_kernel << <grid, block, 0, ctx->stream >> > (ctx->d_machines_in, ctx->d_machines_out, N);

    if (ctx->use_async) cudaStreamSynchronize(ctx->stream);
    else                cudaDeviceSynchronize();

    st = ctx->use_async
        ? cudaMemcpyAsync(machines, ctx->d_machines_out, sizeof(Machines) * N, cudaMemcpyDeviceToHost, ctx->stream)
        : cudaMemcpy(machines, ctx->d_machines_out, sizeof(Machines) * N, cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) { updateAndSortDistances(machines, N); return; }

    if (ctx->use_async) cudaStreamSynchronize(ctx->stream);

    // CPU 端排序（保持与原始逻辑一致）
    for (int i = 0; i < N; ++i) {
        std::sort(machines[i].dis, machines[i].dis + N,
            [](const DistanceInfo& a, const DistanceInfo& b) { return a.distance < b.distance; });
    }
}

double compute_total_throughput_v2_gpu_ctx(GpuContext* ctx,
    const Machines machines[], int machineNum,
    int pmacs[], int pnum, double Treconfig, double T_lc_con) {
    if (!ctx || machineNum <= 0 || pnum <= 0 || machineNum > ctx->capN || pnum > ctx->capP) {
        // 回退到 CPU
        double max_Tcon = 0.0;
        for (int i = 0; i < pnum; ++i) {
            double Tcon = estimate_Tcon_k(pmacs[i], machines);
            if (Tcon > max_Tcon) max_Tcon = Tcon;
        }
        double T_rec = 3.0 * T_lc_con + max_Tcon + Treconfig;
        double Tepoch = tps_r * max_Tcon + T_rec;
        double T_total = 0.0;
        for (int i = 0; i < pnum; ++i) T_total += (tps_r * SB / ST) / Tepoch;
        return T_total;
    }

    cudaSetDevice(0);
    auto st1 = ctx->use_async
        ? cudaMemcpyAsync(ctx->d_machines_in, machines, sizeof(Machines) * machineNum, cudaMemcpyHostToDevice, ctx->stream)
        : cudaMemcpy(ctx->d_machines_in, machines, sizeof(Machines) * machineNum, cudaMemcpyHostToDevice);
    auto st2 = ctx->use_async
        ? cudaMemcpyAsync(ctx->d_pmacs, pmacs, sizeof(int) * pnum, cudaMemcpyHostToDevice, ctx->stream)
        : cudaMemcpy(ctx->d_pmacs, pmacs, sizeof(int) * pnum, cudaMemcpyHostToDevice);
    if (st1 != cudaSuccess || st2 != cudaSuccess) {
        // 回退到 CPU
        double max_Tcon = 0.0;
        for (int i = 0; i < pnum; ++i) {
            double Tcon = estimate_Tcon_k(pmacs[i], machines);
            if (Tcon > max_Tcon) max_Tcon = Tcon;
        }
        double T_rec = 3.0 * T_lc_con + max_Tcon + Treconfig;
        double Tepoch = tps_r * max_Tcon + T_rec;
        double T_total = 0.0;
        for (int i = 0; i < pnum; ++i) T_total += (tps_r * SB / ST) / Tepoch;
        return T_total;
    }

    // 置零 stage_max
    cudaMemsetAsync(ctx->d_stage_max, 0, sizeof(double) * pnum * 3, ctx->stream);

    // 计算 Rn_max 以确定动态共享内存大小
    int Rn_max = 1;
    for (int i = 0; i < pnum; ++i) {
        int primary = pmacs[i];
        int followers = machines[primary].owned;
        int Rn = followers + 1;
        if (Rn > Rn_max) Rn_max = Rn;
    }
    if (Rn_max > 512) Rn_max = 512;

    // blocks_per_shard：保证并行度（目标 ~24 个 block 总量）
    int blocks_per_shard = (pnum >= 24) ? 1 : (std::min)(16, ceil_div_host(24, (std::max)(1, pnum)));
    dim3 grid1(pnum * blocks_per_shard), block1(256);
    size_t shBytes = sizeof(double) * (size_t)Rn_max * 3;
    double four_pi = 4.0 * (double)M_PI;  // 与 CPU 端一致

    // kernel：累计三阶段最大值
    tcon_stage_accumulate_kernel << <grid1, block1, shBytes, ctx->stream >> > (
        ctx->d_machines_in, ctx->d_pmacs, pnum,
        SB_bits, lambda, Pt, Gt, Gr, N, B, M, alpha, beta,
        four_pi, blocks_per_shard,
        ctx->d_stage_max
        );

    // kernel：生成每分片 Tcon
    dim3 grid2(ceil_div_host(pnum, 128)), block2(128);
    tcon_finalize_kernel << <grid2, block2, 0, ctx->stream >> > (
        ctx->d_machines_in, ctx->d_pmacs, pnum, M, alpha, beta,
        ctx->d_stage_max, ctx->d_Tcon
        );

    if (ctx->use_async) cudaStreamSynchronize(ctx->stream);
    else                cudaDeviceSynchronize();

    std::vector<double> hTcon(pnum, 0.0);
    auto st3 = ctx->use_async
        ? cudaMemcpyAsync(hTcon.data(), ctx->d_Tcon, sizeof(double) * pnum, cudaMemcpyDeviceToHost, ctx->stream)
        : cudaMemcpy(hTcon.data(), ctx->d_Tcon, sizeof(double) * pnum, cudaMemcpyDeviceToHost);
    if (ctx->use_async) cudaStreamSynchronize(ctx->stream);
    if (st3 != cudaSuccess) {
        // 回退到 CPU
        double max_Tcon = 0.0;
        for (int i = 0; i < pnum; ++i) {
            double Tcon = estimate_Tcon_k(pmacs[i], machines);
            if (Tcon > max_Tcon) max_Tcon = Tcon;
        }
        double T_rec = 3.0 * T_lc_con + max_Tcon + Treconfig;
        double Tepoch = tps_r * max_Tcon + T_rec;
        double T_total = 0.0;
        for (int i = 0; i < pnum; ++i) T_total += (tps_r * SB / ST) / Tepoch;
        return T_total;
    }

    for (int i = 0; i < pnum; ++i) {
        if (!(hTcon[i] > 0.0) || !std::isfinite(hTcon[i])) {
            hTcon[i] = estimate_Tcon_k(pmacs[i], machines);
        }
    }

    double max_Tcon = 0.0;
    for (int i = 0; i < pnum; ++i) if (hTcon[i] > max_Tcon) max_Tcon = hTcon[i];

    double T_rec = 3.0 * T_lc_con + max_Tcon + Treconfig;
    double Tepoch = tps_r * max_Tcon + T_rec;
    double T_total = 0.0;
    for (int i = 0; i < pnum; ++i) T_total += (tps_r * SB / ST) / Tepoch;
    return T_total;
}
