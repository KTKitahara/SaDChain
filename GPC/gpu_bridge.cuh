#pragma once
#include "comfunc.h"

#ifdef __cplusplus
extern "C" {
#endif

    // GPU 初始化（可选打印设备信息）
    bool gpu_init_and_smoketest(bool silent);

    // GPU 计算两两距离，并写回 machines[i].dis[j].id / machines[i].dis[j].distance
    // 成功返回 true
    bool gpu_compute_pairwise_distance(Machines* machines, int N);

    // 封装版：GPU 优先；如果 GPU 不可用则退回 CPU 版 updateAndSortDistances
    void gpu_update_and_sort_distances(Machines* machines, int N);

    // GPU throughput（与论文口径对齐）:
    //   - Treconfig 传入 T_algo（算法运行时间）
    //   - T_lc_con 传入本地共识延迟（LC consensus delay）
    // GPU 内部计算 max_Tcon = max_i T_i^con，然后：
    //   T_rec   = 3 * T_lc_con + max_Tcon + T_algo
    //   T_epoch = tps_r * max_Tcon + T_rec
    // 输出总 TPS
    double compute_total_throughput_v2_gpu(const Machines machines[], int machineNum,
        int pmacs[], int pnum, double Treconfig, double T_lc_con);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // GPU 上下文（用于复用显存缓冲，减少频繁分配/释放的开销）
    typedef struct GpuContext GpuContext;

    // 创建/销毁上下文：
    //   maxN: 支持的最大节点数
    //   maxP: 支持的最大主节点数（分片数上限）
    //   use_async: 是否使用异步流（0/1）
    GpuContext* gpu_ctx_create(int maxN, int maxP, /*useAsync*/ int use_async);
    void        gpu_ctx_destroy(GpuContext* ctx);

    // ctx 版本：在 ctx 的缓冲上进行距离计算与排序；失败则退回 CPU
    void gpu_update_and_sort_distances_ctx(GpuContext* ctx, Machines* machines, int N);

    // ctx 版本：在 ctx 的缓冲上计算 throughput；失败则退回 CPU
    double compute_total_throughput_v2_gpu_ctx(GpuContext* ctx,
        const Machines machines[], int machineNum,
        int pmacs[], int pnum, double Treconfig, double T_lc_con);

#ifdef __cplusplus
}
#endif
