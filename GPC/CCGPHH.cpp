// CCGPHH.cpp
#include "comfunc.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>

#include "gpu_bridge.cuh"

// dataset control API (implemented in CCGPHHFitness.cpp)
extern "C" void dataset_set_dirs(const char* train_dir, const char* test_dir);
extern "C" void dataset_set_split(int split01);                 // 0=train, 1=test
extern "C" int  dataset_get_total_episodes(int split01);
extern "C" int  dataset_get_steps_per_episode(int split01);

using namespace std;

int main(int argc, char* argv[]) {
    // ===== (optional) GPU init =====
    // 
    gpu_init_and_smoketest(/*silent=*/false);
    GpuContext* gctx = gpu_ctx_create(/*maxN=*/machineNum + 5, /*maxP=*/machineNum, /*useAsync=*/1);

    // ===== dataset dirs =====
    // 默认就是 dataset/train 和 dataset/test
    const char* train_dir = "../dataset/train";
    const char* test_dir = "../dataset/test";
    dataset_set_dirs(train_dir, test_dir);

    cout << "[Dataset] train episodes=" << dataset_get_total_episodes(0)
        << ", steps/ep=" << dataset_get_steps_per_episode(0) << "\n";
    cout << "[Dataset] test  episodes=" << dataset_get_total_episodes(1)
        << ", steps/ep=" << dataset_get_steps_per_episode(1) << "\n";

    // ===== allocate machines =====
    Machines* machines = new Machines[machineNum + 5];

    // ===== run GP training =====
    // 注意：GPHH.cpp 内部每代都会把 pair_train/pair_test 写入 gp_convergence.csv
    dataset_set_split(0); // TRAIN
    tree bestCVr, bestCVs;
    double bestFitness[2] = { 0.0, 0.0 };

    // SIZE/Itr
    int SIZE = 200;
    int Itr = 999;

    

    GPHH(machines, SIZE, Itr, bestFitness, bestCVr, bestCVs);

    // ===== final test =====
    dataset_set_split(1); // TEST
    double final_test = CCGPHHFitness(machines, bestCVr, bestCVs);
    dataset_set_split(0);

    cout << "\n[FINAL] bestFitness(train proxy)=" << bestFitness[0] << "\n";
    cout << "[FINAL TEST] pair fitness=" << final_test << "\n";

    cout << "\nBest primary-node rule:\n";
    showstr(bestCVr, 0);
    cout << "\n\nBest follower-allocation rule:\n";
    showstr(bestCVs, 1);
    cout << "\n";

    // save best rules (optional)
    ofstream fout(result_path("best_rules.txt"), ios::out);
    if (fout.good()) {
        fout << "FINAL_TEST " << final_test << "\n";
        fout << "PRIMARY_RULE\n";
        // showstr 打到 stdout，这里不强制转储树字符串（你工程里如果有序列化函数可替换）
        fout << "(see stdout)\n";
        fout << "FOLLOWER_RULE\n";
        fout << "(see stdout)\n";
    }

    delete[] machines;

    // ===== (optional) GPU destroy =====
    gpu_ctx_destroy(gctx);
    return 0;
}
