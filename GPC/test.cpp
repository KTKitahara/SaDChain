// test.cpp
#include "comfunc.h"
#include <iostream>
#include <fstream>
#include <cstdio>

#include "gpu_bridge.cuh"

// dataset control API (implemented in CCGPHHFitness.cpp)
extern "C" void dataset_set_dirs(const char* train_dir, const char* test_dir);
extern "C" void dataset_set_split(int split01);                 // 0=train, 1=test
extern "C" int  dataset_get_total_episodes(int split01);
extern "C" int  dataset_get_steps_per_episode(int split01);

using namespace std;

int main(int argc, char* argv[]) {
    // ===== (optional) GPU init =====
    gpu_init_and_smoketest(/*silent=*/false);
    GpuContext* gctx = gpu_ctx_create(/*maxN=*/machineNum + 5, /*maxP=*/machineNum, /*useAsync=*/1);

    // ===== dataset dirs =====
    dataset_set_dirs("dataset/train", "dataset/test");

    cout << "[Dataset] test episodes=" << dataset_get_total_episodes(1)
        << ", steps/ep=" << dataset_get_steps_per_episode(1) << "\n";

    Machines* machines = new Machines[machineNum + 5];

    // ---------------------------
    // MODE A: train once -> test
    // ---------------------------
    tree bestCVr, bestCVs;
    double bestFitness[2] = { 0.0, 0.0 };

    int SIZE = 500;
    int Itr = 200;

    dataset_set_split(0); // TRAIN
    GPHH(machines, SIZE, Itr, bestFitness, bestCVr, bestCVs);

    dataset_set_split(1); // TEST
    double testFitness = CCGPHHFitness(machines, bestCVr, bestCVs);

    cout << "\n[TEST] pair fitness=" << testFitness << "\n";
    cout << "\nBest primary-node rule:\n";
    showstr(bestCVr, 0);
    cout << "\n\nBest follower-allocation rule:\n";
    showstr(bestCVs, 1);
    cout << "\n";

    // save test result
    ofstream fout(result_path("test_result.txt"), ios::out);
    if (fout.good()) {
        fout << "TEST_PAIR_FITNESS " << testFitness << "\n";
    }

    delete[] machines;
    gpu_ctx_destroy(gctx);
    return 0;
}
