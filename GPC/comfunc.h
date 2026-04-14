#pragma once

// ===================== Minimal standard includes =====================
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>


// ===================== Output directory helpers =====================
// All result files will be placed under ./result/ (relative to working directory)
#if defined(_WIN32)
#  include <direct.h>   // _mkdir
#else
#  include <sys/stat.h> // mkdir
#  include <sys/types.h>
#endif

static inline void ensure_dir_simple(const std::string& dir) {
#if defined(_WIN32)
    _mkdir(dir.c_str());
#else
    mkdir(dir.c_str(), 0755);
#endif
}

static inline std::string join_path_simple(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    char last = a.back();
    if (last == '/' || last == '\\') return a + b;
    return a + "/" + b;
}

static inline std::string result_path(const std::string& filename) {
    ensure_dir_simple("result");
    return join_path_simple("result", filename);
}


// 兼容：部分平台/编译选项不提供 M_PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===================== Legacy macros / constants =====================
#define ElemType char

#ifndef INT_MIN
#define INT_MIN (-0x80000000)
#endif

// ===================== Global constants (keep your original names) =====================
const int machineNum = 200;
const int MAXN = 1000;
const int WARMJOB = 100;
const int mmove = 10;
const int mnode = 5;

const float  threshold = 0.2f;               // 主节点个数阈值
const int    n0 = 160;// 每组非主节点个数限制
static constexpr int limitednode = n0;              // 每组非主节点个数限制
const double priority_drop_threshold = 0.01; // 主节点停止生成阈值

const double mcal = 4e9;

const int dbmorw = 0;//0-w,1-dbm

// ================= Security penalty config (Eq.11 / Eq.14 / Eq.15) =================
static constexpr double c_xi = 5e9;
static constexpr double lambda_sec = 1.0;
static constexpr double gamma_sec = 1e6;    // 

// ================= Consensus cap (Eq.5) =================
static constexpr double T_max = 100.0; // seconds

// Eq.(11): Ris_i(t) = c_xi / (c_i(t) + c_xi)
static inline double Ris_from_cal(double cal) {
    return c_xi / (cal + c_xi);
}

// ===================== Throughput model constants =====================
const double SB = 8000000;   // 区块大小（bytes）
const double ST = 64;
const int    tps_r = 1000;
const int    M = 3;
const int    alpha = 2;
const int    beta = 1;
extern int testallflag;

// 通信模型常量
const double lambda = 3e8 / 868e6;
const double Pt = std::pow(10.0, 14.0 / 10.0) / 1000.0;
const double Gt = 1.0, Gr = 1.0;
const double N = std::pow(10.0, -100.0 / 10.0) / 1000.0;//环境噪声
const double B = 20e6;
const double SB_bits = SB * 8.0;

// GP 参数
const int dep1 = 7, dep2 = 7;

// ===================== Types =====================
using namespace std;

double max(double a, double b);
double min(double a, double b);
int possion(int Lambda);

//typedef struct {
//    double x;
//    double y;
//    double cal;
//    double attack_prob;
//} BaseStation;

typedef struct {
    double distance;
    int id;  // 与哪台机器（通常用“下标”）
} DistanceInfo;

typedef struct _machines {
    double priority;
    int    id;
    double xloc;
    double yloc;
    double cal;
    double pt;
    double atkprob;      // Ris_i(t)
    int    fprime;       // 0非主 1主 2已选
    int    owned;
    int    team[machineNum];
    DistanceInfo dis[machineNum];
} Machines;

// ===================== Security penalty (Eq.14) =====================
static inline double Pen_sec_from_shards(const Machines* machines, const int* pmacs, int pnum) {
    std::vector<int> primaries;
    primaries.reserve((std::max)(0, pnum));
    for (int i = 0; i < pnum; ++i) {
        if (pmacs[i] >= 0) primaries.push_back(pmacs[i]);
    }
    const int K = (int)primaries.size();
    if (K <= 0) return 0.0;

    std::vector<double> r_bar(K, 0.0);
    for (int si = 0; si < K; ++si) {
        const int primary = primaries[si];

        double sum_risk = machines[primary].atkprob;
        int cnt = 1;

        for (int j = 0; j < machines[primary].owned && j < machineNum; ++j) {
            const int nid = machines[primary].team[j];
            if (nid < 0) break;
            sum_risk += machines[nid].atkprob;
            ++cnt;
        }
        r_bar[si] = sum_risk / (double)(std::max)(1, cnt);
    }

    double r_global = 0.0;
    for (double v : r_bar) r_global += v;
    r_global /= (double)K;

    double var_term = 0.0;
    for (double v : r_bar) {
        const double diff = v - r_global;
        var_term += diff * diff;
    }
    var_term /= (double)K;

    double primary_term = 0.0;
    for (int si = 0; si < K; ++si) primary_term += machines[primaries[si]].atkprob;
    primary_term /= (double)K;

    return var_term + lambda_sec * primary_term;
}

// ===================== Distance helpers (必须在头文件定义，避免 LNK2019) =====================

// 统一距离计算：供 LowLevel_heuristics1 / 其他模块调用
static inline double calcDistance(const Machines& a, const Machines& b) {
    const double dx = a.xloc - b.xloc;
    const double dy = a.yloc - b.yloc;
	double distance = std::sqrt(dx * dx + dy * dy);
    if (distance < 1) {
        return 1;
    }		
    else
        return distance;
}

// 计算每台机器到所有机器的距离并排序（升序）
// 注意：只计算 [0..machinesNum-1] 的有效节点；剩余槽位填充 INF 以防误用。
static inline void updateAndSortDistances(Machines machines[], int machinesNum) {
    if (!machines || machinesNum <= 0) return;
    const double INF = (std::numeric_limits<double>::max)();

    for (int i = 0; i < machinesNum; ++i) {
        for (int j = 0; j < machinesNum; ++j) {
            machines[i].dis[j].id = j;
            machines[i].dis[j].distance = (i == j) ? 0.0 : calcDistance(machines[i], machines[j]);
        }
        for (int j = machinesNum; j < machineNum; ++j) {
            machines[i].dis[j].id = -1;
            machines[i].dis[j].distance = INF;
        }
        std::sort(machines[i].dis, machines[i].dis + machinesNum,
            [](const DistanceInfo& a, const DistanceInfo& b) {
                return a.distance < b.distance;
            }
        );
    }
}

// ===================== GP tree structures (关键：恢复“带函数体”的版本，避免 LNK2019) =====================
struct node {
    int a;
    int d;//深度
    int t;//左右方向长树
    node* l;//左子节点
    node* r;//右子结点
    node* f;//父节点

    node() {
        a = rand() % (17 - 10) + 10; // [10,17)
        d = 1;
        t = 0;
        r = NULL;
        l = NULL;
        f = NULL;
    }

    node(int v, node* fa, int u) {
        a = v;
        f = fa;
        d = (fa->d) + 1;
        t = u;
        r = NULL;
        l = NULL;
    }
};

struct tree {
    node* root;
    double fitness;

    tree() {
        root = new node();
        fitness = 0.0;
    }
    // ★析构：释放整棵树
    ~tree() {
        deleteT(root);
        root = nullptr;
    }

    // ★拷贝构造：深拷贝整棵树
    tree(const tree& other) {
        fitness = other.fitness;
        root = copy_tree(other.root, nullptr);
    }

    // ★拷贝赋值：深拷贝（copy-and-swap 写法更安全）
    tree& operator=(const tree& other) {
        if (this == &other) return *this;
        tree tmp(other);
        this->swap(tmp);
        return *this;
    }

    // ★移动构造：偷指针，避免不必要拷贝
    tree(tree&& other) noexcept {
        root = other.root;
        fitness = other.fitness;
        other.root = nullptr;
        other.fitness = 0.0;
    }

    // ★移动赋值
    tree& operator=(tree&& other) noexcept {
        if (this == &other) return *this;
        deleteT(root);
        root = other.root;
        fitness = other.fitness;
        other.root = nullptr;
        other.fitness = 0.0;
        return *this;
    }

    void swap(tree& other) noexcept {
        std::swap(root, other.root);
        std::swap(fitness, other.fitness);
    }
    // 节点个数
    int node_num(node* r) {
        if (r == NULL) return 0;
        return 1 + node_num(r->l) + node_num(r->r);
    }

    // 随机抽取节点
    node* get_node_random(node* r) {
        int sum = node_num(r);
        int id = rand() % sum + 1;
        node* ans = NULL;
        get_node_random(r, &id, &ans);
        return ans;
    }

    void get_node_random(node* r, int* id, node** ans) {
        if (r == NULL) return;
        (*id)--;
        if (*id == 0) {
            *ans = r;
            return;
        }
        get_node_random(r->l, id, ans);
        get_node_random(r->r, id, ans);
    }

    // 可用S节点个数
    int s_num(node* r, int d) {
        if (r == NULL) return 0;
        if ((r->l != NULL && r->r != NULL) || r->d >= d) return s_num(r->l, d) + s_num(r->r, d);
        if (r->l == NULL || r->r == NULL)
            return (r->a >= 10 && r->a <= 16) + s_num(r->l, d) + s_num(r->r, d);
        return s_num(r->l, d) + s_num(r->r, d);
    }

    // 随机抽取可用S节点
    node* get_s_random(node* r, int d) {
        int sum = s_num(r, d);
        if (sum == 0) return NULL;
        int id = rand() % sum + 1;
        node* ans = NULL;
        get_s_random(r, &id, &ans, d);
        return ans;
    }

    void get_s_random(node* r, int* id, node** ans, int d) {
        if (r == NULL || (r->a >= 0 && r->a <= 9) || r->d >= d) return;
        if (r->l == NULL || r->r == NULL) {
            (*id)--;
            if (*id == 0) {
                (*ans) = r;
                return;
            }
        }
        get_s_random(r->l, id, ans, d);
        get_s_random(r->r, id, ans, d);
    }

    // 随机生成一棵树
    void construct(double rate, int Nstart, int Nend, int dep, node* r) {
        if (r == NULL) r = new node();

        while (1) {
            node* f = get_s_random(r, dep);
            if (f == NULL) break;

            if (f->d == dep - 1) {
                int v = Nstart + rand() % (Nend - Nstart);
                if (f->l != NULL) f->r = new node(v, f, 1);
                else if (f->r != NULL) f->l = new node(v, f, 0);
                else {
                    if (rand() % 2 == 0) f->l = new node(v, f, 0);
                    else                 f->r = new node(v, f, 1);
                }
            }
            else {
                double rd = 1.0 * rand() / RAND_MAX;

                if (rd < rate) { // 生长非叶子
                    int a = rand() % 7 + 10; // [10,17)
                    if (f->l != NULL) f->r = new node(a, f, 1);
                    else if (f->r != NULL) f->l = new node(a, f, 0);
                    else {
                        if (rand() % 2 == 0) f->l = new node(a, f, 0);
                        else                 f->r = new node(a, f, 1);
                    }
                }
                else { // 生长叶子
                    int v = Nstart + rand() % (Nend - Nstart);
                    if (f->l != NULL) f->r = new node(v, f, 1);
                    else if (f->r != NULL) f->l = new node(v, f, 0);
                    else {
                        if (rand() % 2 == 0) f->l = new node(v, f, 0);
                        else                 f->r = new node(v, f, 1);
                    }
                }
            }
        }
    }

    // 变异
    void mutation(double rate, int Nstart, int Nend, int dep) {
        node* r = get_node_random(root);

        if (r->d == dep) {
            if (r->t == 0) {
                (r->f)->l = new node(rand() % (Nend - Nstart) + Nstart, (r->f), 0);
                r = (r->f)->l;
            }
            else {
                (r->f)->r = new node(rand() % (Nend - Nstart) + Nstart, (r->f), 1);
                r = (r->f)->r;
            }
        }
        else {
            if (r->d == 1) {
                root = new node();
            }
            else {
                if (r->t == 0) {
                    (r->f)->l = new node(rand() % (Nend - Nstart) + Nstart, (r->f), 0);
                    r = (r->f)->l;
                }
                else {
                    (r->f)->r = new node(rand() % (Nend - Nstart) + Nstart, (r->f), 1);
                    r = (r->f)->r;
                }
            }
            construct(rate, Nstart, Nend, dep, root);
        }
    }

    // 更新高度
    int update(node* r, int dep, node* f) {
        if (r == NULL) return 0;
        r->d = dep;
        r->f = f;
        return max(dep, max(update(r->r, dep + 1, r), update(r->l, dep + 1, r)));
    }

    // 中序输出（vector版本）
    void output(node* r, std::vector<int>& a, std::vector<int>& dep, int& cnt) const {

        if (r == NULL) return;
        output(r->l, a, dep, cnt);
        a.push_back(r->a);
        dep.push_back(r->d);
        cnt++;
        output(r->r, a, dep, cnt);
    }

    // 复制树
    node* copy_tree(node* n, node* f) {
        if (n == NULL) return NULL;
        node* tr = new node();
        tr->a = n->a;
        tr->d = n->d;
        tr->l = copy_tree(n->l, tr);
        tr->r = copy_tree(n->r, tr);
        tr->f = f;
        tr->t = n->t;
        return tr;
    }

    // 释放空间
    void deleteT(node* f) {
        if (f == NULL) return;
        deleteT(f->l);
        deleteT(f->r);
        delete(f);
    }
};

typedef struct Node {
    ElemType elem;
    struct Node* lchild;
    struct Node* rchild;
} BTNode;

// ===================== Function declarations (keep original) =====================
bool crossover(tree& t1, tree& t2, int dep);
//int cmp(const tree& a, const tree& b);

void GPHH(Machines* machines, int SIZE, int Itr, double* bestFitness, tree& FinCVr, tree& FinCVs);
void createBT(BTNode*& BT, string str);
void displayBT(BTNode*& BT);
void destroyBT(BTNode*& root);

double decode(Machines* machines, int machineNum_, int* primes, int primeNum_, node* root);
double decode1(Machines* machines, int machineNum_, int* primes, int primeNum_, node* root);

double LowLevel_heuristics(Machines* machines, int machineNum_, int* primes, int primeNum_, int selection);
double LowLevel_heuristics1(Machines* machines, int machineNum_, int* primes, int primeNum_, int selection);

double CCGPHHFitness(Machines* machines, const tree& rout, const tree& seq);
// Batch evaluation APIs (population-wise) to reduce redundant distance computation
extern "C" void CCGPHHFitness_batch_fixed_seq(
    Machines* machines,
    const tree* rout_pop,
    int pop_size,
    const tree* seq_fixed,
    double* out_fitness);

extern "C" void CCGPHHFitness_batch_fixed_rout(
    Machines* machines,
    const tree* seq_pop,
    int pop_size,
    const tree* rout_fixed,
    double* out_fitness);

void showstr(const tree& dispatching, int flag);

//void test(Machines* machines, const tree& rout, const tree& seq);

double compute_total_throughput_v2(const Machines machines[], int machineNum, int pmacs[], int pnum, double Treconfig);
double estimate_Tcon_k(int pmac_id, const Machines machines[]);

void writeData();
void init();

// ================= Dataset-driven GP training/testing =================
enum class DatasetSplit { TRAIN = 0, TEST = 1 };

struct DatasetConfig {
    std::string train_dir;
    std::string test_dir;
    int train_steps = 1000;
    int test_steps = 200;
    int start_index = 1;
};

void set_dataset_config(const DatasetConfig& cfg);
const DatasetConfig& get_dataset_config();

void set_dataset_split(DatasetSplit split);
DatasetSplit get_dataset_split();
