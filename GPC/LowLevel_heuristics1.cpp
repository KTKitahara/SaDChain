#include "comfunc.h"
using namespace std;
/*--------------------------ֹ---------------------------*/			  
double LowLevel_heuristics1(Machines* machines, int machineNum_, int* primes, int primeNum_, int selection) {//primeNumʾnow_p,primesΪpmacs
	int type=-1;
	double res=0; 
	type=selection;
	bool flagll = 0;
	switch(type){ 
	case 0://节点i与当前轮询主节点距离
		{
		for (int j = 0;j < machineNum;j++) {
			if (primes[primeNum_] == machines[machineNum_].dis[j].id) {
				res = machines[machineNum_].dis[j].distance;
					break;
			}
			}
		flagll = 1;
		break;
		}
	case 1://节点i与当前轮询主节点通信能力
		{
		double distance = calcDistance(machines[machineNum_], machines[primes[primeNum_]]);
		double Pt2 = machines[machineNum_].pt;
		if (abs(Pt2 - 0) < 0.00001) Pt2 = Pt;
		double Pr = Pt2 * Gt * Gr * pow(lambda / (4 * M_PI * distance), 2);
		double R = B * log2(1.0 + Pr / N);
		res = R;
		flagll = 1;
			break;	
		}
	case 2://节点i计算能力
		{
		res = machines[machineNum_].cal;
		flagll = 1;
			break;
		}
	case 3://节点i至离自身最近的非轮询主节点距离
		{
		int j = 0;
	
		for (j = 0;j < machineNum;j++) {

			if ((machines[machineNum_].dis[j].id != primes[primeNum_]) && (machines[machines[machineNum_].dis[j].id].fprime == 1)) {
				res = machines[machineNum_].dis[j].distance;
				flagll = 1;
				break;
			}

			/*if (j == 9 && flagll == 0) {
				j = 0;
				
			}*/

		}
		if (!flagll) {
			//printf("warning");
			res = 1000000;
			flagll = 1;
		}
			break;
		}
	case 4://常数
	{

		res = 2; flagll = 1;

		break;
	}
	case 5://节点i风险概率
	{
		res = machines[machineNum_].atkprob;
		flagll = 1;
		break;
	}
	case 6: // Risk_{shard_now}^{sum} 当前轮询分片风险概率和（含候选节点）
	{
		// 当前 shard = 当前轮询主节点 primes[primeNum_] + 已分配副本 + 当前候选节点 machineNum_
		res = 0.0;
		int primary_idx = primes[primeNum_];   // 当前 shard 的主节点（在 machines[] 中的下标）

		// 主节点风险
		res += machines[primary_idx].atkprob;

		// 已分配副本风险（RNAR 过程中 owned/team 会逐步增长）
		for (int t = 0; t < machines[primary_idx].owned; ++t) {
			int ridx = machines[primary_idx].team[t];
			if (ridx >= 0) res += machines[ridx].atkprob;
		}

		// 候选节点风险：必须加，否则对同一 shard 的所有候选会变成常数
		res += machines[machineNum_].atkprob;

		flagll = 1;
		break;
	}


	case 7:// 
	{
		res = 2;
		flagll = 1;
		break;
	}
	case 8://  
	{
		res = 2;
		flagll = 1;
		break;
	}
	case 9:// 
	{
		res = 2;
		flagll = 1;
		break;
	}
	
	}
	if (!flagll) {
		printf("LL1,%d", type);
		exit(1);
	}
	return res;
}