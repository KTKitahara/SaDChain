#include "comfunc.h"
using namespace std;
/*--------------------------ֹ---------------------------*/			  
double LowLevel_heuristics(Machines *machines,int machineNum_, int* primes, int primeNum_,int selection){//machineNumʾ± selectionʾֵ
	int type=-1;
	double res=0; 
	bool flagll = 0;
	type=selection;
	switch(type){ 
	case 0://节点i计算能力
		{
		res = machines[machineNum_].cal;
		flagll = 1;
			break;
		}
	case 1://节点i至最近m个节点的通信能力和
		{
		double Rsum = 0.0;
		double Pt2 = machines[machineNum_].pt;
		if (abs(Pt2 - 0) < 0.00001) Pt2 = Pt;
		for (int j = 1; j < mnode+1; ++j) {
			//if (j >= machineNum_) break; 
			
			double distance = machines[machineNum_].dis[j].distance;

			double Pr = Pt2 * Gt * Gr * pow(lambda / (4 * M_PI * distance), 2);
			

			double R = B * log2(1.0 + Pr / N);

			Rsum += R;
		}

			res = Rsum;
			flagll = 1;
			break;	
		}
	case 2://节点i与最近m个节点的距离和
		{
		double ndis = 0;
		//int n0 = machineNum * (1 - threshold);
		for (int j = 1;j < mnode+1;j++) {
			ndis += machines[machineNum_].dis[j].distance;
		}
		res = ndis;
		flagll = 1;
			break;
		}
	case 3://节点i至全部已有主节点距离和
		{
		double npdis = 0;
		//int n0 = machineNum * (1 - threshold);
		for (int j = 0;j < machineNum;j++) {
			if (machines[machines[machineNum_].dis[j].id].fprime == 1) {
				npdis += machines[machineNum_].dis[j].distance;
			}
			
		}
		res = npdis;
		flagll = 1;
			break;
		}
	case 4://常数
		{
		res = 2;
		flagll = 1;
			break;
		}
 case 5://节点i风险概率
		{
			res = machines[machineNum_].atkprob;
			flagll = 1;
			break;
		}
	case 6://  
		{
			
			res = 2;
			break;
		}
	case 7:// 
		{
		res = 0;
			break;
		}
	case 8://  
		{
			res = 2;
			break;
		}
	case 9:// 
		{
			res = 2;
			break;
		}
	}

	if (!flagll) {
		printf("LL,%d", type);
		exit(1);
	}
	return res;
}