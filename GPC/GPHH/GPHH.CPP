#include "comfunc.h"
using namespace std;
/*--------------------------种群的进化---------------------------*/

void GPHH(Machines *machines,Jobs *jobs,int SIZE,int Itr,int **b,int **c,int **d,int *arr,int *cnt,int **bt,int **ct,int **dt,int *arrt,int *cntt,double *bestFitness)//SIZE种群大小 Itr 迭代次数
{   
	int i;
	int N=10,dep1=6,dep2=6;//dep1 2 二叉树深度
	double rate=0.5,prom=0.15,totaltime_one=0;// prom 变异率 totaltime_one 记录时间
	double rnd,rnd1,rnd2;//随机数

	//int LLNum=50,totalNum,ccmakespan;//叶子结点个数，叶子结点个数，最大加工时间
	//totalNum=LLNum;//叶子结点个数
	float *EP1=new float[SIZE];//轮盘赌
	float *EP2=new float[SIZE];//轮盘赌
	tree *routing=new tree[SIZE];//routing rule初始化
	tree *sequencing=new tree[SIZE];//sequencing rule初始化
	tree *tmpRouting=new tree[SIZE];//临时routing rule
	tree *tmpSequencing=new tree[SIZE];//临时sequencing rule
	float EpSum1,EpSum2;//轮盘赌数据记录 
	
	//生成初始种群 
	int elite = 4;//精英个体
	tree CVr,CVs;//最终解向量
	
	for (i=0;i<SIZE;i++){
		routing[i].construct(rate,N,dep1,routing[i].root);//初始化routing rule
		sequencing[i].construct(rate,N,dep2,sequencing[i].root);//初始化sequencing rule
	}
//	system("pause");
	CVr = routing[0];
	CVs = sequencing[0];
	init();
//	int b[100];
//	int cnt=0;
//	CVr.output(CVr.root,b,&cnt);
//	CVr.fitness = CCGPHHFitness(CVr,CVs);//计算CVrfitness
//	CVs.fitness = CCGPHHFitness(CVr,CVs);//计算CVsfitness
//	cout<<"CVr.fitness "<<CVr.fitness <<" ";
//	cout<<"CVs.fitness "<<CVs.fitness <<endl;
//	ofstream f("clock1.txt",ios::app);
	clock_t start_one,finish_one,start_one1,finish_one1,start_fitness,finish_fitness,start_iter,end_iter;//开始、结束时间 
	start_fitness = clock();
	CCGPHHFitness(machines,jobs,routing[0],CVs,b,c,d,arr,cnt);
	finish_fitness = clock();
//	f<<"计算一个fitness时间"<<(finish_fitness-start_fitness)*1.0/CLK_TCK<<" "<<finish_fitness-start_fitness<<endl;
	for (i=0;i<SIZE;i++){
		routing[i].fitness = CCGPHHFitness(machines,jobs,routing[i],CVs,b,c,d,arr,cnt);//计算所有routing rule的fitness
		sequencing[i].fitness = CCGPHHFitness(machines,jobs,CVr,sequencing[i],b,c,d,arr,cnt);//计算所有sequencing rule的fitness
		//cout<<"routing[i].fitness:"<<routing[i].fitness<<"  "<<"sequencing[i].fitness:"<<sequencing[i].fitness<<endl;
	}
//	cout<<"fitness时间为:"<<finish_fitness-start_fitness<<endl;
	ofstream ftg("result_fitness.txt",ios::app);
	start_iter = clock();
	
	for (int t=0;t<Itr;t++){//开始迭代
		if(t>0){
			randop(b,c,d,cnt);
		}
		double avgCVr=0,avgCVs=0;
		cout<<"*****************begin Itr"<<t<<"************"<<endl;
		srand(time(NULL));		
		sort(routing,routing+SIZE,cmp);//升序排序
		sort(sequencing,sequencing+SIZE,cmp);//升序排序
	
		CVr = routing[0];
		CVs = sequencing[0];
//		cout<<"CVr:"<<CVr<<"  "<<"CVs:"<<CVs<<endl;
		
		if(t==(Itr-1)){
			showstr(CVr);
			showstr(CVs);
		}
				
		if(t==Itr){
			showstr(CVr);
			showstr(CVs);
		}
		ftg<<CVr.fitness<<endl;
		ftg<<CVs.fitness<<endl;
		//cout<<"CVr.fitness "<<CVr.fitness <<" ";
		cout<<"CVr.fitness "<<CVr.fitness <<" ";
		cout<<"CVs.fitness "<<CVs.fitness <<endl;
//		showstr(CVr);
//		showstr(CVr);
//		cout<<"*****************begin 计算种群的适应度值************"<<endl;
//		cout<<"*****************begin routing.fitness************"<<endl;
		for (i=0;i<SIZE;i++){
			avgCVr+=routing[i].fitness;
//			cout<<"routing["<<i<<"].fitness:"<<routing[i].fitness<<endl;
		}
//		cout<<"*****************end routing.fitness************"<<endl;
//		cout<<"*****************begin sequencing.fitness************"<<endl;
		for (i=0;i<SIZE;i++){
			avgCVs +=sequencing[i].fitness;
//			cout<<"sequencing["<<i<<"].fitness:"<<sequencing[i].fitness<<endl;
		}
		avgCVr = avgCVr/SIZE;
		avgCVs = avgCVs/SIZE;
//		cout<<"*****************end sequencing.fitness************"<<endl;
		cout<<avgCVr<<" "<<avgCVs<<endl;
//		cout<<"*****************end 计算种群的适应度值************"<<endl;
		for (i=0;i<SIZE;i++){//将rule 赋值给 临时rule
			tmpRouting[i] = routing[i];
			tmpSequencing[i] = sequencing[i];
		}
//		cout<<"*****************begin 轮盘赌************"<<endl;
		//从精英个体中随机选取两个个体产生下一代
		start_one=clock();
		tree tmp[2];
		//轮盘赌 
		EpSum1=0;
		for (i=0;i<elite;i++)
		{//适应度总和 
			EpSum1=EpSum1+1/(routing[i].fitness);
		}
//		cout<<"EpSum1:"<<EpSum1<<endl;
		for (i=0;i<elite;i++)
		{//每个个体适应度占比 
			EP1[i]=1/(routing[i].fitness)/EpSum1;
//			cout<<"EP1:"<<i<<" "<< EP1[i]<<endl;
		}
		for (i=1;i<elite;i++)
		{//适应度累计值 
			EP1[i]=EP1[i]+EP1[i-1];
			cout<<"EP1:"<<i<<" "<< EP1[i]<<endl;
		}
		EpSum2=0;
		for (i=0;i<elite;i++)
		{//适应度总和 
			EpSum2=EpSum2+1/(sequencing[i].fitness);
		}
		for (i=0;i<elite;i++)
		{//每个个体适应度占比 
			EP2[i]=1/(sequencing[i].fitness)/EpSum2;
		}
		for (i=1;i<elite;i++)
		{//适应度累计值 
			EP2[i]=EP2[i]+EP2[i-1];
		}
		finish_one=clock();
//		f<<"精英选择时间"<<(finish_one-start_one)*1.0/CLK_TCK<<" "<<finish_one-start_one<<endl;
//		cout<<"*****************end 轮盘赌************"<<endl;
//		cout<<"*****************begin routing rules 交叉************"<<endl;
		//对 routing rules 进行交叉、变异、繁殖 
		start_one=clock();
		for (i=0;i<(SIZE-elite)/2;i++){
			for (int j=0;j<2;j++){
				rnd1 = 1.0*rand()/RAND_MAX;
				for (int k=0;k<elite;k++)
				{   			
//					cout<<i<<" "<<j<<" "<<k<<" "<<"rnd1"<<" "<<rnd1<<endl;
					if (rnd1<EP1[k])
					{
						tmp[j] = routing[k];
						break;
					}
				}
			}
//			cout<<"crossover"<<endl;
//			cout<<"交叉前：";
//			showstr(tmp[0]);
//			showstr(tmp[1]);
			crossover(tmp[0],tmp[1],dep1);//tmp0 1 交叉
//			cout<<"交叉后：";
//			showstr(tmp[0]);
//			showstr(tmp[1]);
//			cout<<"crossover"<<elite+i*2<<" "<<elite+i*2+1<<endl;
			tmp[0].fitness = CCGPHHFitness(machines,jobs,tmp[0],CVs,b,c,d,arr,cnt);//计算tmp0的fitness
			if(tmpRouting[elite+i*2].fitness>tmp[0].fitness)//fitness 越小越好
				tmpRouting[elite+i*2] = tmp[0];
			tmp[1].fitness = CCGPHHFitness(machines,jobs,tmp[1],CVs,b,c,d,arr,cnt);//计算tmp1的fitness
			if(tmpRouting[elite+i*2+1].fitness>tmp[1].fitness)//fitness 越小越好
				tmpRouting[elite+i*2+1] = tmp[1];
		}
		finish_one=clock();
		cout<<"交叉时间:"<<(finish_one-start_one)*1.0<<endl;
//		f<<"routing rule 交叉时间"<<(finish_one-start_one)*1.0/CLK_TCK<<" "<<finish_one-start_one<<endl;
//		cout<<"*****************end routing rules 交叉************"<<endl;
//		cout<<"*****************begin routing rules 变异************"<<endl;
		start_one1=clock();
		for (i=0;i<SIZE;i++)
		{   
			rnd = 1.0*rand()/RAND_MAX;
			if (rnd<prom)//当概率小于prom时二叉树进行变异
			{
				tree befortmp;
				befortmp = tmpRouting[i];
				befortmp.mutation(rate,N,dep1);//进行变异
				befortmp.fitness = CCGPHHFitness(machines,jobs,befortmp,CVs,b,c,d,arr,cnt);//计算fitness
				if(tmpRouting[i].fitness>befortmp.fitness){//fitness 越小越好
					tmpRouting[i] = befortmp;
				}
			}
		}
		finish_one1=clock();
		cout<<"变异时间:"<<(finish_one1-start_one1)*1.0<<endl;
//		f<<"routing rule 变异时间"<<(finish_one1-start_one1)*1.0/CLK_TCK<<" "<<finish_one1-start_one1<<endl;
//		cout<<"*****************end routing rules 变异************"<<endl;


		for (i=0;i<SIZE;i++){
			routing[i] = tmpRouting[i];//临时赋值给原来的
		}
        sort(routing,routing+SIZE,cmp);//升序排序		
		CVr = routing[0];


		//对 sequencing rules 进行交叉、变异、繁殖
//		cout<<"*****************begin sequencing rules 交叉************"<<endl;
		start_one1=clock();
		for (i=0;i<(SIZE-elite)/2;i++){
			for (int j=0;j<2;j++){
				for (int k=0;k<elite;k++)
				{   
					//RAND_MAX = 0x7fff (32767)D
					rnd2 = 1.0*rand()/RAND_MAX;
					if (rnd2<EP2[k])
					{
						tmp[j] = sequencing[k];
						break;
						
					}
				}
			}
			crossover(tmp[0],tmp[1],dep2);
			tmp[0].fitness = CCGPHHFitness(machines,jobs,CVr,tmp[0],b,c,d,arr,cnt);
			if(tmpSequencing[elite+i*2].fitness>tmp[0].fitness)	
				tmpSequencing[elite+i*2] = tmp[0];
			tmp[1].fitness = CCGPHHFitness(machines,jobs,CVr,tmp[1],b,c,d,arr,cnt);
			if(tmpSequencing[elite+i*2+1].fitness>tmp[1].fitness)
				tmpSequencing[elite+i*2+1] = tmp[1];
		}
		finish_one1=clock();
		cout<<"交叉时间:"<<(finish_one1-start_one1)*1.0<<endl;
//		f<<"sequencing rule 交叉时间"<<(finish_one1-start_one1)*1.0/CLK_TCK<<" "<<finish_one1-start_one1<<endl;
//		cout<<"*****************end sequencing rules 交叉************"<<endl;
//		cout<<"*****************begin sequencing rules 变异************"<<endl;
		start_one1=clock();
		for (i=0;i<SIZE;i++)
		{   
			rnd=1.0*rand()/RAND_MAX;
			if (rnd<prom)//当概率小于prom时二叉树进行变异
			{
				tree befortmp;
				befortmp = tmpSequencing[i];
				befortmp.mutation(rate,N,dep1);
				befortmp.fitness = CCGPHHFitness(machines,jobs,CVr,befortmp,b,c,d,arr,cnt);
				if(tmpSequencing[i].fitness>befortmp.fitness){
					tmpSequencing[i] = befortmp;
				}
			}
		}
		finish_one1=clock();
		cout<<"变异时间:"<<(finish_one1-start_one1)*1.0<<endl;
//		f<<"sequencing rule 变异时间"<<(finish_one1-start_one1)*1.0/CLK_TCK<<" "<<finish_one1-start_one1<<endl;
//		cout<<"*****************end sequencing rules 变异************"<<endl;

		for (i=0;i<SIZE;i++){
			sequencing[i] = tmpSequencing[i];
		}
		sort(sequencing,sequencing+SIZE,cmp);//升序排序
		CVs = sequencing[0];

//		cout<<"*****************begin routing rules 交叉************"<<endl;
		//对 routing rules 进行交叉、变异、繁殖 
		start_one=clock();
		for (i=0;i<(SIZE-elite)/2;i++){
			for (int j=0;j<2;j++){
				rnd1 = 1.0*rand()/RAND_MAX;
				for (int k=0;k<elite;k++)
				{   			
//					cout<<i<<" "<<j<<" "<<k<<" "<<"rnd1"<<" "<<rnd1<<endl;
					if (rnd1<EP1[k])
					{
						tmp[j] = routing[k];
						break;
					}
				}
			}
//			cout<<"crossover"<<endl;
//			cout<<"交叉前：";
//			showstr(tmp[0]);
//			showstr(tmp[1]);
			crossover(tmp[0],tmp[1],dep1);//tmp0 1 交叉
//			cout<<"交叉后：";
//			showstr(tmp[0]);
//			showstr(tmp[1]);
//			cout<<"crossover"<<elite+i*2<<" "<<elite+i*2+1<<endl;
			tmp[0].fitness = CCGPHHFitness(machines,jobs,tmp[0],CVs,b,c,d,arr,cnt);//计算tmp0的fitness
			if(tmpRouting[elite+i*2].fitness>tmp[0].fitness)//fitness 越小越好
				tmpRouting[elite+i*2] = tmp[0];
			tmp[1].fitness = CCGPHHFitness(machines,jobs,tmp[1],CVs,b,c,d,arr,cnt);//计算tmp1的fitness
			if(tmpRouting[elite+i*2+1].fitness>tmp[1].fitness)//fitness 越小越好
				tmpRouting[elite+i*2+1] = tmp[1];
		}
		finish_one=clock();
		cout<<"交叉时间:"<<(finish_one-start_one)*1.0<<endl;
//		f<<"routing rule 交叉时间"<<(finish_one-start_one)*1.0/CLK_TCK<<" "<<finish_one-start_one<<endl;
//		cout<<"*****************end routing rules 交叉************"<<endl;
//		cout<<"*****************begin routing rules 变异************"<<endl;
		start_one1=clock();
		for (i=0;i<SIZE;i++)
		{   
			rnd = 1.0*rand()/RAND_MAX;
			if (rnd<prom)//当概率小于prom时二叉树进行变异
			{
				tree befortmp;
				befortmp = tmpRouting[i];
				befortmp.mutation(rate,N,dep1);//进行变异
				befortmp.fitness = CCGPHHFitness(machines,jobs,befortmp,CVs,b,c,d,arr,cnt);//计算fitness
				if(tmpRouting[i].fitness>befortmp.fitness){//fitness 越小越好
					tmpRouting[i] = befortmp;
				}
			}
		}
		finish_one1=clock();
		cout<<"变异时间:"<<(finish_one1-start_one1)*1.0<<endl;
//		f<<"routing rule 变异时间"<<(finish_one1-start_one1)*1.0/CLK_TCK<<" "<<finish_one1-start_one1<<endl;
//		cout<<"*****************end routing rules 变异************"<<endl;
		for (i=0;i<SIZE;i++){
			routing[i] = tmpRouting[i];//临时赋值给原来的
			sequencing[i] = tmpSequencing[i];
		}
	/*	int index1 = 0,index2 = 0;//找到最小的fitness 
		for (i=1;i<SIZE;i++){
			if (routing[i].fitness<routing[index1].fitness)
				index1 = i;
			if (sequencing[i].fitness<sequencing[index2].fitness)
				index2 = i;
		}
		
		if (routing[index1].fitness<CVr.fitness)//进行替换
			CVr = routing[index1];
		if (sequencing[index2].fitness<CVr.fitness)
			CVs = sequencing[index2];*/
	
		cout<<"*****************end Itr"<<t<<"************"<<endl;
	}
	sort(routing,routing+SIZE,cmp);
	sort(sequencing,sequencing+SIZE,cmp);
//	 
//	showstr(routing[0]);
//	showstr(sequencing[0]);
//	writeData();
//	init();

//	writeData();
	CVr = routing[0];
	CVs = sequencing[0];//最优解 
//	f<<CVr.fitness<<endl;
//	f<<CVs.fitness<<endl;
	if(CVr.fitness>CVs.fitness){
		bestFitness[0] = CVs.fitness;
		
	}
	else{
		bestFitness[0] = CVr.fitness;
		
	}

		
	end_iter = clock();
//	f<<"计算iter的完整时间"<<(end_iter-start_iter)*1.0/CLK_TCK<<" "<<end_iter-start_iter<<endl;
	start_one1=clock();
//	test(machines,jobs,CVr,CVs,bt,ct,dt,arrt,cntt);//测试最优解 
//	test(machines,jobs,CVr,CVs,bt,ct,dt,arrt,cntt);//测试最优解
	test(machines,jobs,CVr,CVs,bt,ct,dt,arrt,cntt);//测试最优解 

	ftg<<"*********************************************"<<endl;
	finish_one1=clock();
//	ftg<<"********************************************************************"<<endl;
//	f<<"test 时间"<<(finish_one1-start_one1)*1.0/CLK_TCK<<" "<<finish_one1-start_one1<<endl;
	printf("----------------\n");

	finish_one=clock();//计算时间
	totaltime_one=(double)(finish_one-start_one);
	delete[] EP1;
	delete[] EP2;
	delete[] routing;
	delete[] sequencing;
	delete[] tmpRouting;
	delete[] tmpSequencing;
	
}