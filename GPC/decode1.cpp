#include "comfunc.h"
using namespace std;
/*--------------------------个体的解码---------------------------*/

//int machineNum, job,op,timeBase,selection;
/*unsigned long int num[10]={0};
void init(){
	for (int i=0;i<10;i++){
		num[i] = 0;
	}
}*/

/*

void createBT(BTNode* &BT, string str) {
    stack<BTNode*> sta;
    BTNode *p;
    for(int i = 0; i < str.size(); i++) {
        p = new BTNode;
        p->lchild = NULL;
        p->rchild = NULL;
        char ch = str[i];
        if(ch == '*' || ch == '/' || ch == 'm' || ch == 'n') {
            p->elem = ch;
            BTNode *r = new BTNode;
            i++;
            r->elem = str[i];
            r->lchild = NULL;
            r->rchild = NULL;
            p->rchild = r;
            p->lchild = sta.top();
            sta.pop();
            sta.push(p);
        } else if(ch == '+' || ch == '-') {
            p->elem = ch;
            if(sta.size() == 2) {
                BTNode *r = sta.top();
                sta.pop();
                sta.top()->rchild = r;
            }
            p->lchild = sta.top();
            sta.pop();
            sta.push(p);
        } else {
            p->elem = ch;
            sta.push(p);
        }
        p = NULL;
        free(p);
    }
	
    if (sta.size() == 2) {
        BTNode *r = sta.top();
        sta.pop();
        sta.top()->rchild = r;
    }
    BT = sta.top();
}



void displayBT(BTNode* &BT) {                                                                      
    if(BT != NULL){
		cout << BT->elem;
		displayBT(BT->lchild);
		displayBT(BT->rchild);
	}
	else{
		printf("#");
	}
}

void destroyBT(BTNode* &root) {
    if(root != NULL) {
        destroyBT(root->lchild);
        destroyBT(root->rchild);
        free(root);
    }
}
 */

double decode1(Machines *machines,int machineNum_, int* primes, int primeNum_, node* root){
	
	int ch = root->a;
	if(ch == 10) return decode1(machines,machineNum_,primes,primeNum_,root->l) + decode1(machines,machineNum_,primes, primeNum_,root->r);
	else if(ch == 11) return decode1(machines,machineNum_,primes,primeNum_,root->l) - decode1(machines,machineNum_,primes, primeNum_,root->r);
	else if(ch == 12) {
        double tmpl = decode1(machines, machineNum_, primes, primeNum_, root->l);
        double tmpr = decode1(machines, machineNum_, primes, primeNum_, root->r);
        if (!tmpl)tmpl = 1;
        if (!tmpr)tmpr = 1;
        return tmpl * tmpr;
    }
	else if(ch == 13) return decode1(machines,machineNum_,primes, primeNum_,root->r) == 0 ? 1 : (decode1(machines,machineNum_,primes,primeNum_,root->l) / decode1(machines,machineNum_,primes, primeNum_,root->r));
	else if (ch == 14) return max(decode1(machines,machineNum_,primes,primeNum_,root->l), decode1(machines,machineNum_,primes, primeNum_,root->r));//max 
	else if (ch == 15) return min(decode1(machines,machineNum_,primes,primeNum_,root->l), decode1(machines,machineNum_,primes, primeNum_,root->r));//min
    else if (ch == 16) {
        // protected log: log(|x| + 1)
        double x = decode1(machines, machineNum_, primes, primeNum_, root->l);
        if (!std::isfinite(x)) x = 0.0;
        return std::log(std::fabs(x) + 2.0);
    }
    else {
	//	num[ch]++;
		
		return LowLevel_heuristics1(machines,machineNum_,primes, primeNum_,ch);}
	//else return ch;x

}
/*void writeData(){
	ofstream f("effective.txt",ios::app);
	for (int i=0;i<10;i++){
		f<<num[i]<<" ";
	}
	f<<endl;
}*/