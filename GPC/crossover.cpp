#include "comfunc.h"
using namespace std;
//交叉 指定深度
bool crossover(tree& t1,tree& t2,int dep)
{
	
    node *n1=t1.get_node_random(t1.root);//随机获得t1的一个结点
    node *n2=t2.get_node_random(t2.root);//随机获得t2的一个结点
	
    if(n1->d==1||n2->d==1)//若两个结点的高度都为1 无法进行交叉
        return false;
	
    if(n1->t==0)//若 n1为左侧树，换掉n1，则需要将n1的父亲结点的左孩子指向n2
        n1->f->l=n2;
    else//若 n1为右侧树，换掉n1，则需要将n1的父亲结点的右孩子指向n2
        n1->f->r=n2;
	
    if(n2->t==0)//同理 n2
        n2->f->l=n1;
    else
         n2->f->r=n1;
	
    swap(n1->t,n2->t);//交换n1，n2的t值
	//计算交叉后的高度
    int d1=t1.update(t1.root,1,NULL);
    int d2=t2.update(t2.root,1,NULL);
	
    if(d1<=dep&&d2<=dep)//若高度合理，返回正确
        return true;
	//若高度不合理，则再换回来，并返回错误
    if(n1->t==0)
        n1->f->l=n2;
    else
        n1->f->r=n2;
	
    if(n2->t==0)
        n2->f->l=n1;
    else
        n2->f->r=n1;
	
    swap(n1->t,n2->t);
	
    d1=t1.update(t1.root,1,NULL);
    d2=t2.update(t2.root,1,NULL);
	
	
    return false;
}