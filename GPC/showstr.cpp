#include "comfunc.h"
using namespace std;
unsigned long int num[10] = { 0 };
void init() {
    for (int i = 0; i < 10; i++) {
        num[i] = 0;
    }
}
void showstr(const tree& dispatching, int flag) {

    vector<int> a, dep;
    int cnt = 0;
    string str = "";
    /*ofstream fstr("str_tree.txt", ios::app);
    ofstream f("size.txt", ios::app);*/
    ofstream fstr(result_path("str_tree.txt"), ios::app);
    ofstream f(result_path("size.txt"), ios::app);

    dispatching.output(dispatching.root, a, dep, cnt); // ÐÞ¸Äoutput²ÎÊý

    int depnow = flag ? dep2:dep1;

    for (int i = 0; i < cnt; i++) {
        if (a[i] == 10)
            str += '+';
        else if (a[i] == 11)
            str += '-';
        else if (a[i] == 12)
            str += '*';
        else if (a[i] == 13)
            str += '/';
        else if (a[i] == 14)
            str += 'm';
        else if (a[i] == 15)
            str += 'n';
        else if (a[i] == 16)
            str += 'l'; // log
        else {
            num[a[i]]++;
            str += a[i] + '0';
        }
        cout << dep[i];
        fstr << dep[i];
    }
    cout << endl;
    fstr << endl;
    f << str.size() << endl;
    for (int h = 1; h <= depnow; h++) {
        for (int j = 0; j < cnt; j++) {
            if (dep[j] == h) {
                cout << str[j] << " ";
                fstr << str[j] << " ";
            }
        }
        cout << endl;
        fstr << endl;
    }
    fstr << "*************************************" << endl;
}
void writeData() {
    //ofstream f("effective.txt", ios::app);
    ofstream f(result_path("effective.txt"), ios::app);
    for (int i = 0; i < 10; i++) {
        f << num[i] << " ";
    }
    f << endl;
}
