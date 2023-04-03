#include "network.cpp"
#include <iostream>

using namespace std;

std::vector<DataBlock> DATA = {
	{{1,1},{0,0}},
	
	{{3,2},{1,0}},
	{{4,2},{1,0}},
	{{3,2},{1,0}},
	{{3,1},{1,0}},
	{{3,3},{1,0}},
	{{4,4},{1,0}},
	{{4,5},{1,0}},

	{{6,7},{0,1}},
	{{7,8},{0,1}},
	{{5,6},{0,1}},
	{{5,7},{0,1}},
	{{5,3},{0,1}},
	{{6,6},{0,1}},
	{{3,6},{0,1}},
	{{7,7},{0,1}},
	{{8,8},{0,1}},
};

int main(){
	Network net({2,3,2});
	while(1){
		if(((1.0 -  net.loss(DATA)) * 100)  >= 99.5) break;
		cout << "accuracy : "<<(1.0 -  net.loss(DATA)) * 100 << "%"<< '\n';
		net.Learn(DATA, 0.2);
	}
	net.printLayers();
	double a,b;
	while(1){
		cin >> a >> b;
		if(net.classify({a,b}) == 0) cout << "MOZNA\n";
		else cout << "NIE MOZNA\n";
	}
}
