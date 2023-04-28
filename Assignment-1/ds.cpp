#include<bits/stdc++.h>

using std::vector;
using std::pair;

class DisjointSet{
public:
	vector<int> rank,parent;
	int n=0;

	void  newSet(){
		rank.push_back(0);
		parent.push_back(this->n);
		this->n = this->n + 1;
	}

	int find(int x){
		if(parent[x]!=x)
			parent[x] = find(parent[x]);
		return parent[x];
	}

	void Union(int x, int y){
		int setx = find(x);
		int sety = find(y);

		if(setx==sety)
			return;
		int rankx = rank[setx];
		int ranky = rank[sety];
		if(rankx<ranky)
			parent[setx] = sety;
		else if(rankx>ranky)
			parent[sety]=setx;
		else {
			parent[sety] = setx;
			rank[setx]++;
		}
	}
};

bool componentSort(const vector<pair<int,int>> &a, const vector<pair<int,int>> &b){
	return a.size()>b.size();
}

//two pass connected component algorithm (8-connectivity)
//output returns components where components[i] contains (i,j) coordinates belonging to the same component i.
vector<vector<pair<int,int>>> connectedComponent(vector<vector<int>> image){
	int rows = image.size();
	int cols = image[0].size();
	vector<vector<int>> label (rows, vector<int>(cols,0));
	DisjointSet set;
	
	for(int i = 0;i<rows;i++)
		for(int j = 0;j<cols;j++){
			int curr = image[i][j];
			vector<int> res;
			if((j>0) && (curr == image[i][j-1])) res.push_back(label[i][j-1]);
			if((i>0) && (j>0) && (curr == image[i-1][j-1])) res.push_back(label[i-1][j-1]);
			if((i>0) && (curr == image[i-1][j])) res.push_back(label[i-1][j]);
			if((i>0) && (j<cols-1) && (curr == image[i-1][j+1])) res.push_back(label[i-1][j+1]);

			if(res.size()==0){
				label[i][j] = set.n;
				set.newSet();
			}
			else {
				label[i][j] = *min_element(res.begin(),res.end());
				for(int k = 0;k<res.size();++k)
					set.Union(label[i][j], res[k]);
			}
		}
	vector<vector<pair<int,int>>> components;
	std::unordered_map<int,int> umap;
	for(int i = 0;i<rows;++i)
		for(int j = 0;j<cols;++j){
			int x = set.find(label[i][j]);
			if(umap.find(x)==umap.end()){
				umap[x] = umap.size();
				vector<pair<int,int>> temp;
				components.push_back(temp);
			}
			label[i][j] = umap[x];
			components[label[i][j]].push_back({i,j});
		}

	sort(components.begin(), components.end(), componentSort);
	return components;
}