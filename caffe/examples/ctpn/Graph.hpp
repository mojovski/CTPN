#ifndef GRAPH_H
#define GRAPH_H

#include <opencv2/opencv.hpp>


class Graph
{
public:
	typedef bool** BoolMat;
	Graph(BoolMat data, int h, int w){
		this->data=data;
		this->width=w;
		this->height=h;
	}

	std::vector<std::vector<int> > sub_graphs_connected(){
		std::vector<std::vector<int> > sub_graphs;
		for (int index=0; index<this->height; index++)
		{
			if (!(hasTrueValues(index, 0)) && hasTrueValues(index, 1))
			{
				int v=index;
				sub_graphs.push_back(std::vector<int>(1,v));
				while (hasTrueValues(v,1))
				{
					//v=np.where(self.graph[v, :])[0][0]
					//selects the first element from the row v, which is true
					v=getFirstTrueIndex(v,1);
					//sub_graphs[-1].append(v)
					(*(sub_graphs.end()-1)).push_back(v);
				}
			}
			
		}
		return sub_graphs;

		/*
		for index in xrange(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v=index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v=np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs
        */
		
	}

protected:
	int width, height;
	BoolMat data;

	/**
	searches the graph for the first true values along a row or column (if dim=0, then row, if dim=01, then column)
	In python this reimplements np.where(graph[:,v])[0][0]
	so call getFirstTrueIndex(v,0)
	**/
	int getFirstTrueIndex(int index, int dim)
	{
		if (dim==0)
		{
			for (int i=0; i<this->height; i++)
			{
				if (this->data[i][index])
					return i;
			}
		}
		if (dim==1)
		{
			for (int i=0; i<this->width; i++)
			{
				if (this->data[index][i])
					return i;
			}

		}

		std::cerr << "getFirstTrueIndex returns -1. No True values in data found with index=" << index << " and dim=" << dim << std::endl;
		return 0;

	}

	/**
	checks if data has true values in the row or column (if dim=0, then row, if dim=01, then column)
	In python this corresponds to data[:, index].any()
	so call hasTrueValues(index, 0)
	**/
	bool hasTrueValues(int index, int dim)
	{
		bool res=false;
		if (dim==0)
		{
			for (int i=0; i<this->height; i++)
			{
				res=this->data[i][index] || res;
			}
		}

		if (dim==1)
		{
			for (int i=0; i<this->width; i++)
			{
				res=this->data[index][i] || res;
			}
		}
		return res;
	}
};


#endif