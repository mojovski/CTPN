#ifndef CONNECTOR_H
#define CONNECTOR_H

#include <opencv2/opencv.hpp>


class Connector
{
public:
	typedef bool** BoolMat;

	class Options
	{
	public:
		int MAX_HORIZONTAL_GAP;
		float MIN_V_OVERLAPS;
		float MIN_SIZE_SIM;
		Options():MAX_HORIZONTAL_GAP(50),
		MIN_V_OVERLAPS(0.7),
		MIN_SIZE_SIM(0.7)();

	};

	std::vector<cv::Rect> text_proposals;
	std::vector<float> scores, heights;
	cv::Size im_size;
	std::vector<std::vector<int> > boxes_table;
	Options options;

	Connector(){}

	void build_graph(std::vector<cv::Rect>& text_proposals, std::vector<float>& scores, cv::Size& im_size)
	{
		this->text_proposals=text_proposals
        this->scores=scores;
        this->im_size=im_size
        this->heights=extract_heights(text_proposals);

        boxes_table=std::vector<std::vector<int> >(im_size.width);
        int idx=0;
        for (std::vector<cv::Rect>::iterator it=text_proposals.begin(); it!=text_proposals.end(); it++)
        {
        	boxes_table[(int)(*it).x].push_back(idx);
        	idx++;
        }
        //create a two dim array on heap
        BoolMat graph = new bool*[text_proposals.size()];
		for(int i = 0; i < text_proposals.size(); ++i)
		{
		    graph[i] = new bool[text_proposals.size()];
		    std::fill_n(graph[i], text_proposals.size(), false); 
		}

		int idx=0;
		for (std::vector<cv::Rect>::iterator text_proposals.begin(); it!=text_proposals.end(); it++)
		{
			std::vector<int> successions=self.get_successions(index)
			idx++;
			if (successions.size()==0)
				continue;
			int succession_index=arg_max<float>(scores, successions);
		}

		#python
        for index, box in enumerate(text_proposals):
            successions=self.get_successions(index)
            if len(successions)==0:
                continue
            succession_index=successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index]=True
        return Graph(graph)
	}

protected:

	bool is_succession_node(int idx, int succession_index)
	{
		#TODO
		precursors=self.get_precursors(succession_index)
        if self.scores[index]>=np.max(self.scores[precursors]):
            return True
        return False
	}

	template<typename T>
	int arg_max(std::vector<T>& arr, std::vector<int>& indices)
	{
		int idx_best=-1;
		T max_val=0;
		for (std::vector<int>::iterator iit=indices.begin(); iit!=indices.end(); iit++)
		{
			if (max_val < arr[*iit])
			{
				max_val=arr[*iit];
				idx_best=*iit;
			}
		}
		return idx_best;
	}


	std::vector<float> extract_heights(std::vector<cv::Rect>& text_proposals)
	{
		std::vector<float> heights;
		for(std::vector<cv::Rect>::iterator it=text_proposals.begin(); it!=text_proposals.end(); it++)
		{
			heights.push_back(it->height+1.0)
		}
		return heights;
	}

	std::vector<int> get_precursors(int index)
	{
		TODO
	}

	std::vector<int> get_successions(int index)
	{
		cv::Rect box=this->text_proposals[index];
	    std::vector<int> results;

	    for left in range((int)box.x+1, min(int(box[0])+cfg.MAX_HORIZONTAL_GAP+1, self.im_size[1])):
	    int min_pixel=std::min((int)box.x+options.MAX_HORIZONTAL_GAP+1, this->im_size.width);
		for (int left=(int)box.x+1; left<min_pixel; left++)
		{
			std::vector<int> adj_box_indices=this->boxes_table[left];
			for(std::vector<int>::iterator it=adj_box_indices.begin(); it!=adj_box_indices.end(); it++)
			{
				if self.meet_v_iou(*it, index):
	                results.append(*it)
			}
			if results.size()!=0:
				return results;

		}
	    return results;
	}


	float overlaps_v(index1, index2)
	{
		float h1=this->heights[index1];
        float h2=this->heights[index2];
        float y0=std::max(this->text_proposals[index2].y, this->text_proposals[index1].y);
        float y1=std::max(this->text_proposals[index2].y+this->text_proposals[index2].height, this->text_proposals[index1].y+this->text_proposals[index1].height);
        return std::max(0, y1-y0+1.0)/std::min(h1, h2);
	}

	float size_similarity(index1, index2)
	{
        float h1=this->heights[index1];
        float h2=this->heights[index2];
        return std::min(h1, h2)/std::max(h1, h2);
	}
            

	bool meet_v_iou(int index1, int index2)
	{
        return overlaps_v(index1, index2)>=options.MIN_V_OVERLAPS and size_similarity(index1, index2)>=options.MIN_SIZE_SIM
	}
            
}


#endif