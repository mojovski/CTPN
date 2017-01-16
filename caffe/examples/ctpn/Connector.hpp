#ifndef CONNECTOR_H
#define CONNECTOR_H

#include <opencv2/opencv.hpp>
#include "Graph.hpp"

class Connector
{
public:
	typedef Graph::BoolMat BoolMat;
	struct TextLine{
		cv::Rect rect;
		float score;
	}
	typedef std::vector<TextLine> TextLines;

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
	std::vector<std::vector<int> > boxes_table; //stores to each horizontal pixel a list of indices
	Options options;

	Connector(){}

	Graph build_graph(std::vector<cv::Rect>& text_proposals, std::vector<float>& scores, cv::Size& im_size)
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

		int index=-1;
		for (std::vector<cv::Rect>::iterator text_proposals.begin(); it!=text_proposals.end(); it++)
		{
			index++;
			std::vector<int> successions=self.get_successions(index)
			if (successions.size()==0)
				continue;
			int succession_index=successions[arg_max<float>(scores, successions)];
			if (is_succession_node(index, succession_index))
			{
				graph[index, succession_index]=true;
			}
		}
		Graph g(graph, text_proposals.size(), text_proposals.size());
		return g;
	}



	TextLines getTextLines(std::vector<cv::Rect>& text_proposals, std::vector<float>& scores, cv::Size& im_size)
	{
		
		Graph graph=connector.build_graph(text_proposals, scores, im_size);
		std::vector<std::vector<int> > groups=graph.sub_graphs_connected();

		TextLines text_lines;

		for (std::vector<std::vector<int> >::iterator it=groups.begin(); it!=groups.end(); it++)
		{
			//text_line_boxes=text_proposals[list(tp_indices)]
			std::vector<cv::Rect> text_line_boxes=selectTextProposals(text_proposals, *it);
			//x0=np.min(text_line_boxes[:, 0]) //select minimal x
            //x1=np.max(text_line_boxes[:, 2]) //select minimal y
            cv::Rect bbox=getBoundingBox(text_line_boxes);
            float offset=(text_line_boxes[0].y-text_line_boxes[0].x)*0.5;
            //lt_y, rt_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0+offset, x1-offset)
            float lt_y, rt_y;
            fit_y(text_line_boxes, bbox.x+offset, bbox.x-offset, &lt_y, &rt_y);
            //lb_y, rb_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0+offset, x1-offset)
            float lb_y, rb_y;
            fit_y(text_line_boxes, bbox.x+offset, bbox.x-offset, &lb_y, &rb_y, false);
            /*
            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line
            score=scores[list(tp_indices)].sum()/float(len(tp_indices))
            */
            float score=getMeanScore(*it, scores);
            /*
            text_lines[index, 0]=x0
            text_lines[index, 1]=min(lt_y, rt_y)
            text_lines[index, 2]=x1
            text_lines[index, 3]=max(lb_y, rb_y)
            text_lines[index, 4]=score
            */
            TextLine tl;
            tl.rect.x=bbox.x;
            tl.rect.y=std::min(lt_y, rt_y);
            tl.rect.width=bbox.width;
            tl.rect.height=std::max(lb_y, rb_y)-bbox.y;
            tl.score=score;
            text_lines.push_back(tl);
		}

		return text_lines;

	}

	float getMeanScore(std::vector<int>& indices, std::vector<float>& scores)
	{
		float res=0;
		for (int i=0; i<indices.size(); i++)
			res=res+scores[indices[i]];
		res=res/((float)indices.size());
		return res;
	}

	/** fits a polynomial (1st order) to given x coordinates and given y heigh values.
	Returns the value of the fitted polynomial at p1 and p2 by writing them into y_p1 and y_p2
	if "top" is passed, then the top corner is fitted. if false, then the bottom.
	**/
	void fit_y(std::vector<cv::Rect>& text_boxes, float p1, float p2, float* y_p1, float* y_p2, bool top=true)
	{
		/*
		model: y=mx+b=(x,1)(x,b)^T = Aw with A(x,1)
		solution: w=(A^T*A)^-1 * A^T *Y
		*/
		//build matrix A
		cv::Mat A(x.size(),2, CV_64F); 
		cv::Mat Y(y.size(), 1, CV_64F);
		for (int i=0; i<x.size(); i++)
		{
			A.at<double>(i,0)=text_boxes[i].x;
			A.at<double>(i,1)=1.0;
			Y.at<double>(i,0)=text_boxes[i].y;
			if (!top)
				Y.at<double>(i,0)=Y.at<double>(i,0)+text_boxes[i].height;

		}
		cv::Mat AtA=A.t()*A;
		cv::Mat AtAinv=AtA.inv();
		cv::Mat w=AtAinv*Y;
		std::cout << "w: \n"<< w << std::endl;

		double m=w.at<double>(0,0);
		double bias=w.at<double>(1,0);
		(*y_p1)=m*p1+bias;
		(*y_p2)=m*p2+bias;
	}

	cv::Rect getBoundingBox(std::vector<cv::Rect>& elements)
	{
		int x0=1e4;
		int x1=0;
		int y0,y1;
		for (std::vector<cv::Rect>::iterator it=elements.begin(); it!=elements.end(); it++)
		{
			if (x0>(*it).x)
				{
					x0=(*it).x;
					y0=it->y;
				}
			if (x1<(*it).x)
			{
				x1=(*it).x;
				y1=(*it).y;
			}
		}
		cv::Rect rect;
		rect.x=x1;
		rect.y=y0;
		rect.width=(x1-x0);
		rect.height=(y1-y0);
		return rect;
	}

	std::vector<cv::Rect> selectTextProposals(std::vector<cv::Rect>& text_proposals, std::vector<int>& indices)
	{
		std::vector<cv::Rect> res;
		for(std::vector<int>::iterator it=indices.begin(); it!=indices.end(); it++)
		{
			res.push_back(text_proposals[*it]);
		}
		return res;
	}





protected:

	bool is_succession_node(int idx, int succession_index)
	{
		std::vector<int> precursors=get_precursors(succession_index)
		if (this->scores[index]>=this->scores[arg_max<float>(this->scores, precursors)])
		{
			return true;
		}
		return false;
	}

	/*returns the index where the arr has its maximum. 
	Only elements in arr are considered, which are indexed by "indices".
	**/
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
		cv::Rect box=this->text_proposals[index];
	    std::vector<int> results;

	    //for left in range((int)box.x+1, min(int(box[0])+cfg.MAX_HORIZONTAL_GAP+1, self.im_size[1])):
	    int min_pixel=std::max((int)box.x - options.MAX_HORIZONTAL_GAP, 0)-1;
		for (int left=(int)box.x-1; left > min_pixel; left--)
		{
			std::vector<int> adj_box_indices=this->boxes_table[left];
			for(std::vector<int>::iterator it=adj_box_indices.begin(); it!=adj_box_indices.end(); it++)
			{
				if (this->meet_v_iou(*it, index))
	                results.push_back(*it)
			}
			if results.size()!=0:
				return results;

		}
	    return results;
	}

	std::vector<int> get_successions(int index)
	{
		cv::Rect box=this->text_proposals[index];
	    std::vector<int> results;

	    //for left in range((int)box.x+1, min(int(box[0])+cfg.MAX_HORIZONTAL_GAP+1, self.im_size[1])):
	    int min_pixel=std::min((int)box.x+options.MAX_HORIZONTAL_GAP+1, this->im_size.width);
		for (int left=(int)box.x+1; left<min_pixel; left++)
		{
			std::vector<int> adj_box_indices=this->boxes_table[left];
			for(std::vector<int>::iterator it=adj_box_indices.begin(); it!=adj_box_indices.end(); it++)
			{
				if (this->meet_v_iou(*it, index))
	                results.push_back(*it)
			}
			if results.size()!=0:
				return results;

		}
	    return results;
	}


	float overlaps_v(int index1, int index2)
	{
		float h1=this->heights[index1];
        float h2=this->heights[index2];
        float y0=std::max(this->text_proposals[index2].y, this->text_proposals[index1].y);
        float y1=std::max(this->text_proposals[index2].y+this->text_proposals[index2].height, this->text_proposals[index1].y+this->text_proposals[index1].height);
        return std::max(0, y1-y0+1.0)/std::min(h1, h2);
	}

	float size_similarity(int index1, int index2)
	{
        float h1=this->heights[index1];
        float h2=this->heights[index2];
        return std::min(h1, h2)/std::max(h1, h2);
	}
            

	bool meet_v_iou(int index1, int index2)
	{
        return overlaps_v(index1, index2) >= options.MIN_V_OVERLAPS and size_similarity(index1, index2) >= options.MIN_SIZE_SIM;
	}
            
}


#endif