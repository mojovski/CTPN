//
// Created by eugen on 10.01.2017
//

#ifndef CTPN_H
#define CTPN_H

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include "boost/python.hpp"
namespace bp = boost::python;

#include "Utils.hpp"
#include "Connector.hpp"
#include "Graph.hpp"


using namespace caffe;  // NOLINT(build/namespaces)

class CTPN {
public:
	class Options{
	public:
		Options():score_threshold(0.7),nms_threshold(0.3){};
		float score_threshold;
		float nms_threshold;
	};


	Options options;
	std::vector<cv::Rect> text_proposals;
	std::vector<float> scores;

	CTPN(
		const string& model_file,
		const string& trained_file,
		cv::Vec3f& mean
		)
	{
		this->_mean=mean;
		#ifdef CPU_ONLY
		  Caffe::set_mode(Caffe::CPU);
		#else
		  Caffe::set_mode(Caffe::GPU);
		#endif

		  /* Load the network. */
		  _net.reset(new Net<float>(model_file, TEST));
		  _net->CopyTrainedLayersFrom(trained_file);

		  //std::cout << "_net->num_inputs(): " << _net->num_inputs() << std::endl;
		  //std::cout << "_net->num_outputs(): " << _net->num_outputs() << std::endl;

		  //CHECK_EQ(_net->num_inputs(), 1) << "Network should have exactly one input.";
		  //CHECK_EQ(_net->num_outputs(), 1) << "Network should have exactly one output.";

		  const boost::shared_ptr<caffe::Blob<float> > input_layer = _net->blob_by_name("data");//_net->input_blobs()[0];
		  _num_channels = input_layer->channels();
		  CHECK(_num_channels == 3)
			<< "Input layer should have 3 channels.";
		  _input_geometry = cv::Size(input_layer->width(), input_layer->height());
		  std::cout << "The NET has input geometry of (wxh):" << _input_geometry << std::endl;
	}

	void process(cv::Mat& img)
	{

		//prepare the input in the net.
		Blob<float>* input_layer = _net->input_blobs()[0];
		input_layer->Reshape(1, _num_channels,
			_input_geometry.height, _input_geometry.width);

		//write to im_info layer
		const boost::shared_ptr<caffe::Blob<float> > input_layer_im_info = _net->blob_by_name("im_info"); //()[1];
		input_layer_im_info->Reshape(1,2,1,1);
		float* im_info_data=input_layer_im_info->mutable_cpu_data();
		*(im_info_data)=(float)_input_geometry.height;
		*(im_info_data+1)=(float)_input_geometry.width;

		/* Forward dimension change to all layers. */
		_net->Reshape();

		//create cv::Mat wrapper pointing to the memory managed by caffe
		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels);

		//prepare the image
		cv::Mat sample=preprocess(img);
		//std::cout << "sample: " << sample.at<cv::Vec3f>(100,100) << std::endl;
		/* This operation will write the separate BGR planes directly to the
		* input layer of the network because it is wrapped by the cv::Mat
		* objects in input_channels. */
		cv::split(sample, input_channels);
		//std::cout << "channel[0](100,100): " << input_channels[0].at<float>(100,100) << std::endl;
		//std::cout << "channel[1](100,100): " << input_channels[1].at<float>(100,100) << std::endl;
		//std::cout << "channel[2](100,100): " << input_channels[2].at<float>(100,100) << std::endl;

		CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
			== _net->input_blobs()[0]->cpu_data())
			<< "Input channels are not wrapping the input layer of the network.";
		//std::cout << "*_net->input_blobs()[0]->cpu_data(): " << *(_net->input_blobs()[0]->cpu_data()+200000) << std::endl;
		//std::cout << "*input_channels.at(0).data: " << *((float*)input_channels.at(0).data+200000) << std::endl;

		//run the forward stuff
		try{
			_net->ForwardPrefilled();
		} catch (bp::error_already_set) {
	      PyErr_Print();
	    }

		//check if there are any zero weights (indicating errors)
		//const vector< shared_ptr< Blob< float > > > params=_net->params();
		//for (int pi=0; pi<params.size(); pi++)
		//{
		//	std::cout << ": " << *params[pi]->cpu_data()<< ". ";
		//}

		//http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1Net.html
		//boost::shared_ptr< Blob< float > > rois=_net->blob_by_name("rois");
		//boost::shared_ptr< Blob< float > > scores=_net->blob_by_name("scores");

		const Blob< float >* rois=_net->output_blobs()[0];
		const Blob< float >* scores=_net->output_blobs()[1];

		//for (int si=0; si<rois->shape()[0]*rois->shape()[1]; si++)
		//{
		//	std::cout << " x: " << *(rois->cpu_data()+si);
		//}

		std::vector<int> shape_rois=rois->shape();
		std::vector<int> shape_scores=scores->shape();
		/*//print shapes
		std::cout << "shape_rois: [";
		for (int i=0; i<shape_rois.size(); i++)
		{
			std::cout << shape_rois[i] << ", ";
		}
		std::cout << "]\n";
		
		std::cout << "shape_scores: [";
		for (int i=0; i<shape_scores.size(); i++)
		{
			std::cout << shape_scores[i] << ", ";
		}
		std::cout << "]\n";
		*/


		convertBlobToRectAndScores(rois, scores);
		//before here: check. same results as in python.

		std::vector<int> indices_to_keep=applyNMS(options.nms_threshold); //CHECK. 1:1 to the python version.
		
		filterRects(indices_to_keep);
		/*std::cout << "\nText proposals before connector:\n";
		for (int i=0; i<10; i++)
  			std::cout << "x: " << text_proposals[i].x << ' ';
		*/
		

		std::cout << "Forward DNN processing done!\n";
		normalize_scores();

		Connector connector;
		text_lines=connector.getTextLines(this->text_proposals, this->scores, this->_input_geometry);

		std::cout << "Connector returned text lines:\n";
		for (int i=0; i<10; i++)
		{
			std::cout << "x: " << text_lines[i].rect.x;
		}


	}

	void normalize_scores()
	{
	    std::vector<float> n_scores;
	    float min_score=*min_element(scores.begin(), scores.end());
	    float max_score=*max_element(scores.begin(), scores.end());
	    
	    for (std::vector<float>::iterator it=scores.begin(); it!=scores.end(); it++)
	    {
	    	float n_score=((*it)-min_score)/(max_score-min_score);
	    	n_scores.push_back(n_score);
	    }
	    scores=n_scores;

	}

	void filterRects(std::vector<int>& indices_to_keep)
	{
		std::vector<cv::Rect> _text_proposals_tmp;
		std::vector<float> _scores_tmp;
		Connector::TextLines _text_lines;

		for (int i=0; i<indices_to_keep.size(); i++)
		{
			_text_proposals_tmp.push_back(this->text_proposals[indices_to_keep[i]]);
			_scores_tmp.push_back(this->scores[indices_to_keep[i]]);
			//DEBUG
			//std::cout << "  (" << text_proposals[indices_to_keep[i]].x << ", " << text_proposals[indices_to_keep[i]].y << ")";

		}
		this->scores=_scores_tmp;
		this->text_proposals=_text_proposals_tmp;
	}

	void drawResults(cv::Mat& img)
	{
		std::cout << "Drawing results into image\n";
		cv::Scalar color( 40, 255, 200 );
		for(Connector::TextLines::iterator it=text_lines.begin(); it!=text_lines.end(); it++)
		{
			//std::cout << "rect.x: " << (*it).rect.x << std::endl;
			cv::rectangle(img, (*it).rect, color);
		}
	}

	/*void convertBlobToRectAndScores(boost::shared_ptr< Blob< float > >& rois_blob, boost::shared_ptr< Blob< float > >& scores_blob,
		std::vector<int>& idx_to_keep)
	{*/
	void convertBlobToRectAndScores(const Blob< float >* rois_blob, const Blob< float >* scores_blob)
	{
		for (int i=0; i<rois_blob->shape()[0]; i++)
		{
			cv::Rect r(rois_blob->data_at(i,0,0,0)+0.5 //x
				rois_blob->data_at(i, 1,0,0)+0.5, //y
				rois_blob->data_at(i, 2,0,0)-rois_blob->data_at(i, 0,0,0), //width
				rois_blob->data_at(i, 3,0,0)-rois_blob->data_at(i, 1,0,0) //height
				);
			float score=scores_blob->data_at(i, 0,0,0);
			//std::cout << "s: " << score << std::endl;
			if (score<options.score_threshold)
				continue;
			//std::cout << "rect: " << (r) << std::endl;
			
			this->text_proposals.push_back(r);
			this->scores.push_back(score);
		}

	}

	//sorts an array and keeps track of the resorted indices. see post on stackoverflow: 
	//http://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
	/*@brief returns new indices after sorting the passed vector.
	Example: passed vector: [1,4,3], is sorted to [1,3,4], and the indices returned:[0,2,1]
	*/
	template <typename T>
	vector<size_t> sort_indexes(const vector<T> &v) {
	  // initialize original index locations
	  vector<size_t> idx(v.size());
	  iota(idx.begin(), idx.end(), 0);
	  // sort indexes based on comparing values in v
	  sort(idx.begin(), idx.end(),
	       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
	  return idx;
	}


	/** Applies non maximum suppression (NMS) and 
	returns only indices which should be kept
	**/
	/*std::vector<int> applyNMS(boost::shared_ptr< Blob< float > >& rois_blob, boost::shared_ptr< Blob< float > >& scores_blob, 
		float min_score_threshold)
	{
	*/
	std::vector<int> applyNMS(float min_threshold)
	{
		
		int ndets=scores.size();

		std::vector<size_t> order=sort_indexes<float>(scores);
		// std::cout << "Order before NMS:\n";
		// for (int oi=0; oi<10; oi++)
		// {
		// 	std::cout << " " << order[oi];//<< ":" << scores[order[oi]];
		// }
		// std::cout << "\n";
		
		//std::cout << "argsort done! order.size: " << order.size() << "\n";
		//a vector to keep track of suppressed elements
		std::vector<int> suppressed(ndets,0);

		//nominal indices
		int _i, _j;
		// sorted indices
   		int i, j;
   		//temp variables for box i's (the box currently under consideration)
    	float ix1, iy1, ix2, iy2, iarea;
    	//variables for computing overlap with box j (lower scoring box)
	    float xx1, yy1, xx2, yy2;
	    float w, h;
	    float inter, ovr;
	    std::vector<int> keep;
	    for (_i=0; _i<ndets; _i++)
	    {
	    	//std::cout << "i: " << _i << std::endl;
	        i = order[_i];
	        if (suppressed[i] == 1)
	            continue;
	        keep.push_back(i);
	        //std::cout << "tokeep++: " << i << "\t";
	        ix1 = text_proposals[i].x;
	        iy1 = text_proposals[i].y;
	        ix2 = text_proposals[i].x+text_proposals[i].width;
	        iy2 = text_proposals[i].y+text_proposals[i].height;
	        iarea = (text_proposals[i].width+1)*(text_proposals[i].height+1);
	        for (_j=(_i+1); _j<ndets; _j++)
	        {
	            j = order[_j];
	            if (suppressed[j] == 1)
	                continue;
	            xx1 = std::max(ix1, (float)text_proposals[j].x);
	            yy1 = std::max(iy1, (float)text_proposals[j].y);
	            xx2 = std::min(ix2, (float)(text_proposals[j].x+text_proposals[j].width));
	            yy2 = std::min(iy2, (float)(text_proposals[j].y+text_proposals[j].height));
	            w = std::max(0.0f, xx2 - xx1 + 1);
	            h = std::max(0.0f, yy2 - yy1 + 1);
	            inter = w * h;
	            ovr = inter / (iarea + (text_proposals[j].width+1)*(text_proposals[j].height+1) - inter);
	            if (ovr >= min_threshold)
	                suppressed[j] = 1;
	        }
	    }
	    //std::cout << "Return kept indices!\n";
	    return keep;
	}

	/* Wrap the input layer of the network in separate cv::Mat objects
	 * (one per channel). This way we save one memcpy operation and we
	 * don't need to rely on cudaMemcpy2D. The last preprocessing
	 * operation will write the separate channels directly to the input
	 * layer. 
	 It basically creates cv::Mat objets, pointing to the memory managed by caffe*/
	void WrapInputLayer(std::vector<cv::Mat>* input_channels) 
	{
		const boost::shared_ptr<caffe::Blob<float> > input_layer = _net->blob_by_name("data");//_net->input_blobs()[0];

		int width = input_layer->width();
		int height = input_layer->height();
		float* input_data = input_layer->mutable_cpu_data();
		for (int i = 0; i < input_layer->channels(); ++i) {
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		}
	}

	cv::Mat preprocess(cv::Mat& img)
	{
		//1. subtract mean
		//std::cout << "Subtracting mean: " << this->_mean << std::endl;
		cv::Mat normalized_img=Utils::subtract_mean(img, this->_mean);
		//std::cout << "normalized_img: " << normalized_img.at<cv::Vec3f>(100,100) << std::endl;

		cv::Mat sample_resized;
		if (normalized_img.size() != _input_geometry)
		{
		  	//std::cout << "Resizing image to " << _input_geometry << std::endl;
			cv::resize(normalized_img, sample_resized, _input_geometry);
		}
		  else
			sample_resized = normalized_img;
		return sample_resized;
	}

	cv::Size getImgGeometry(){return _input_geometry;}

protected:
	cv::Vec3f _mean;
	shared_ptr<Net<float> > _net;
	cv::Size _input_geometry;
	int _num_channels;
	Connector::TextLines text_lines;
};


#endif //CTPN_H
