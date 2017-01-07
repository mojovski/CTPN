//
// Created by eugen on 10.01.2017
//

#ifndef CTPN_H
#define CTPN_H

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Utils.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

class CTPN {
public:
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

		  std::cout << "_net->num_inputs(): " << _net->num_inputs() << std::endl;
		  std::cout << "_net->num_outputs(): " << _net->num_outputs() << std::endl;

		  //CHECK_EQ(_net->num_inputs(), 1) << "Network should have exactly one input.";
		  //CHECK_EQ(_net->num_outputs(), 1) << "Network should have exactly one output.";

		  Blob<float>* input_layer = _net->input_blobs()[0];
		  _num_channels = input_layer->channels();
		  CHECK(_num_channels == 3)
			<< "Input layer should have 3 channels.";
		  _input_geometry = cv::Size(input_layer->width(), input_layer->height());
	}

	void process(cv::Mat& img)
	{
		//prepare the input in the net.
		Blob<float>* input_layer = _net->input_blobs()[0];
		input_layer->Reshape(1, _num_channels,
			_input_geometry.height, _input_geometry.width);
		/* Forward dimension change to all layers. */
		_net->Reshape();

		//create cv::Mat wrapper pointing to the memory managed by caffe
		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels);

		//prepare the image
		cv::Mat sample=preprocess(img);
		/* This operation will write the separate BGR planes directly to the
		* input layer of the network because it is wrapped by the cv::Mat
		* objects in input_channels. */
		cv::split(sample, input_channels);
		CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
			== _net->input_blobs()[0]->cpu_data())
			<< "Input channels are not wrapping the input layer of the network.";

		//run the forward stuff
		_net->ForwardPrefilled();

		/* Copy the output layer to a std::vector */
		/*Blob<float>* output_layer = _net->output_blobs()[0]; //how many blobs are there???
		const float* begin = output_layer->cpu_data();
		const float* end = begin + output_layer->channels();
		return std::vector<float>(begin, end);
		*/

	}

	/* Wrap the input layer of the network in separate cv::Mat objects
	 * (one per channel). This way we save one memcpy operation and we
	 * don't need to rely on cudaMemcpy2D. The last preprocessing
	 * operation will write the separate channels directly to the input
	 * layer. 
	 It basically creates cv::Mat objets, pointing to the memory managed by caffe*/
	void WrapInputLayer(std::vector<cv::Mat>* input_channels) 
	{
		Blob<float>* input_layer = _net->input_blobs()[0];

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
		cv::Mat normalized_img=Utils::subtract_mean(img, this->_mean);
		cv::Mat sample_resized;
		  if (normalized_img.size() != _input_geometry)
			cv::resize(normalized_img, sample_resized, _input_geometry);
		  else
			sample_resized = normalized_img;
		return sample_resized;
	}

protected:
	cv::Vec3f _mean;
	shared_ptr<Net<float> > _net;
	cv::Size _input_geometry;
	int _num_channels;
};


#endif //CTPN_H
