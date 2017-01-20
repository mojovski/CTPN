#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>


namespace Utils
{

	/**
	receives a RGB 8 bit image and a vector. 
	**/
	cv::Mat subtract_mean(cv::Mat& inp, cv::Vec3f& mean /*one pixel vector of byte*/)
	{
		cv::Mat out(inp.rows, inp.cols, CV_32FC3);
		inp.convertTo(out, CV_32FC3); 
		inp.convertTo(out, CV_32FC3); 
		//std::cout << "inp(100,100): " << inp.at<cv::Vec3b>(100,100) << std::endl;
		//std::cout << "out(100,100): " << out.at<cv::Vec3f>(100,100) << std::endl;

		for (int i=0; i<inp.rows; i++)
			for (int j=0; j<inp.cols; j++)
			{
				out.at<cv::Vec3f>(i,j)=(out.at<cv::Vec3f>(i,j)-mean);
			}
		return out;
	}

} //namespace utils end



#endif