#include<bits/stdc++.h>
#include<opencv2/opencv.hpp>
#include "ds.cpp"

class Gaussian{
public:
	cv::Vec<float, 3> mean;
	float var;

	Gaussian(){}
	Gaussian(cv::Vec<float,3> mean, float var){
		this->mean = mean;
		this->var = var;
	}

	double pdf(cv::Vec<float, 3> X){
		double c = sqrt(pow(4*acos(0.0),3)*pow(var,3));
		double res = 0;
		for(int i = 0 ;i<X.channels;++i){
			res += (X-mean)[i]*(X-mean)[i];
		}
		res /= ((-2)*var);
		return exp(res)/c;
	}
};

typedef vector<pair<float, Gaussian>> gmm;

bool sortGaussian(const pair<float, Gaussian> &a, const pair<float,Gaussian> &b){
	return ((a.first/sqrt(a.second.var)) > (b.first/sqrt(b.second.var)));
}

class BackgroundSubtractor{
public:
	int K, C;
	float lr,T;
	vector<vector<gmm>> model;

	BackgroundSubtractor(){}
	BackgroundSubtractor(int, float, float, int);
	void apply(cv::Mat image);
	cv::Mat predict(cv::Mat image);
	cv::Mat refineImage(cv::Mat foreground);
	double mahalanobis(cv::Vec<float, 3>, cv::Vec<float, 3>, float);
};

BackgroundSubtractor:: BackgroundSubtractor(int noOfGaussians, float learningRate, float threshold, int minComponents){
	K = noOfGaussians;
	lr = learningRate;
	T = threshold;
	C = minComponents;
}

double BackgroundSubtractor::mahalanobis(cv::Vec<float, 3> v1, cv::Vec<float, 3> v2, float var){
	double res = 0;
	for(int i = 0;i<v1.channels; ++i){
		res+= (v1-v2)[i]*(v1-v2)[i];
	}
	return sqrt(res/var);
}

void BackgroundSubtractor::apply(cv::Mat image){
	int height = image.rows;
	int width = image.cols;

	if(model.empty()){
		for(int i = 0;i<height;i++){
			model.push_back(vector<gmm>(width));
			for(int j = 0;j<width;j++){
				model[i].push_back(gmm(K));
				for(int k = 0; k<K;k++){
					model[i][j].push_back(pair<float, Gaussian>(0.0, Gaussian({0.0, 0.0,0.0},1)));
				}
			}
		}
	}
	for(int i = 0; i<height; i++){
		for(int j = 0;j<width; j++){
			int bestGaussian = -1;
			float total = 0.0;
			cv::Vec<float, 3> pixel = (cv::Vec<float ,3>)image.at<cv::Vec<uchar, 3>>(i,j);

			for(int k = 0; k<K; k++){
				if((mahalanobis(pixel,model[i][j][k].second.mean, model[i][j][k].second.var) < 3.0f)
					 && (bestGaussian==-1)){
					bestGaussian=k;
					model[i][j][k].first = (1-lr)*model[i][j][k].first+lr;
					double rho = lr*model[i][j][k].second.pdf(pixel);
					model[i][j][k].second.mean = (1-rho)*model[i][j][k].second.mean + rho*pixel;
					double ans = 0;
					for(int a = 0; a<pixel.channels; ++a)
						ans += pow((pixel[a]-model[i][j][k].second.mean[a]),2);
					model[i][j][k].second.var = (1-rho)*model[i][j][k].second.var + rho*ans;
					
					//bound variance between 4.0 and 75.0
					model[i][j][k].second.var = std::max(4.0f, model[i][j][k].second.var);
					model[i][j][k].second.var = std::min(75.0f, model[i][j][k].second.var);
				}
				else {
					model[i][j][k].first = (1-lr)*model[i][j][k].first;
				}
			}
			sort(model[i][j].begin(), model[i][j].end(), sortGaussian);
			if(bestGaussian==-1){
				model[i][j][K-1].first = lr;
				model[i][j][K-1].second.mean = pixel;
				model[i][j][K-1].second.var = 15; 
			}

			//normalize the weights
			for(int k = 0;k<K;++k)
				total += model[i][j][k].first;
			for(int k = 0;k<K;++k)
				model[i][j][k].first /= total;
			sort(model[i][j].begin(), model[i][j].end(), sortGaussian);
		}
	}
}

cv::Mat BackgroundSubtractor::predict(cv::Mat image){
	int height = image.rows;
	int width = image.cols;
	float total;
	cv::Mat foreground(height, width, CV_8UC1, cv::Scalar(0));

	for(int i = 0;i<height; ++i)
		for(int j = 0;j<width; ++j){
			int B;
			total = 0.0;
			cv::Vec<float, 3> pixel = (cv::Vec<float ,3>)image.at<cv::Vec<uchar, 3>>(i,j);
			sort(model[i][j].begin(), model[i][j].end(), sortGaussian);

			for(B = 0; (B<K) && (total<=T); ++B){
				total+= model[i][j][B].first;
			}
			//B is the no of gaussians describing the background model
			for(int k = 0;k<B;++k){
				if(mahalanobis(pixel, model[i][j][k].second.mean, model[i][j][k].second.var)<9.0f){
					break;
				}
				if(k==B-1){
					foreground.at<uchar>(i,j) = (uchar)255;
				}
			}
		}
	
  	return foreground;
}

cv::Mat BackgroundSubtractor::refineImage(cv::Mat foreground){
	int height = foreground.rows;
	int width = foreground.cols;

	cv::Mat clearedImage(height, width, CV_8UC1, cv::Scalar(0));

	vector<vector<int>> vect_foreground(height, vector<int>(width, 0));
	for(int i = 0; i<height; ++i){
		for(int j = 0;j<width; ++j){
			vect_foreground[i][j] = (int)foreground.at<uchar>(i,j);
		}
	}
	vector<vector<pair<int,int>>> components = connectedComponent(vect_foreground);

	for(int i = 0 ; i<components.size() ; ++i){
		for(int j=0; (j< components[i].size()) && (components[i].size()>C); ++j){
			int p = components[i][j].first, q = components[i][j].second;
			clearedImage.at<uchar>(p, q) = (uchar)vect_foreground[p][q];
		}
	}
	return clearedImage;
}

int main(){
	cv::Mat image, foreground, clearedImage;
	BackgroundSubtractor bgs(5,0.003, 0.8, 50);

	//give dataset filename here
	//cv::VideoCapture capture("COL780 Dataset/Candela_m1.10/input/Candela_m1.10_%6d.png");
	cv::VideoCapture capture("COL780 Dataset/IBMtest2/input/in%6d.png");
	cv::Size S = cv::Size((int) capture.get(cv::CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) capture.get(cv::CAP_PROP_FRAME_HEIGHT));
	
	//give output file name here
	cv::VideoWriter outputVideo("IBMtest2.avi", cv::VideoWriter::fourcc('M','P','4','2'), 15, S,false);
	int i = 0;
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE );
	while(1){
		std::cout << "Frame no: "<< ++i << std::endl;
		capture >> image;
		bgs.apply(image);
		foreground = bgs.predict(image);
		clearedImage = bgs.refineImage(foreground);
		cv::imshow("Display window", clearedImage);
		outputVideo << clearedImage;
		if(cv::waitKey(10)==27){
			break;
		}
	}
}