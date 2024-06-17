#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

typedef pair<vector<KeyPoint>, vector<vector<float>>> feat; 

Mat padImageuchar(Mat image, int padSize){
    //pad by padSize on each side
    int h = (int)image.rows+2*padSize;
    int w = (int)image.cols+2*padSize;

    Mat paddedImage(h, w, CV_8UC1);
    for(int i = 0;i<h;++i){
        for(int j=0;j<w;++j){
            if((i<padSize) || (i>=padSize+image.rows) || (j<padSize) || (j>=padSize+image.cols)){
                paddedImage.at<uchar>(i,j) = (uchar)0;
                continue;
            }
            paddedImage.at<uchar>(i,j) = image.at<uchar>(i-padSize, j-padSize); 
        }
    }
    return paddedImage;
}

Mat padImagefloat(Mat image, int padSize){
    //pad by padSize on each side
    int h = (int)image.rows+2*padSize;
    int w = (int)image.cols+2*padSize;

    Mat paddedImage(h, w, CV_32F);
    for(int i = 0;i<h;++i){
        for(int j=0;j<w;++j){
            if((i<padSize) || (i>=padSize+image.rows) || (j<padSize) || (j>=padSize+image.cols)){
                paddedImage.at<float>(i,j) = (float)0;
                continue;
            }
            paddedImage.at<float>(i,j) = image.at<float>(i-padSize, j-padSize); 
        }
    }
    return paddedImage;
}

Mat gradient(Mat image, bool dir_x){
    int h = image.rows;
    int w = image.cols;

    image = padImageuchar(image, 1);
    Mat gradientImage(h, w, CV_32F);
    for(int i=0;i<h;++i)
        for(int j=0;j<w;++j){
            int v11 = (int)(image.at<uchar>(i,j));
            int v12 = (int)(image.at<uchar>(i,j+1));
            int v13 = (int)(image.at<uchar>(i,j+2));
            int v21 = (int)(image.at<uchar>(i+1,j));
            int v22 = (int)(image.at<uchar>(i+1,j+1));
            int v23 = (int)(image.at<uchar>(i+1,j+2));
            int v31 = (int)(image.at<uchar>(i+2,j));
            int v32 = (int)(image.at<uchar>(i+2,j+1));
            int v33 = (int)(image.at<uchar>(i+2,j+2));

            int tmp_add = v33-v11;
            int tmp_sub = v13-v31;
            int tmp_dx = v23-v21;
            int tmp_dy = v32-v12;
            int dx = tmp_add + tmp_sub + tmp_dx + tmp_dx;
            int dy = tmp_add - tmp_sub + tmp_dy + tmp_dy;
            int value;
            if(dir_x==true)
                value = dx;
            else
                value = dy;
            gradientImage.at<float>(i,j) = value;
        }
    return gradientImage;
}

Mat getKernel(int ksize, float sigma){
    //ksize should be odd;
    Mat kernel(ksize, ksize, CV_32F);
    float total = 0.0;

    for(int i = -ksize/2; i<=ksize/2; ++i){
        for(int j = -ksize/2; j<=ksize/2; ++j){
            float tmp = exp(-(i*i + j*j)/(2*sigma*sigma));
            kernel.at<float>(i+ksize/2,j+ksize/2) = tmp;
            total += tmp;
        }
    }
    kernel *= 1./total;
    return kernel;
}

Mat convolution(Mat image, Mat kernel){
    Mat convolvedImage(image.rows, image.cols, CV_32F);
    
    image = padImagefloat(image, kernel.rows/2); 
    for(int i = 0; i<convolvedImage.rows; ++i){
        for(int j = 0; j<convolvedImage.cols; ++j){
            float total = 0.0;
            for(int k = 0;k<kernel.rows; ++k)
                for(int l = 0;l<kernel.cols; ++l)
                    total += (kernel.at<float>(k,l)*image.at<float>(i+k,j+l));
            convolvedImage.at<float>(i,j) = total;
        }
    }
    return convolvedImage;
}

Mat nms(Mat harris, int winSize){
    //winSize should be odd
    int h = harris.rows;
    int w = harris.cols;

    harris = padImagefloat(harris, winSize/2);
    Mat nmsHarris(h, w, CV_32F);
    for(int i =0;i<h;++i){
        for(int j = 0;j<w;++j){
            float mx = harris.at<float>(i+winSize/2,j+winSize/2);
            for(int k = 0;k<winSize; k++)
                for(int l = 0; l<winSize; l++)
                    mx = max(mx, harris.at<float>(i+k, j+l));
            if(harris.at<float>(i+winSize/2,j+winSize/2) == mx)
                nmsHarris.at<float>(i,j) = harris.at<float>(i+winSize/2, j+winSize/2);
            else
                nmsHarris.at<float>(i,j) = 0.0;
        }
    }
    return nmsHarris;
}
Mat corners(Mat image){
    //using harris measure l1*l2-k*(l1+l2)  k=0.1
    Mat I_x2,I_y2,I_xy,dx,dy;

    dx = gradient(Mat(image), true);
    dy = gradient(Mat(image), false);
    I_x2 = dx.mul(dx);
    I_y2 = dy.mul(dy);
    I_xy = dx.mul(dy);

    Mat mask = getKernel(5, 0.5);
    I_x2 = convolution(I_x2, mask);
    I_y2 = convolution(I_y2, mask);
    I_xy = convolution(I_xy, mask);

    Mat harris(image.rows, image.cols, CV_32F);
    for(int i = 0; i<image.rows; ++i){
        for(int j =0; j<image.cols; ++j){
            harris.at<float>(i,j) = (I_x2.at<float>(i,j)*I_y2.at<float>(i,j))-pow(I_xy.at<float>(i,j),2);
            harris.at<float>(i,j) -= (0.1*pow(I_x2.at<float>(i,j)+I_y2.at<float>(i,j),2));
        }
    }
    Mat nmsHarris = nms(harris, 7);
    return nmsHarris;
}
bool sortStrength(const pair<float, KeyPoint> &a, const pair<float, KeyPoint> &b){
    return a.first<b.first;
}

vector<KeyPoint> topCorners(Mat harris, int N){
    int h = harris.rows;
    int w = harris.cols;

    vector<pair<float, KeyPoint>> temp;
    vector<KeyPoint> keypoints;
    for(int i = 0;i<h; ++i)
        for(int j = 0;j<w; ++j)
            temp.push_back({harris.at<float>(i,j), KeyPoint(j,i,1)});

    sort(temp.begin(),temp.end(), sortStrength);
    for(int i = 0;i<N;++i){
        auto it = temp.end()-i-1;
        keypoints.push_back((*it).second);
    }
    return keypoints;
}

feat getFeatures(Mat image, vector<KeyPoint> keypoints){
    int h = image.rows;
    int w = image.cols;

    image.convertTo(image, CV_32F);
    Mat mask = getKernel(5, 0.5);
    image = convolution(image, mask);

    feat features;
    int padSize = 20;
    image = padImagefloat(image, padSize);
    for(int i = 0;i<keypoints.size(); i++){
        vector<float> descriptor;
        Mat des(2*padSize,2*padSize,CV_32F);
        for(int k = 0;k<2*padSize; k++)
            for(int l = 0; l<2*padSize; l++)
                des.at<float>(k,l) = image.at<float>((int)(keypoints[i].pt.y)+k, (int)(keypoints[i].pt.x)+l);
        
        Mat resized;
        resize(des, resized, cv::Size(), 0.2, 0.2);
        for(int k = 0; k<resized.rows; k++)
            for(int l = 0; l<resized.cols;l++)
                descriptor.push_back(resized.at<float>(k,l));
        features.first.push_back(keypoints[i]);
        features.second.push_back(descriptor);
    }
    return features;
}

float getDistance(vector<float> descriptor1, vector<float> descriptor2){
    float distance = 0;
    for(int i = 0;i<descriptor1.size(); i++){
        distance += pow(descriptor1[i]-descriptor2[i],2);
    }
    return distance;
}

vector<DMatch> matching(feat features1, feat features2){
    //matches from features1 to features2
    float thresh = 100000;
    vector<DMatch> matches;
    for(int i = 0; i<features1.second.size(); i++){
        int matchidx = 0;
        float mi = getDistance(features1.second[i], features2.second[0]);
        for(int j = 1; j<features2.second.size(); j++){
            float tmp = getDistance(features1.second[i], features2.second[j]);
            if(tmp<mi){
                matchidx = j;
                mi = tmp;
            }
        }
        if(mi > thresh)
            continue;
        matches.push_back(DMatch(i, matchidx, mi));
    }
    return matches;
}

Mat calculateAffine(feat features1, feat features2, vector<DMatch> matches){
    int n = matches.size();
    Mat sourceSet(n, 2, CV_32F);
    Mat destSet(n, 2, CV_32F);
    for(int i = 0;i<n; ++i){
        sourceSet.at<float>(i,0) = features2.first[matches[i].trainIdx].pt.x;
        sourceSet.at<float>(i,1) = features2.first[matches[i].trainIdx].pt.y;
        destSet.at<float>(i,0) = features1.first[matches[i].queryIdx].pt.x;
        destSet.at<float>(i,1) = features1.first[matches[i].queryIdx].pt.y;
    }
    return estimateAffine2D(sourceSet, destSet);
}

Mat combine(Mat image1, Mat image2){
    Mat combined(image1.rows, image1.cols, CV_8UC3);
    for(int i = 0;i<image1.rows; ++i){
        for(int j = 0;j<image1.cols; ++j){
            Matx<int, 1, 3> pixel1 = image1.at<Matx<uchar, 1, 3>>(i,j);
            Matx<int, 1, 3> pixel2 = image2.at<Matx<uchar, 1, 3>>(i,j);
            Matx<int, 1, 3> u(1,1,1);
            if(pixel1.dot(u)==0 && pixel2.dot(u)==0)
                combined.at<Matx<uchar, 1, 3>>(i,j) = pixel1;
            else if(pixel1.dot(u)==0 && pixel2.dot(u)!=0)
                combined.at<Matx<uchar, 1, 3>>(i,j) = 0.95*pixel2;
            else if(pixel1.dot(u)!=0 && pixel2.dot(u)==0)
                combined.at<Matx<uchar, 1, 3>>(i,j) = 0.95*pixel1;
            else
                combined.at<Matx<uchar, 1, 3>>(i,j) = 0.5*pixel1 + 0.5*pixel2;
        }
    }
    return combined;
}

Mat removeBlackBorder(Mat result){
    Mat gray, thresh;
    cvtColor(result, gray, COLOR_BGR2GRAY);
    Rect roi = boundingRect(gray);
    return result(roi);
}

Mat pairwiseStiching(Mat image1, Mat image2){
    int N = 150; //top N strengths to pick as corners from an image
    Mat grayImage1, grayImage2, blurred;
    
    //image1 computation
    resize(image1, image1, Size(800,600));
    cvtColor(image1, grayImage1, COLOR_BGR2GRAY);
    GaussianBlur(grayImage1, blurred, Size(5,5), 0.5);
    Mat harris1 = corners(blurred);
    vector<KeyPoint> idx = topCorners(harris1, N);
    feat features1 = getFeatures(grayImage1, idx);

    //image2 computation
    resize(image2, image2, Size(800,600));
    cvtColor(image2, grayImage2, COLOR_BGR2GRAY);
    GaussianBlur(grayImage2, blurred, Size(5,5), 0.5);
    Mat harris2 = corners(blurred);
    idx = topCorners(harris2, N);
    feat features2 = getFeatures(grayImage2, idx);

    //perform matching
    vector<DMatch> matches = matching(features1, features2);

    //get affine matrix
    Mat affine = calculateAffine(features1, features2, matches);

    //perform stiching adjustments
    Mat M1, image1_temp, image2_temp, result;
    int h = max(image1.rows, image2.rows);
    int w = max(image1.cols, image2.cols);
    M1.push_back(vector<float>{1, 0, (float)h/3, 0, 1, (float)w/3});
    M1 = M1.reshape(0,2);
    Size s(image1.rows+image2.rows, image1.cols+image2.cols);
    //translate image1
    warpAffine(image1, image1_temp, M1, s);
    //warp image2
    warpAffine(image2, image2_temp, affine, s);
    //translate warped image2
    warpAffine(image2_temp, image2_temp, M1, s);
    
    //combine images
    result = combine(image1_temp, image2_temp);
    
    //eliminate black border;
    return removeBlackBorder(result);
}

int main( int argc, char** argv )
{   
    vector<string> filenames;
    string path = "Dataset/*.jpg";
    glob(path, filenames, false);

    Mat stiched = imread(filenames[0], IMREAD_COLOR);
    for(int i = 1;i<filenames.size(); ++i){
        cout << i << endl;
        Mat image = imread(filenames[i], IMREAD_COLOR);
        Mat result = pairwiseStiching(stiched, image);
        stiched = result;
    }
    imwrite("stiched.jpg", stiched);
    return 0;
}