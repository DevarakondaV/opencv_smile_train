#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <fstream>

#include <opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <boost/filesystem.hpp>


namespace fs = boost::filesystem;
using namespace std;
using namespace cv;

//function declerations
vector<string> getfilesinDir(const string & dir);
int getlabel(const string & dir);
void train_test(int nclasses, const Mat& train_data, const Mat &train_labels, const Mat & test_data, const Mat & test_labels, Mat & confusion);

int main() {

	//Training Data Directories
	string smiles_dir = "/home/vishnu/opencv_ws/CameraProj/MLData/Smile";	 	
	string other_dir = "/home/vishnu/opencv_ws/CameraProj/MLData/Other";	

	//List of Files in training dir
	vector<string> files_smile = getfilesinDir(smiles_dir);
	vector<string> files_other = getfilesinDir(other_dir);
	

	//shuffling files to prevent bias
	random_shuffle(files_smile.begin(),files_smile.end());
	random_shuffle(files_other.begin(),files_other.end());

	//definition vectors for testing and training data
	vector<string> Training;
	vector<string> Testing;
	//vector lengths	
	int Slen = files_smile.size();
	int Olen = files_other.size();
	//Testing lengths	
	int STestLen = (1.0/3)*(Slen);
	int OTestLen = (1.0/3)*(Olen);
	
	//Filling the testing file vectors
	for(int i=0;i<STestLen;++i)
	{
		Testing.push_back(files_smile[files_smile.size()-1]);
		files_smile.pop_back();	
	}

	for(int i=0;i<OTestLen;++i)
	{
		Testing.push_back(files_other[files_other.size()-1]);
		files_other.pop_back();
	}
	//cout << Testing.size() << "\t" << STrainLen+OTrainLen << endl;
	
	//Filling training file vectors
	Training.insert(Training.end(),files_smile.begin(),files_smile.end());
	Training.insert(Training.end(),files_other.begin(),files_other.end());
	
	//Shuffling vectors again
	random_shuffle(Testing.begin(),Testing.end());
	random_shuffle(Training.begin(),Training.end());

	int nclasses = 2;	//There are two classes
	cv::Mat confusion(nclasses,nclasses,CV_32S, cv::Scalar(0));	
	//2 classes so smile -> 1 Other -> 0
	Mat train_data, train_labels, test_data, test_labels;	 //Defining image vectors for training
	int label;	
	
	cout << Testing[Testing.size()-2] << endl;
	//Filling vectors train_data, train_labels	
	for (vector<string>::iterator it = Training.begin(); it != Training.end(); ++it)
	{
		label = getlabel(*it);
		//cout << *it << "\t" << label << endl;
		Mat image = imread(*it,0);
		if (image.empty()) {cerr << "Files doesn't Exist" << *it << endl; continue;}
		image.convertTo(image,CV_32F);
		Mat feature;
		image = image.reshape(1,1);
		normalize(image,feature,0,1,NORM_MINMAX,-1);
		
		train_data.push_back(feature);
		train_labels.push_back(label);
	}

	//filling vectors test_data, test_labels
	for (vector<string>::iterator it = Testing.begin();it != Testing.end();++it)
	{
		label = getlabel(*it);
		Mat image = imread(*it,0);
		if (image.empty()) {cerr << "File doesn't exist:" << *it << endl; continue;}
		image.convertTo(image,CV_32F);
		Mat feature;
		image = image.reshape(1,1);
		normalize(image,feature,0,1,NORM_MINMAX,-1);	
		//cout << feature << endl;
		test_data.push_back(feature);
		test_labels.push_back(label);
	}

	
	train_test(nclasses, train_data, train_labels, test_data, test_labels, confusion);
	//cout << test_labels.rows << "\t" << test_labels.cols << endl;
	//cout << train_labels.rows << "\t" << train_labels.cols << endl;
	/*
	for(int i=0;i<test_labels.rows;i++)
	{
		cout << test_labels.at<int>(i,0) << endl;
	}*/
}

int getlabel(const string & dir)
{
	if (dir.find("Smile") != -1)
		return 1;
	else
		return 0;
}

vector<string> getfilesinDir(const string & dir)
{
	vector<string> files;
	fs::path root(dir);
	fs::directory_iterator it_end;
	for (fs::directory_iterator it(root); it != it_end; ++it)
	{
		if (fs::is_regular_file(it->path()))
		{
			files.push_back(it->path().string());
		}
	}
	return files;
}

void train_test(int nclasses, const Mat& train_data, const Mat &train_labels, const Mat & test_data, const Mat & test_labels, Mat & confusion)
{

	//number of features
	int nfeatures = train_data.cols;
	//Ptr to neural network
	Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::create();
	
	
	//Defining neural network layers
	Mat_<int> layers(4,1);
	layers(0) = nfeatures;		
	layers(1) = nclasses*625;
	layers(2) = nclasses*312;
	layers(3) = nclasses;

	//defining neural net
	ann->setLayerSizes(layers);
	ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM,1,1);
	ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,500,0.00001));
	ann->setTrainMethod(ml::ANN_MLP::BACKPROP,0.00001);
	

	Mat train_classes = Mat::zeros(train_data.rows,nclasses,CV_32FC1);
	for(int i = 0; i<train_classes.rows;i++)
	{
		train_classes.at<float>(i,train_labels.at<int>(i)) = 1.f;

	}
	

	//Training network
	ann->train(train_data, ml::ROW_SAMPLE,train_classes);
	
	
	//saving trained network
	ann->save("nn1.yml");

	//Making predictions using testing data
	for(int i = 0; i<test_data.rows;i++)
	{
		int pred = ann->predict(test_data.row(i), noArray());
		int truth = test_labels.at<int>(i);
		confusion.at<int>(truth,pred)++;
		if (i == test_data.rows-2)
			cout << pred << "\t" << truth << endl;
	}
	
	//Determining accuracy and confusion matrix.
	Mat correct = confusion.diag();
	float accuracy = sum(correct)[0]/sum(confusion)[0];
	cerr << "acc: " << accuracy <<endl;
	cerr << "conf:\n"<<confusion << endl;
	
}


	
