#pragma once
#include <iostream>
#include<memory>
#include <opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h> 
#include<iostream>
#include <numeric>
#include<io.h>

struct OutputSeg {
	int id;//类别          
	float confidence; //类别的置信度  
	cv::Rect box; //类别的框      
	cv::Mat boxMask; //框内的掩码Mask 
};
struct MaskParams {
	int segChannels = 32;
	int segWidth = 160;//seg分支的输入尺寸（宽）
	int segHeight = 160;//seg分支的输入尺寸（高）
	int netWidth = 640; //模型输入图片的尺寸（宽）
	int netHeight = 640;//模型输入图片的尺寸（高）
	float maskThreshold = 0.5;
	cv::Size srcImgShape;
	cv::Vec4d params;

};

class Yolov8SegOnnx {
public:
	Yolov8SegOnnx() :_OrtMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput)) {};
	~Yolov8SegOnnx() {};

	bool ReadModel(const std::string& modelPath, bool isCuda = false, int cudaID = 0, bool warmUp = false);//读模型
	bool OnnxDetect(const cv::Mat& srcImg, std::vector<OutputSeg>& output);//图片推理过程
	void DrawPred(cv::Mat& img, cv::Mat* outImage, std::vector<OutputSeg> result, std::vector<std::string> classNames);//画图，显示
	std::vector<std::string> _className = {
		"box"
	};//类别名称,需要根据实际修改

	const int _netWidth = 640;   //模型输入图片的尺寸（宽），需要根据模型进行修改
	const int _netHeight = 640;  //模型输入图片的尺寸（高），需要根据模型进行修改
	int _batchSize = 1;  //批次设置
	bool _isDynamicShape = false;//模型是否支持动态设置
	float _classThreshold = 0.25;
	float _nmsThreshold = 0.45;
	float _maskThreshold = 0.5;

	template <typename T>
	T VectorProduct(const std::vector<T>& v)
	{
		return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
	};
	bool OnnxBatchDetect(std::vector<cv::Mat>& srcImg, std::vector<std::vector<OutputSeg>>& output);
	bool CheckModelPath(std::string modelPath);
	bool CheckParams(int netHeight, int netWidth, const int* netStride, int strideSize);
	int Preprocessing(const std::vector<cv::Mat>& SrcImgs, std::vector<cv::Mat>& OutSrcImgs, std::vector<cv::Vec4d>& params);
	void LetterBox(const cv::Mat& image, cv::Mat& outImage,
		cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
		const cv::Size& newShape = cv::Size(640, 640),//模型的输入尺寸，需要根据模型进行修改
		bool autoShape = false,
		bool scaleFill = false,
		bool scaleUp = true,
		int stride = 32,
		const cv::Scalar& color = cv::Scalar(114, 114, 114));
	void GetMask2(const cv::Mat& maskProposals, const cv::Mat& maskProtos, OutputSeg& output, const MaskParams& maskParams);

	//ONNXRUNTIME	
	Ort::Env _OrtEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov8-Seg");
	Ort::SessionOptions _OrtSessionOptions = Ort::SessionOptions();
	Ort::Session* _OrtSession = nullptr;
	Ort::MemoryInfo _OrtMemoryInfo;
	std::shared_ptr<char> _inputName, _output_name0, _output_name1;
	std::vector<char*> _inputNodeNames; 
	std::vector<char*> _outputNodeNames;
	size_t _inputNodesNum = 0;        
	size_t _outputNodesNum = 0;      
	ONNXTensorElementDataType _inputNodeDataType; 
	ONNXTensorElementDataType _outputNodeDataType;
	std::vector<int64_t> _inputTensorShape; 
	std::vector<int64_t> _outputTensorShape;
	std::vector<int64_t> _outputMaskTensorShape;
};