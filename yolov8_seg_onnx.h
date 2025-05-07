#pragma once
#include <iostream>
#include<memory>
#include <opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
//#include <tensorrt_provider_factory.h> 
#include<iostream>
#include <numeric>
#include<opencv2/opencv.hpp>
#include<io.h>

// 模型推理输出
struct OutputSeg {
	int id;//类别id
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
	// 构造函数，初始化Ort内存环境
	Yolov8SegOnnx() :_OrtMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput)) {};
	~Yolov8SegOnnx() {};
	/** @brief 读取onnx模型， 设置cuda
	@param const std::string& modelPath - 即模型输入路径
	@param bool isCuda - 即是否使用cuda
	@param int cudaID - 即cuda的id，保持默认0即可
	@param bool warmUp - 即推理时显卡是否需要热身
	@return true or false - 即是否运行成功
	 */
	bool ReadModel(const std::string& modelPath, bool isCuda = false, int cudaID = 0, bool warmUp = false);
	/** @brief 读取待推理图片，输出模型推理结果
	@param cv::Mat& srcImg - 即待推理图片
	@param std::vector<OutputSeg>& output - 即模型推理结果，多个框存储在vector中
	@return - 无
	 */
	bool OnnxDetect(cv::Mat& srcImg, std::vector<OutputSeg>& output);
	/** @brief 读取待推理图片，输出模型推理结果
	@param cv::Mat& Img - 即待推理的原图
	@param cv::Mat* outImage - 即存储推理mask图
	@param std::vector<OutputSeg>& result - 即模型推理结果
	@param std::vector<std::string> classNames - 即传入类别名称，直接使用对象中的_className属性即可
	@return - 无
	 */
	void DrawPred(cv::Mat& img, cv::Mat* outImage, std::vector<OutputSeg> result, std::vector<std::string> classNames);//画图，显示
	/** @brief 此函数是为了提取前景在原图中的位置，如果没有前景分割模型步骤则无需调用此函数
	@param cv::Mat& Img - 即待推理的原图
	@param cv::Mat* outImage - 即存储推理得到的前景图片，可以手动再imwrite一下存储到本地
	@param std::vector<OutputSeg>& result - 即模型推理结果
	@param std::vector<std::string> classNames - 即传入类别名称，根据前景模型的类别名称自定义传入
	@return - 无
	 */
	void DrawPred2(cv::Mat& img, cv::Mat* outImage, std::vector<OutputSeg> result, std::vector<std::string> classNames);
	std::vector<std::string> _className = {"box"}; //类别名称,需要根据实际修改，这里是表示空框

	const int _netWidth = 640;   //模型输入图片的尺寸（宽），需要根据模型进行修改
	const int _netHeight = 640;  //模型输入图片的尺寸（高），需要根据模型进行修改
	int _batchSize = 1;  //批次设置，单张推理
	bool _isDynamicShape = false;//模型是否支持动态设置
	float _classThreshold = 0.25;  // 推理时指定的类别阈值
	float _nmsThreshold = 0.45;    // 推理时指定的NMS阈值，决定框的个数
	float _maskThreshold = 0.5;    // 推理时指定的mask阈值，决定mask的大小

	template <typename T>
	T VectorProduct(const std::vector<T>& v)
	{
		return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
	};
	// 调用OnnxDetect执行推理即可，此函数功能一致
	bool OnnxBatchDetect(std::vector<cv::Mat>& srcImg, std::vector<std::vector<OutputSeg>>& output);
	/** @brief 检查模型路径有无问题
	@param std::string modelPath - 即模型路径
	@return bool - 即路径是否有问题
	 */
	bool CheckModelPath(std::string modelPath);
	/** @brief 检查模型输入输出参数是否正确
	@param int netHeight - 即模型输入图片的尺寸（高）
        @param int netWidth - 即模型输入图片的尺寸（宽）
        @param const int* netStride - 即模型输入图片的步长
        @param int strideSize - 即模型输入图片的步长大小
	@return bool - 即成功或失败
	*/
	bool CheckParams(int netHeight, int netWidth, const int* netStride, int strideSize);
	/** @brief 图片前处理函数
	@param const std::vector<cv::Mat>& SrcImgs - 即待推理的原图
	@param std::vector<cv::Mat>& OutSrcImgs - 即处理后的图片
	@param std::vector<cv::Vec4d>& params - 即图片处理后保存的参数
	@return 无
	*/
	int Preprocessing(const std::vector<cv::Mat>& SrcImgs, std::vector<cv::Mat>& OutSrcImgs, std::vector<cv::Vec4d>& params);
	/** @brief 对图片进行裁剪归一化操作
	@param const cv::Mat& image - 即待处理的图片
        @param cv::Mat& outImage - 即处理后的图片
        @param cv::Vec4d& params - 即图片处理后保存的参数
        @param cv::Size newShape - 即模型的输入尺寸，需要根据模型进行修改
        @param bool autoShape - 即是否自动调整尺寸
        @param bool scaleFill - 即是否填充
        @param bool scaleUp - 即是否放大
        @param int stride - 即步长
        @param const cv::Scalar& color - 即填充颜色
        @return 无
	*/
	void LetterBox(const cv::Mat& image, cv::Mat& outImage,
		cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
		const cv::Size& newShape = cv::Size(640, 640),//模型的输入尺寸，需要根据模型进行修改
		bool autoShape = false,
		bool scaleFill = false,
		bool scaleUp = true,
		int stride = 32,
		const cv::Scalar& color = cv::Scalar(114, 114, 114));
        /** @brief 将检测框与原型掩码结合，生成实例级分割结果
	@param const cv::Mat& maskProposals - 即检测框对应的掩码系数向量
        @param const cv::Mat& maskProtos - 即模型输出的原型掩码
        @param OutputSeg& output - 即最终包含检测框的模型推理结果
        @param MaskParams& maskParams - 即掩码预处理参数
        @return 无
	*/	
	void GetMask2(const cv::Mat& maskProposals, const cv::Mat& maskProtos, OutputSeg& output, const MaskParams& maskParams);

	// 计算两个二值 mask 的 IoU，mask 中仅包含 0 和 255
	double computeIoU(const cv::Mat& mask1, const cv::Mat& mask2);
	// 并查集查找函数（带路径压缩）
	int findParent(int i, std::vector<int>& parent);
	// 并查集合并操作
	void unionSets(int i, int j, std::vector<int>& parent);
	// 遍历所有 mask，两两比较，如果 IoU 超过阈值，则将它们合并到同一组中，最后返回不同组的数量
	int reComputeInstances(std::vector<cv::Mat>& masks, float iouThreshold);
	
	/** @brief 将mask都转到图像坐标系，再进行比较，剔除误识别的掩码框
	@param std::vector<OutputSeg>& result - 即模型推理结果
        @param float iouThreshold - 即IoU阈值
	@return int - 即输出最终实例数量
	*/
	int compMaskAndInstances(std::vector<OutputSeg>& result, float iouThreshold);

	//ONNXRUNTIME 的一些相关变量预定义，无需了解	
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
