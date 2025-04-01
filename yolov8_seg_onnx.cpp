#include "yolov8_seg_onnx.h"
using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Ort;

bool Yolov8SegOnnx::CheckParams(int netHeight, int netWidth, const int* netStride, int strideSize) {
	if (netHeight % netStride[strideSize - 1] != 0 || netWidth % netStride[strideSize - 1] != 0)
	{
		cout << "Error:_netHeight and _netWidth must be multiple of max stride " << netStride[strideSize - 1] << "!" << endl;
		return false;
	}
	return true;
}
bool Yolov8SegOnnx::CheckModelPath(std::string modelPath) {
	if (0 != _access(modelPath.c_str(), 0)) {
		cout << "Model path does not exist,  please check " << modelPath << endl;
		return false;
	}
	else
		return true;

}
void Yolov8SegOnnx::LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
	bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
{
	if (false) {
		int maxLen = MAX(image.rows, image.cols);
		outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
		image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
		params[0] = 1;
		params[1] = 1;
		params[3] = 0;
		params[2] = 0;
	}

	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height,
		(float)newShape.width / (float)shape.width);
	if (!scaleUp)
		r = std::min(r, 1.0f);

	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - new_un_pad[0]);
	auto dh = (float)(newShape.height - new_un_pad[1]);

	if (autoShape)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		new_un_pad[0] = newShape.width;
		new_un_pad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
	{
		cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
	}
	else {
		outImage = image.clone();
	}

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}


bool Yolov8SegOnnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaID, bool warmUp) {
	if (_batchSize < 1) _batchSize = 1;
	try
	{
		if (!CheckModelPath(modelPath)) {
			cout << "Path Error" << endl;
			return false;
		}
			
		std::vector<std::string> available_providers = GetAvailableProviders();
		auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");

		if (isCuda && (cuda_available == available_providers.end()))
		{
			std::cout << "Your ORT build without GPU. Change to CPU." << std::endl;
			std::cout << "************* Infer model on CPU! *************" << std::endl;
		}
		else if (isCuda && (cuda_available != available_providers.end()))
		{
			std::cout << "************* Infer model on GPU! *************" << std::endl;
			//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(_OrtSessionOptions, cudaID);
		}
		else
		{
			std::cout << "************* Infer model on CPU! *************" << std::endl;
		}
		//
		_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

		std::wstring model_path(modelPath.begin(), modelPath.end());
		_OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);

		Ort::AllocatorWithDefaultOptions allocator;
		//init input
		_inputNodesNum = _OrtSession->GetInputCount();

		_inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
		_inputNodeNames.push_back(_inputName.get());

		Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
		auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
		_inputNodeDataType = input_tensor_info.GetElementType();
		_inputTensorShape = input_tensor_info.GetShape();

		if (_inputTensorShape[0] == -1)
		{
			_isDynamicShape = true;
			_inputTensorShape[0] = _batchSize;

		}
		if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
			_isDynamicShape = true;
			_inputTensorShape[2] = _netHeight;
			_inputTensorShape[3] = _netWidth;
		}
		//init output
		_outputNodesNum = _OrtSession->GetOutputCount();
		if (_outputNodesNum != 2) {
			cout << "This model has " << _outputNodesNum << "output, which is not a segmentation model.Please check your model name or path!" << endl;
			return false;
		}

		_output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
		_output_name1 = std::move(_OrtSession->GetOutputNameAllocated(1, allocator));

		Ort::TypeInfo type_info_output0(nullptr);
		Ort::TypeInfo type_info_output1(nullptr);
		bool flag = false;

		flag = strcmp(_output_name0.get(), _output_name1.get()) < 0;
		if (flag)  //make sure "output0" is in front of  "output1"
		{
			type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0
			type_info_output1 = _OrtSession->GetOutputTypeInfo(1);  //output1

			_outputNodeNames.push_back(_output_name0.get());
			_outputNodeNames.push_back(_output_name1.get());

		}
		else {
			type_info_output0 = _OrtSession->GetOutputTypeInfo(1);  //output0
			type_info_output1 = _OrtSession->GetOutputTypeInfo(0);  //output1

			_outputNodeNames.push_back(_output_name1.get());
			_outputNodeNames.push_back(_output_name0.get());

		}

		auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
		_outputNodeDataType = tensor_info_output0.GetElementType();
		_outputTensorShape = tensor_info_output0.GetShape();
		auto tensor_info_output1 = type_info_output1.GetTensorTypeAndShapeInfo();

		if (isCuda && warmUp) {
			//draw run
			cout << "Start warming up" << endl;
			size_t input_tensor_length = VectorProduct(_inputTensorShape);
			float* temp = new float[input_tensor_length];
			std::vector<Ort::Value> input_tensors;
			std::vector<Ort::Value> output_tensors;
			input_tensors.push_back(Ort::Value::CreateTensor<float>(
				_OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
				_inputTensorShape.size()));
			for (int i = 0; i < 3; ++i) {
				output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
					_inputNodeNames.data(),
					input_tensors.data(),
					_inputNodeNames.size(),
					_outputNodeNames.data(),
					_outputNodeNames.size());
			}

			delete[]temp;
			
		}
	}
	catch (const std::exception&) {
		return false;
	}
	return true;
}

int Yolov8SegOnnx::Preprocessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params) {
	outSrcImgs.clear();
	Size input_size = Size(_netWidth, _netHeight);
	for (int i = 0; i < srcImgs.size(); ++i) {
		Mat temp_img = srcImgs[i];
		Vec4d temp_param = { 1,1,0,0 };
		if (temp_img.size() != input_size) {
			Mat borderImg;
			LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
			//cout << borderImg.size() << endl;
			outSrcImgs.push_back(borderImg);
			params.push_back(temp_param);
		}
		else {
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}

	int lack_num = srcImgs.size() % _batchSize;
	if (lack_num != 0) {
		for (int i = 0; i < lack_num; ++i) {
			Mat temp_img = Mat::zeros(input_size, CV_8UC3);
			Vec4d temp_param = { 1,1,0,0 };
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}
	return 0;

}
bool Yolov8SegOnnx::OnnxDetect(cv::Mat& srcImg, std::vector<OutputSeg>& output) {
	std::vector<cv::Mat> input_data = { srcImg };
	std::vector<std::vector<OutputSeg>> tenp_output;
	if (OnnxBatchDetect(input_data, tenp_output)) {
		output = tenp_output[0];
		return true;
	}
	else return false;
}
bool Yolov8SegOnnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputSeg>>& output) {
	vector<Vec4d> params;
	vector<Mat> input_images;
	cv::Size input_size(_netWidth, _netHeight);
	//对输入图片预处理
	Preprocessing(srcImgs, input_images, params);
	cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0, 0, 0), true, false);

	int64_t input_tensor_length = VectorProduct(_inputTensorShape);
	std::vector<Ort::Value> input_tensors;
	std::vector<Ort::Value> output_tensors;
	input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));

	output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
		_inputNodeNames.data(),
		input_tensors.data(),
		_inputNodeNames.size(),
		_outputNodeNames.data(),
		_outputNodeNames.size()
	);

	//对模型输出结果进行后处理
	float* all_data = output_tensors[0].GetTensorMutableData<float>();
	_outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	_outputMaskTensorShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
	vector<int> mask_protos_shape = { 1,(int)_outputMaskTensorShape[1],(int)_outputMaskTensorShape[2],(int)_outputMaskTensorShape[3] };
	int mask_protos_length = VectorProduct(mask_protos_shape);
	int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];
	int net_width = (int)_outputTensorShape[1];
	cout << "all_data: " << dec << all_data << " _outputTensorShape: " << _outputTensorShape[1] << "," << _outputTensorShape[2] << " mask_protos_length: " << mask_protos_length << " one_output_length: " << one_output_length << endl;

	for (int img_index = 0; img_index < srcImgs.size(); ++img_index) {
		Mat output0 = Mat(Size((int)_outputTensorShape[2], (int)_outputTensorShape[1]), CV_32F, all_data).t();  //[bs,116,8400]=>[bs,8400,116]
		all_data += one_output_length;
		float* pdata = (float*)output0.data;
		int rows = output0.rows;
		std::vector<int> class_ids;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;
		std::vector<vector<float>> picked_proposals;  //output0[:,:, 5 + _className.size():net_width]===> for mask
		for (int r = 0; r < rows; ++r) {    //stride
			cv::Mat scores(1, _className.size(), CV_32F, pdata + 4);
			//cout << "params: " << params[0] << " pdata: " << pdata[0] << "," << pdata[1] << "," << pdata[2] << "," << pdata[3] << " net_width: " << net_width << endl;
			Point classIdPoint;
			double max_class_socre;
			minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
			max_class_socre = (float)max_class_socre;
			if (max_class_socre >= _classThreshold) {
				vector<float> temp_proto(pdata + 4 + _className.size(), pdata + net_width);
				picked_proposals.push_back(temp_proto);
				//rect [x,y,w,h]
				float x = (pdata[0] - params[img_index][2]) / params[img_index][0];  //x
				float y = (pdata[1] - params[img_index][3]) / params[img_index][1];  //y
				float w = pdata[2] / params[img_index][0];  //w
				float h = pdata[3] / params[img_index][1];  //h
				//cout << "----params: " << params[0] << " pdata: " << pdata[0] << "," << pdata[1] << "," << pdata[2] << "," << pdata[3] << " net_width: " << net_width << endl;
				int left = MAX(int(x - 0.5 * w + 0.5), 0);
				int top = MAX(int(y - 0.5 * h + 0.5), 0);
				class_ids.push_back(classIdPoint.x);
				confidences.push_back(max_class_socre);
				boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
			}
			pdata += net_width;
		}

		vector<int> nms_result;
		cv::dnn::NMSBoxes(boxes, confidences, _classThreshold, _nmsThreshold, nms_result);
		std::vector<vector<float>> temp_mask_proposals;
		cv::Rect holeImgRect(0, 0, srcImgs[img_index].cols, srcImgs[img_index].rows);
		std::vector<OutputSeg> temp_output;
		for (int i = 0; i < nms_result.size(); ++i) {
			int idx = nms_result[i];
			OutputSeg result;
			result.id = class_ids[idx];
			result.confidence = confidences[idx];
			result.box = boxes[idx] & holeImgRect;
			temp_mask_proposals.push_back(picked_proposals[idx]);
			temp_output.push_back(result);
		}

		MaskParams mask_params;
		mask_params.params = params[img_index];
		mask_params.srcImgShape = srcImgs[img_index].size();
		mask_params.netHeight = _netHeight;
		mask_params.netWidth = _netWidth;
		mask_params.maskThreshold = _maskThreshold;
		Mat mask_protos = Mat(mask_protos_shape, CV_32F, output_tensors[1].GetTensorMutableData<float>() + img_index * mask_protos_length);
		for (int i = 0; i < temp_mask_proposals.size(); ++i) {
			GetMask2(Mat(temp_mask_proposals[i]).t(), mask_protos, temp_output[i], mask_params);
		}
		output.push_back(temp_output);
	}

	if (output.size())
		return true;
	else
		return false;
}

void Yolov8SegOnnx::GetMask2(const Mat& maskProposals, const Mat& maskProtos, OutputSeg& output, const MaskParams& maskParams) {
	int net_width = maskParams.netWidth;
	int net_height = maskParams.netHeight;
	int seg_channels = maskProtos.size[1];
	int seg_height = maskProtos.size[2];
	int seg_width = maskProtos.size[3];
	float mask_threshold = maskParams.maskThreshold;
	Vec4f params = maskParams.params;
	Size src_img_shape = maskParams.srcImgShape;

	Rect temp_rect = output.box;
	//crop from mask_protos
	int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
	int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
	int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
	int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

	//如果下面的 mask_protos(roi_rangs).clone()位置报错，说明你的output.box数据不对，或者矩形框就1个像素的，开启下面的注释部分防止报错。
	rang_w = MAX(rang_w, 1);
	rang_h = MAX(rang_h, 1);
	if (rang_x + rang_w > seg_width) {
		if (seg_width - rang_x > 0)
			rang_w = seg_width - rang_x;
		else
			rang_x -= 1;
	}
	if (rang_y + rang_h > seg_height) {
		if (seg_height - rang_y > 0)
			rang_h = seg_height - rang_y;
		else
			rang_y -= 1;
	}

	vector<Range> roi_rangs;
	roi_rangs.push_back(Range(0, 1));
	roi_rangs.push_back(Range::all());
	roi_rangs.push_back(Range(rang_y, rang_h + rang_y));
	roi_rangs.push_back(Range(rang_x, rang_w + rang_x));

	//crop
	Mat temp_mask_protos = maskProtos(roi_rangs).clone();
	Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
	Mat matmul_res = (maskProposals * protos).t();
	Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
	Mat dest, mask;

	//sigmoid
	cv::exp(-masks_feature, dest);
	dest = 1.0 / (1.0 + dest);

	int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
	int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
	int width = ceil(net_width / seg_width * rang_w / params[0]);
	int height = ceil(net_height / seg_height * rang_h / params[1]);

	resize(dest, mask, Size(width, height), INTER_NEAREST);
	Rect mask_rect = temp_rect - Point(left, top);
	mask_rect &= Rect(0, 0, width, height);
	mask = mask(mask_rect) > mask_threshold;
	output.boxMask = mask;

}

void Yolov8SegOnnx::DrawPred(Mat& img, Mat* outImage, vector<OutputSeg> result, std::vector<std::string> classNames) {
	Mat img_black;
	img_black = cv::Mat::zeros(img.size(), img.type());
	//Mat mask = img_black.clone();
	Mat mask = img.clone();//将原图clone过来

	// 设置随机种子（只需在程序初始化时执行一次）
	static bool seed_set = false;
	if (!seed_set) {
		srand(time(0)); // 确保每次运行颜色不同[1](@ref)
		seed_set = true;
	}

	for (int i = 0; i < result.size(); i++) {
		if (result[i].boxMask.rows && result[i].boxMask.cols > 0) {
			Scalar random_color(rand() % 256, rand() % 256, rand() % 256); // [1,4,6](@ref)
			mask(result[i].box).setTo(random_color, result[i].boxMask);//将推理得到的结果（boxMask）在mask图像（原图）中的特定位置显示出来
		}
			
	}
	addWeighted(img, 0.5, mask, 0.5, 0, img); //add mask to src

	*outImage = mask;
	//destroyAllWindows();

}

double Yolov8SegOnnx::computeIoU(const cv::Mat& mask1, const cv::Mat& mask2) {
	cv::Mat intersection, unionMat;
	// 求交集：两个 mask 都为1的位置
	cv::bitwise_and(mask1, mask2, intersection);
	// 求并集：两个 mask 中有1的位置
	cv::bitwise_or(mask1, mask2, unionMat);

	double inter = static_cast<double>(cv::countNonZero(intersection));
	double uni = static_cast<double>(cv::countNonZero(unionMat));
	return uni > 0 ? inter / uni : 0.0;
}

int Yolov8SegOnnx::findParent(int i, std::vector<int>& parent) {
	if (parent[i] != i) {
		parent[i] = findParent(parent[i], parent);
	}
	return parent[i];
}

void Yolov8SegOnnx::unionSets(int i, int j, std::vector<int>& parent) {
	int rootI = findParent(i, parent);
	int rootJ = findParent(j, parent);
	if (rootI != rootJ) {
		parent[rootJ] = rootI;
	}
}

int Yolov8SegOnnx::reComputeInstances(std::vector<cv::Mat>& masks, float iouThreshold) {
	int n = masks.size();
	if (n == 0) return 0;

	// 初始化并查集
	std::vector<int> parent(n);
	for (int i = 0; i < n; i++) {
		parent[i] = i;
	}

	// 遍历 mask 对，每对只计算一次
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			double iou = computeIoU(masks[i], masks[j]);
			if (iou > iouThreshold) {
				unionSets(i, j, parent);
			}
		}
	}

	// 统计不同的父节点，即代表不同的实例
	std::set<int> clusters;
	for (int i = 0; i < n; i++) {
		clusters.insert(findParent(i, parent));
	}
	return clusters.size();
}

int Yolov8SegOnnx::compMaskAndInstances(std::vector<OutputSeg>& result, float iouThreshold) {
	std::vector<cv::Mat> masks;

	for (int i = 0; i < result.size(); i++) {
		if (result[i].boxMask.rows > 0 && result[i].boxMask.cols > 0) {
			cv::Mat img_black = cv::Mat::zeros(1080, 1920, CV_8UC1);
			// 将 mask 按照对应的 box 放到原图尺寸中
			img_black(result[i].box).setTo(255, result[i].boxMask);
			masks.push_back(img_black);
		}
	}
	int count = reComputeInstances(masks, iouThreshold);
	return count;
}
