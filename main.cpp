#include "yolov8_seg_onnx.h"

using namespace cv;

int main() {
    Yolov8SegOnnx cengshu_model;
    cengshu_model.ReadModel("line1_cengshu.onnx");
    cengshu_model._classThreshold = 0.78;
    cengshu_model._nmsThreshold = 0.85;
    cengshu_model._maskThreshold = 0.7;

    std::string pattern2 = "C:/Users/Administrator/Desktop/行李实例分割/ultralytics-main/datasets/line1/images/*.jpg";
    std::vector<cv::String> filenames2;
    int num = 0;
    cv::glob(pattern2, filenames2, false);
    for (const auto& filename : filenames2) {
        cv::Mat img = cv::imread(filename);
        std::vector<OutputSeg> outPut;
        cengshu_model.OnnxDetect(img, outPut);
        Mat* outImg = new Mat(img.size(), CV_8UC3);
        cengshu_model.DrawPred(img, outImg, outPut, cengshu_model._className);
        imshow("out", *outImg);
        cv::waitKey(0);
        int num = 0;
        for (int i = 0; i < outPut.size(); i++) {
            std::cout << outPut[i].confidence << std::endl;
            std::cout << "掩码个数：" << outPut[i].boxMask.size() << std::endl;
            num++;
        }
        int test = cengshu_model.compMaskAndInstances(outPut, 0.5);
        std::cout << "合并后实例数：" << test << std::endl;
    }
	
	return 0;
}
