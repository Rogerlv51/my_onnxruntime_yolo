#include "yolov8_seg_onnx.h"

using namespace std;
using namespace cv;
using namespace dnn;

Yolov8SegOnnx seg_test;

int main(){
    if(!=seg_test.ReadModel("best.onnx", false)){
        cout << "未成功加载模型" << endl;
        return 0;
    }
    Mat img = imread("test.jpg");
    vector<OutputSeg> output;
    seg_test.OnnxDetect(img, output);
    Mat* pred_out = new Mat(img.rows, img.cols, CV_8UC3);
    seg_test.DrawPred(img, pred_out, output, seg_test._className);
    imshow("pred_img", pred_out);
    waitKey(0);
    return 0;
}

