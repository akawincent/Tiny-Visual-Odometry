#include<iostream>
#include<opencv2/opencv.hpp>
#include<Eigen/Core>
#include<Eigen/Dense>
#include<string>
#include<chrono>

using namespace std;

string filename_1 = "../LK1.png";
string filename_2 = "../LK2.png";

/******************双线性插值求浮点像素坐标处的灰度值*******************/
//内联函数提高调用运行时间
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    // if (x < 0) x = 0;
    // if (y < 0) y = 0;
    // if (x >= img.cols - 1) x = img.cols - 2;
    // if (y >= img.rows - 1) y = img.rows - 2;
    
    //取浮点像素坐标的四个最邻近坐标
    int x_1 = (int)x;
    int y_1 = (int)y;
    int x_2 = (int)x + 1;
    int y_2 = (int)y + 1;

    // cout<<"————————————————————"<<endl;
    // cout<<"("<<y_1<<","<<x_1<<")"<<"     "<<"("<<y_1<<","<<x_2<<")"<<endl;
    // cout<<"("<<y_2<<","<<x_1<<")"<<"     "<<"("<<y_2<<","<<x_2<<")"<<endl;

    //取浮点像素坐标的小数部分
    float xx = x - floor(x);
    float yy = y - floor(y);
    
    //返回经过双线性插值计算后的浮点坐标处的灰度值
    return (1 - xx) * (1 - yy) * img.at<uchar>(cv::Point(x_1,y_1))
    + xx * (1 - yy) * img.at<uchar>(cv::Point(x_2,y_1))
    + (1 - xx) * yy * img.at<uchar>(cv::Point(x_1,y_2))
    + xx * yy * img.at<uchar>(cv::Point(x_2,y_2));
}

// inline float GetPixelValue(const cv::Mat &img, float x, float y) {
//     // boundary check
//     if (x < 0) x = 0;
//     if (y < 0) y = 0;
//     if (x >= img.cols - 1) x = img.cols - 2;
//     if (y >= img.rows - 1) y = img.rows - 2;
    
//     float xx = x - floor(x);
//     float yy = y - floor(y);
//     int x_a1 = std::min(img.cols - 1, int(x) + 1);
//     int y_a1 = std::min(img.rows - 1, int(y) + 1);
    
//     return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
//     + xx * (1 - yy) * img.at<uchar>(y, x_a1)
//     + (1 - xx) * yy * img.at<uchar>(y_a1, x)
//     + xx * yy * img.at<uchar>(y_a1, x_a1);
// }

/******************判断像素位置是否合理*******************/
inline bool PixelPositionRight(const cv::Mat &img,const cv::Point2f pixel,int x,int y){
    auto u = pixel.x + x;
    auto v = pixel.y + y;
    auto w = img.cols;
    auto h = img.rows;
    if(u >= 1 && u <= w - 2 && v >= 1 && v <= h - 2)
        return true;
    else return false;
}

/**********************构建光流追踪器类*************************/
class OpticalFlowTracker{
    public:
        OpticalFlowTracker(const cv::Mat &img_1,const cv::Mat &img_2,const vector<cv::KeyPoint> &kp1,bool inverse);
        ~OpticalFlowTracker();

        void CalcuSingleOpticalFlowByGN();

        cv::Mat GetTrackerImg1(){
            return this->OF_img_1;
        }

        cv::Mat GetTrackerImg2(){
            return this->OF_img_2;
        }

        vector<cv::KeyPoint> GetTrackerKeyponit1(){
            return this->OF_kp1;
        }

        vector<cv::KeyPoint> GetTrackerKeyponit2(){
            return this->OF_kp2;
        }

        vector<bool> GetTrackerSuccess(){
            return this->success;
        }
        
    private:
        cv::Mat OF_img_1;               //追踪器的第一张图片
        cv::Mat OF_img_2;               //追踪器的第二张图片
        vector<cv::KeyPoint> OF_kp1;    //第一张图片提取出来的关键点
        vector<cv::KeyPoint> OF_kp2;    //第二张等待估计出来的关键点
        vector<bool> success;           //记录每个关键点追踪情况的好坏
        bool inverse;                   //是否启用反向光流法
};

/**************光流追踪器构造函数***************/
OpticalFlowTracker::OpticalFlowTracker(const cv::Mat &img_1,const cv::Mat &img_2,const vector<cv::KeyPoint> &kp1,bool inverse){
        this->OF_img_1 = img_1;
        this->OF_img_2 = img_2;
        this->OF_kp1 = kp1;
        this->inverse = inverse;
        OF_kp2.resize(kp1.size());
        success.resize(kp1.size());
}

/**************光流追踪器析构函数***************/
OpticalFlowTracker::~OpticalFlowTracker(){
}

/**************利用高斯牛顿法进行单层光流追踪***************/
void OpticalFlowTracker::CalcuSingleOpticalFlowByGN(){
    //定义像素邻域大小（要让邻域也参与到优化中）
    int half_patch_size = 8;        //9*9的像素领域
    //优化迭代此次数
    int iter_num = 20;

    //遍历优化每个关键点的dx dy
    for(int i = 0; i < OF_kp1.size(); i++){
        auto kp = OF_kp1[i];        //图1中的关键点
        double dx,dy = 0;           //等待优化估计求解的关键点位移
        double cost,lastcost = 0;   //需最小化的残差      
        bool succ = 0;              //记录迭代求解的好坏结果

        //创建高斯牛顿法增量方程所需的矩阵
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        Eigen::Vector2d J = Eigen::Vector2d::Zero();
        
        for(int iterator = 0; iterator < iter_num; iterator++){
            if(inverse == false){                   //正向光流法要每次迭代时重新计算雅可比矩阵J
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
                J = Eigen::Vector2d::Zero();
            }
            else{                                   //反向光流法的雅可比矩阵在第一次迭代中就被确定下来了 因此不需要重置
                b = Eigen::Vector2d::Zero();
            }

            //每次迭代要重新计算残差函数
            cost = 0;

            //让像素邻域参与优化实际上是假定了同一个区域内的点都具有相同的运动
            for(int x = -half_patch_size; x < half_patch_size; x++){
                for(int y = -half_patch_size; y < half_patch_size; y++){
                    //判断像素坐标是否在合理范围 是则参与优化
                    if(PixelPositionRight(OF_img_1,kp.pt,x,y) == true && 
                        PixelPositionRight(OF_img_2,kp.pt,x + dx,y + dy) == true){
                            //计算误差
                            double error = GetPixelValue(OF_img_1,kp.pt.x + x,kp.pt.y + y) - 
                                           GetPixelValue(OF_img_2,kp.pt.x + x + dx,kp.pt.y + y + dy);;
                            
                            //正向光流法时每次迭代需要重新计算雅可比矩阵
                            if(inverse == false){
                                J = -1.0 * Eigen::Vector2d(
                                    0.5 * (GetPixelValue(OF_img_2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                           GetPixelValue(OF_img_2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                                    0.5 * (GetPixelValue(OF_img_2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                           GetPixelValue(OF_img_2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1)));
                            }
                            //反向光流法时第一次迭代计算雅可比矩阵后就确定下来了
                            else{
                                if(iterator == 0){
                                    J = -1.0 * Eigen::Vector2d(
                                        0.5 * (GetPixelValue(OF_img_1, kp.pt.x + x + 1, kp.pt.y + y) -
                                               GetPixelValue(OF_img_1, kp.pt.x + x - 1, kp.pt.y + y)),
                                        0.5 * (GetPixelValue(OF_img_1, kp.pt.x + x, kp.pt.y + y + 1) -
                                               GetPixelValue(OF_img_1, kp.pt.x + x, kp.pt.y + y - 1)));
                                    cout<<"J矩阵"<<endl;
                                    cout<<J(0,0)<<endl<<J(1,0)<<endl;                                   
                                }
                            }

                            //更新b
                            b += -error * J;            
                            //正向光流法或者在第一次迭代时才需要重新计算
                            if(inverse == false || iterator == 0)
                                H += J * J.transpose();
                            cost += error * error;
                    }
                    else continue;      //对于不满足像素合理位置条件的点 不参与优化
                }
            }

            //求解增量方程的更新值  update[0]是优化dx的更新值   update[1]是优化dy的更新值
            Eigen::Vector2d update = H.ldlt().solve(b);
            
            //当求解的更新值无解时，停止该点的迭代求解
            if(std::isnan(update[0]) || std::isnan(update[1])){
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            //更新需要优化的dx 和 dy
            dx += update[0];
            dy += update[1];
            //cout<<update[0]<<"  "<<update[1]<<endl;
            //更新残差
            lastcost = cost;
            //本次迭代求解成功
            succ = true;

            //当残差已经陷入极小值，停止该点的迭代求解
            if (iterator > 0 && cost > lastcost) {
                break;
            }

            //当更新值小于很小时，停止该点的迭代求解
            if(update.norm() < 1e-3) {
                break;
            }
        }

        //记录该点追踪结果为成功
        success[i] = succ;

        //求解出关键点运动dx和dy 从而接触关键点在图2中的坐标
        OF_kp2[i].pt = kp.pt + cv::Point2f(dx, dy);
    }
}

int main(int argc, char const *argv[])
{
    /* code */
    /*****************导入测试图片******************/
    cv::Mat test_img_1 = cv::imread(filename_1,0);
    cv::Mat test_img_2 = cv::imread(filename_2,0);

    /**************提取图1中的关键点****************/
    vector<cv::KeyPoint> kp1;   //存储图1的关键点
    cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 20); //检测最多500个关键点
    detector->detect(test_img_1, kp1);

    /*************实例化光流追踪器对象***************/
    bool if_inverse = false;        //不使用反向光流法
    OpticalFlowTracker tracker(test_img_1,test_img_2,kp1,if_inverse);   //实例化对象
    //cv::imshow("光流追踪器中的第一个图像",tracker.GetTrackerImg1());
    //cv::imshow("光流追踪器中的第二个图像",tracker.GetTrackerImg2());

    /*************单层光流法追踪关键点***************/
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    tracker.CalcuSingleOpticalFlowByGN();
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Single optical flow by gauss-newton: " << time_used.count() << endl;

    /*************展示追踪结果***************/
    cv::Mat result_img;
    cv::cvtColor(test_img_2, result_img, CV_GRAY2BGR);
    for (int i = 0; i < tracker.GetTrackerKeyponit2().size(); i++) {
        if (tracker.GetTrackerSuccess()[i]) {
            cv::circle(result_img, tracker.GetTrackerKeyponit2()[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(result_img, tracker.GetTrackerKeyponit1()[i].pt, tracker.GetTrackerKeyponit2()[i].pt, cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("tracked by single level Optical Flow using Gauss-Newton", result_img);
    cv:cvWaitKey(0);
    return 0;
}
