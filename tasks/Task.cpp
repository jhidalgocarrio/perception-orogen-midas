/* Generated from orogen/lib/orogen/templates/tasks/Task.cpp */

#include "Task.hpp"

#include <frame_helper/FrameHelper.h>
#include <opencv2/imgproc.hpp>

#include <sys/stat.h>
#include <chrono>

#define DEBUG_PRINTS 1

using namespace midas;

/** From here: https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-14-17-c
 * **/
inline bool file_exist (const std::string& name)
{
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0); 
}

Task::Task(std::string const& name)
    : TaskBase(name)
{
}

Task::~Task()
{
}

void Task::frameCallback(const base::Time &ts, const ::RTT::extras::ReadOnlyPointer< ::base::samples::frame::Frame > &frame_sample)
{
    #ifdef DEBUG_PRINTS
    std::cout<<"** [MIDAS_TASK FRAME] Received Frame at ["<<frame_sample->time.toString()<<"] **"<<std::endl;
    #endif

    /** Convert the image to OpenCV Mat and resize **/
    cv::Mat img = frame_helper::FrameHelper::convertToCvMat(*frame_sample);
    cv::resize(img, img, cv::Size(512, 384), CV_INTER_CUBIC);

    /** Image to Tensor **/
    auto tensor_image = torch::from_blob(img.data, {img.rows, img.cols, img.channels()}, at::kByte);
    tensor_image = tensor_image.permute({ 2,0,1 }); //[C x H x W]
    tensor_image.unsqueeze_(0); //[B x C x H x W]
    tensor_image = tensor_image.toType(c10::kFloat);//.sub(127.5).mul(0.0078125);
    tensor_image.to(c10::DeviceType::CPU);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image);

    /**  Execute the model and turn its output into a tensor. **/
    auto start = std::chrono::steady_clock::now();
    at::Tensor output = this->module.forward(inputs).toTensor();
    auto end = std::chrono::steady_clock::now();

    /** Convert prediction to image and resize to the original size **/
    cv::Mat prediction = this->DepthToCvImage(output.squeeze());
    cv::resize(prediction, prediction, cv::Size(frame_sample->size.width, frame_sample->size.height), CV_INTER_CUBIC);

    /** Write the ouput port **/
    this->outputDepthMap(prediction, frame_sample->time);
}

/// The following lines are template definitions for the various state machine
// hooks defined by Orocos::RTT. See Task.hpp for more detailed
// documentation about them.

bool Task::configureHook()
{
    if (! TaskBase::configureHook())
        return false;

    /** Read model path **/
    this->model_filename = _model_filename.value();
    if (!file_exist(this->model_filename))
    {
        RTT::log(RTT::Error) << "[ERROR]: Given Torch model does not exist: "<<this->model_filename<< RTT::endlog();
        return false;
    }

    /* Load the model **/
    at::init_num_threads();
    at::set_num_threads(4);

    try {
        RTT::log(RTT::Info)<<"Model to load: "<<this->model_filename;
        // Deserialize the ScriptModule from a file using torch::jit::load().
        this->module = torch::jit::load(this->model_filename);
    }
    catch (const c10::Error& e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    this->module.eval();
    RTT::log(RTT::Info) << "...[OK]"<< RTT::endlog();

    return true;
}
bool Task::startHook()
{
    if (! TaskBase::startHook())
        return false;
    return true;
}
void Task::updateHook()
{
    TaskBase::updateHook();
}
void Task::errorHook()
{
    TaskBase::errorHook();
}
void Task::stopHook()
{
    TaskBase::stopHook();
}
void Task::cleanupHook()
{
    TaskBase::cleanupHook();
}

cv::Mat Task::DepthToCvImage(at::Tensor tensor, const uint8_t &bits)
{
    auto depth_min = tensor.min();
    auto depth_max = tensor.max();
    int height = tensor.sizes()[0];
    int width = tensor.sizes()[1];

    std::cout<<"height: "<<height<<" width: "<<width<<std::endl;
    std::cout<<"min depth: "<<depth_min<<std::endl;
    std::cout<<"max depth: "<<depth_max<<std::endl;

    int max_val = std::pow(2, (8*bits))-1;
    at::Tensor out = max_val * (tensor - depth_min) / (depth_max - depth_min);
    out = out.to(torch::kUInt8);
    std::cout<<"out min depth: "<<out.min()<<std::endl;
    std::cout<<"out max depth: "<<out.max()<<std::endl;

    try
    {
        cv::Mat output_mat (cv::Size{ width, height }, CV_8UC1, out.data_ptr<uint8_t>());

        std::cout<<"out image size: "<<output_mat.size()<<std::endl;
        return output_mat.clone();
    }
    catch (const c10::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
    return cv::Mat(height, width, CV_8UC1);
}


void Task::outputDepthMap(cv::Mat &image, const ::base::Time &timestamp)
{
    RTT::extras::ReadOnlyPointer<base::samples::frame::Frame> depthmap;
    ::base::samples::frame::Frame *img = new ::base::samples::frame::Frame();
    depthmap.reset(img);

    ::base::samples::frame::Frame *depthmap_ptr = depthmap.write_access();
    depthmap_ptr->image.clear();
    frame_helper::FrameHelper::copyMatToFrame(image, *depthmap_ptr);
    depthmap.reset(depthmap_ptr);
    depthmap_ptr->time = timestamp;
    _depthmap.write(depthmap);
}