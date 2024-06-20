#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <string>


cv::Mat load_image(std::string image_path) {
    std::string path = cv::samples::findFile(image_path);
    cv::Mat image = cv::imread(path);
    return image;
}

torch::Tensor image_to_tensor(cv::Mat& image) {
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32FC3);
    torch::Tensor tensor = torch::from_blob(
        image.data,
        { image.rows, image.cols, image.channels() },
        torch::kFloat32);
    tensor = tensor.permute({ 2, 0, 1 });
    tensor = tensor.unsqueeze(0);
    return tensor;
}

torch::jit::Module load_model(std::string model_path) {
    torch::jit::Module model = torch::jit::load(model_path);
    model.eval();
    return model;
}

cv::Mat tensor_to_image(torch::Tensor tensor) {
    tensor = tensor.squeeze().detach();
    tensor = tensor.permute({ 1, 2, 0 });
    tensor = tensor.contiguous();
    tensor = tensor.clamp(0, 255).to(torch::kU8);
    tensor = tensor.to(torch::kCPU);
    int64_t height = tensor.size(0);
    int64_t width = tensor.size(1);

    cv::Mat mat(height, width, CV_8UC3, tensor.data_ptr());
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    return mat;
}


int main(int argc, char* argv[]) {
    std::string input_type(argv[1]);
    int resolution_width = atoi(argv[2]);
    int resolution_height = atoi(argv[3]);
    std::string model_name(argv[4]);
    std::string input_path(argv[5]);
    std::string output_path(argv[6]);

    // make width and height divisible by 4
    resolution_width = (resolution_width + 3) / 4 * 4;
    resolution_height = (resolution_height + 3) / 4 * 4;

    // loading model from file
    torch::jit::Module model = load_model("../models/traced_" + model_name + ".pt");

    if (input_type == "image") {
        // loading image from file
        cv::Mat input = load_image(input_path);
        cv::resize(input, input, cv::Size(resolution_width, resolution_height));

        // transforming image to input data for model
        torch::Tensor input_tensor = image_to_tensor(input);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        // running model
        torch::Tensor output_tensor = model.forward(inputs).toTensor();

        // transforming models output to image format
        cv::Mat output = tensor_to_image(output_tensor);

        // saving image to file
        cv::imwrite(output_path, output);
    }
    else if (input_type == "video") {
        cv::VideoCapture cap(input_path);
        int fps = cap.get(cv::CAP_PROP_FPS);
        cv::VideoWriter video(
            output_path,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            fps,
            cv::Size(resolution_width, resolution_height)
        );

        while (true) {
            cv::Mat frame;
            cap >> frame;

            if (frame.empty()) {
                break;
            }

            // resizing frame
            cv::resize(frame, frame, cv::Size(resolution_width, resolution_height));

            // transforming image to input data for model
            torch::Tensor input_tensor = image_to_tensor(frame);
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);

            // running model
            torch::Tensor output_tensor = model.forward(inputs).toTensor();

            // transforming models output to image format
            cv::Mat output = tensor_to_image(output_tensor);

            video.write(output);
        }

        cap.release();
        video.release();
    }
}
