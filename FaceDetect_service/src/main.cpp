#include <iostream>
#include <string>
#include <vector>
#include <mutex>
#include <cmath>
#include <cstdlib>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <httplib.h>
#include <nlohmann/json.hpp>
#include "base64.hpp"

using json = nlohmann::json;

// Rounding helper
inline float round_1(float val) {
    return std::round(val * 10.0f) / 10.0f;
}

inline float round_3(float val) {
    return std::round(val * 1000.0f) / 1000.0f;
}

// Global detector and mutex for thread safety
class FaceDetectorService {
public:
    FaceDetectorService(const std::string& model_path,
                        float score_threshold = 0.6f,
                        float nms_threshold = 0.3f,
                        int top_k = 5000)
        : default_score_threshold_(score_threshold),
          default_nms_threshold_(nms_threshold),
          top_k_(top_k) {
        
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("Model file not found: " + model_path);
        }

        std::cout << "[INFO] Loading YuNet model from: " << model_path << std::endl;
        detector_ = cv::FaceDetectorYN::create(
            model_path,
            "",
            cv::Size(640, 640),
            default_score_threshold_,
            default_nms_threshold_,
            top_k_,
            cv::dnn::DNN_BACKEND_OPENCV,
            cv::dnn::DNN_TARGET_CPU
        );

        if (detector_.empty()) {
            throw std::runtime_error("Failed to initialize FaceDetectorYN!");
        }
        std::cout << "[INFO] YuNet model loaded successfully." << std::endl;
    }

    json detect(const cv::Mat& image, float score_threshold = -1.0f, float nms_threshold = -1.0f) {
        if (image.empty()) {
            return {
                {"success", false},
                {"error", "Empty image"}
            };
        }

        int img_w = image.cols;
        int img_h = image.rows;

        // This model was validated for a fixed 640x640 NCHW input.
        // Resize before detection to avoid incompatible OpenCV DNN tensor shapes.
        constexpr int input_width = 640;
        constexpr int input_height = 640;
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(input_width, input_height));
        const float scale_x = static_cast<float>(img_w) / input_width;
        const float scale_y = static_cast<float>(img_h) / input_height;

        cv::Mat faces;
        {
            std::lock_guard<std::mutex> lock(mutex_);

            // Set dynamic thresholds if requested
            float score = (score_threshold > 0.0f && score_threshold <= 1.0f) ? score_threshold : default_score_threshold_;
            float nms = (nms_threshold > 0.0f && nms_threshold <= 1.0f) ? nms_threshold : default_nms_threshold_;
            
            detector_->setScoreThreshold(score);
            detector_->setNMSThreshold(nms);
            detector_->setInputSize(cv::Size(input_width, input_height));

            detector_->detect(resized, faces);
        }

        // Check if faces were found
        int count = faces.empty() ? 0 : faces.rows;
        if (count == 0) {
            return {
                {"success", false},
                {"count", 0},
                {"image_width", img_w},
                {"image_height", img_h},
                {"faces", json::array()}
            };
        }

        // Restore every coordinate (bbox and landmarks) to the original image.
        for (int i = 0; i < count; ++i) {
            for (int col = 0; col < 14; col += 2) {
                faces.at<float>(i, col) *= scale_x;
                faces.at<float>(i, col + 1) *= scale_y;
            }
        }

        json face_array = json::array();
        for (int i = 0; i < count; ++i) {
            float x = round_1(faces.at<float>(i, 0));
            float y = round_1(faces.at<float>(i, 1));
            float w = round_1(faces.at<float>(i, 2));
            float h = round_1(faces.at<float>(i, 3));

            float right_eye_x = round_1(faces.at<float>(i, 4));
            float right_eye_y = round_1(faces.at<float>(i, 5));
            float left_eye_x  = round_1(faces.at<float>(i, 6));
            float left_eye_y  = round_1(faces.at<float>(i, 7));
            float nose_x      = round_1(faces.at<float>(i, 8));
            float nose_y      = round_1(faces.at<float>(i, 9));
            float mouth_r_x   = round_1(faces.at<float>(i, 10));
            float mouth_r_y   = round_1(faces.at<float>(i, 11));
            float mouth_l_x   = round_1(faces.at<float>(i, 12));
            float mouth_l_y   = round_1(faces.at<float>(i, 13));

            float conf = round_3(faces.at<float>(i, 14));

            json face_item = {
                {"confidence", conf},
                {"bbox", {
                    {"x", x},
                    {"y", y},
                    {"width", w},
                    {"height", h}
                }},
                {"landmarks", {
                    {right_eye_x, right_eye_y},
                    {left_eye_x,  left_eye_y},
                    {nose_x,      nose_y},
                    {mouth_r_x,   mouth_r_y},
                    {mouth_l_x,   mouth_l_y}
                }}
            };
            face_array.push_back(face_item);
        }

        return {
            {"success", true},
            {"count", count},
            {"image_width", img_w},
            {"image_height", img_h},
            {"faces", face_array}
        };
    }

private:
    cv::Ptr<cv::FaceDetectorYN> detector_;
    std::mutex mutex_;
    float default_score_threshold_;
    float default_nms_threshold_;
    int top_k_;
};

int main(int argc, char* argv[]) {
    // Giới hạn OpenCV sử dụng 1 thread
    cv::setNumThreads(1);
    std::cout << "[INFO] OpenCV Threads: "
              << cv::getNumThreads() << std::endl;
    // Read environment variables or command-line parameters
    const char* env_model = std::getenv("MODEL_PATH");
    std::string model_path = (env_model && *env_model) ? env_model : "models/face_detection_yunet_2023mar.onnx";
    if (argc > 1) {
        model_path = argv[1];
    }

    const char* env_host = std::getenv("HOST");
    std::string host = (env_host && *env_host) ? env_host : "0.0.0.0";

    const char* env_port = std::getenv("PORT");
    int port = (env_port && *env_port) ? std::atoi(env_port) : 8080;
    if (argc > 2) {
        port = std::atoi(argv[2]);
    }

    const char* env_score = std::getenv("SCORE_THRESHOLD");
    float default_score = (env_score && *env_score) ? std::atof(env_score) : 0.6f;
    const char* env_nms = std::getenv("NMS_THRESHOLD");
    float default_nms = (env_nms && *env_nms) ? std::atof(env_nms) : 0.3f;

    std::cout << "====================================================\n";
    std::cout << "         YuNet Face Detection Service (C++)         \n";
    std::cout << "====================================================\n";
    std::cout << "Host:            " << host << "\n";
    std::cout << "Port:            " << port << "\n";
    std::cout << "Model:           " << model_path << "\n";
    std::cout << "Score Threshold: " << default_score << "\n";
    std::cout << "NMS Threshold:   " << default_nms << "\n";
    std::cout << "====================================================\n";

    std::unique_ptr<FaceDetectorService> service;
    try {
        service = std::make_unique<FaceDetectorService>(model_path, default_score, default_nms);
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Initialization error: " << e.what() << std::endl;
        return 1;
    }

    httplib::Server svr;

    // CORS & Common Headers
    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With"}
    });

    // Handle OPTIONS for browser CORS preflight
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    // Health check endpoint
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        json j = {
            {"status", "ok"},
            {"service", "yunet-face-detection"},
            {"engine", "C++ OpenCV FaceDetectorYN"}
        };
        res.set_content(j.dump(2), "application/json");
    });

    // Welcome & API info
    svr.Get("/", [port](const httplib::Request&, httplib::Response& res) {
        json j = {
            {"service", "YuNet High-Performance C++ Face Detection API"},
            {"endpoints", {
                {
                    {"path", "POST /detect"},
                    {"description", "Detect faces in an image"},
                    {"formats_supported", {
                        "multipart/form-data (file or image param)",
                        "application/json ({\"image\": \"<base64>\"})",
                        "raw binary (image/jpeg, image/png, application/octet-stream)"
                    }},
                    {"query_parameters", {
                        {"score_threshold", "float (optional, default: 0.6)"},
                        {"nms_threshold", "float (optional, default: 0.3)"}
                    }}
                },
                {
                    {"path", "GET /health"},
                    {"description", "Service health check"}
                }
            }}
        };
        res.set_content(j.dump(2), "application/json");
    });

    // Main Face Detection Endpoint
    svr.Post("/detect", [&service](const httplib::Request& req, httplib::Response& res) {
        try {
            // Optional query params
            float score_thresh = -1.0f;
            float nms_thresh = -1.0f;
            if (req.has_param("score_threshold")) {
                score_thresh = std::stof(req.get_param_value("score_threshold"));
            }
            if (req.has_param("nms_threshold")) {
                nms_thresh = std::stof(req.get_param_value("nms_threshold"));
            }

            cv::Mat image;

            // Method 1: Multipart form-data
            if (req.has_file("file") || req.has_file("image")) {

                httplib::MultipartFormData file;

                if (req.has_file("file")) {
                    file = req.get_file_value("file");
                } else {
                    file = req.get_file_value("image");
                }

                std::vector<uchar> buf(
                    file.content.begin(),
                    file.content.end()
                );

                image = cv::imdecode(buf, cv::IMREAD_COLOR);
            }
            // Method 2: JSON payload with Base64 image
            else if (req.get_header_value("Content-Type").find("application/json") != std::string::npos) {
                json body_json = json::parse(req.body, nullptr, false);
                if (body_json.is_discarded() || !body_json.contains("image")) {
                    json err = {
                        {"success", false},
                        {"error", "JSON payload must contain an 'image' field with base64 string"}
                    };
                    res.status = 400;
                    res.set_content(err.dump(2), "application/json");
                    return;
                }

                std::string base64_str = body_json["image"].get<std::string>();
                std::vector<uint8_t> decoded_bytes = base64::decode(base64_str);
                if (decoded_bytes.empty()) {
                    json err = {
                        {"success", false},
                        {"error", "Failed to decode base64 image data"}
                    };
                    res.status = 400;
                    res.set_content(err.dump(2), "application/json");
                    return;
                }

                std::vector<uchar> buf(decoded_bytes.begin(), decoded_bytes.end());
                image = cv::imdecode(buf, cv::IMREAD_COLOR);
            }
            // Method 3: Raw image binary body
            else if (!req.body.empty()) {
                std::vector<uchar> buf(req.body.begin(), req.body.end());
                image = cv::imdecode(buf, cv::IMREAD_COLOR);
            }

            // Validate decoded image
            if (image.empty()) {
                json err = {
                    {"success", false},
                    {"error", "Unable to decode image. Please ensure valid JPEG/PNG/WebP/BMP bytes or base64 format."}
                };
                res.status = 400;
                res.set_content(err.dump(2), "application/json");
                return;
            }

            // Run detection
            json result = service->detect(image, score_thresh, nms_thresh);
            res.status = 200;
            res.set_content(result.dump(2), "application/json");

        } catch (const std::exception& e) {
            json err = {
                {"success", false},
                {"error", std::string("Internal server error: ") + e.what()}
            };
            res.status = 500;
            res.set_content(err.dump(2), "application/json");
        }
    });

    std::cout << "[INFO] Server started and listening on http://" << host << ":" << port << "\n";
    std::cout << "[INFO] Ready to receive HTTP requests from clients!\n";

    if (!svr.listen(host.c_str(), port)) {
        std::cerr << "[ERROR] Failed to start HTTP server on " << host << ":" << port << std::endl;
        return 1;
    }

    return 0;
}

