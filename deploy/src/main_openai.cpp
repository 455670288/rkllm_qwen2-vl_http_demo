#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>
#include <opencv2/opencv.hpp>

#include <nlohmann/json.hpp>
#include <httplib.h>
#include <curl/curl.h>

#include "image_enc.h"
#include "rkllm.h"

#define PROMPT_TEXT_PREFIX "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
#define PROMPT_TEXT_POSTFIX "<|im_end|>\n<|im_start|>assistant\n"

using namespace std;
using json = nlohmann::json;

LLMHandle llmHandle = nullptr;
rknn_app_context_t rknn_app_ctx;




// 图像预处理：填充为正方形 + 缩放
cv::Mat preprocess_image(const cv::Mat& img, const cv::Size& target_size) {
    cv::Mat square_img;
    int max_dim = max(img.cols, img.rows);
    cv::Scalar bg_color(127.5, 127.5, 127.5);
    square_img = cv::Mat(max_dim, max_dim, img.type(), bg_color);
    img.copyTo(square_img(cv::Rect((max_dim - img.cols)/2, (max_dim - img.rows)/2, img.cols, img.rows)));
    cv::resize(square_img, square_img, target_size);
    return square_img;
}

void callback(RKLLMResult *result, void *userdata, LLMCallState state){

    if (state == RKLLM_RUN_FINISH)
    {
        printf("\n");
    }
    else if (state == RKLLM_RUN_ERROR)
    {
        printf("\\run error\n");
    }
    else if (state == RKLLM_RUN_NORMAL)
    {
        printf("%s", result->text);
        std::string* reply = static_cast<std::string*>(userdata);
        *reply += result->text;        
    }
}


// 使用OpenSSL解码Base64
std::vector<unsigned char> base64_decode(const std::string& encoded_string) {
    BIO* bio = BIO_new_mem_buf(encoded_string.c_str(), encoded_string.size());
    BIO* b64 = BIO_new(BIO_f_base64());
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL); // 不处理换行符
    bio = BIO_push(b64, bio);

    std::vector<unsigned char> decoded(encoded_string.size()); // 分配足够空间
    int decoded_len = BIO_read(bio, decoded.data(), encoded_string.size());
    decoded.resize(decoded_len > 0 ? decoded_len : 0);

    BIO_free_all(bio);
    return decoded;
}




int main(int argc, char** argv) {
    if(argc < 4) {
        cerr << "Usage: " << argv[0] << " <encoder_model_path> <llm_model_path> <port>\n";
        return -1;
    }

    // 初始化RKLLM
    RKLLMParam llm_param = rkllm_createDefaultParam();
    llm_param.model_path = argv[2];
    llm_param.max_new_tokens = 512;
    llm_param.max_context_len = 4096;
    llm_param.skip_special_token = true;
    llm_param.img_start = "<|vision_start|>";
    llm_param.img_end = "<|vision_end|>";
    llm_param.img_content = "<|image_pad|>";

    if(rkllm_init(&llmHandle, &llm_param, callback) != 0) {
        cerr << "Failed to initialize RKLLM\n";
        return -1;
    }

    // 初始化图像编码器
    if(init_imgenc(argv[1], &rknn_app_ctx) != 0) {
        cerr << "Failed to initialize image encoder\n";
        return -1;
    }



    // 初始化libcurl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    cout << "libcurl initialized" << endl;
    

    // 启动HTTP服务器
    httplib::Server server;
    
    //模型列表接口
    server.Get("/v1/models", [](const httplib::Request& req, httplib::Response& res) {
        json response = {
            {"object", "list"},
            {"data", {
                {
                    {"id", "RK3588-Qwen2-VL-2B"},  // 你的模型ID
                    {"object", "model"},
                    {"created", 1686935000},  // 模型创建时间戳（示例）
                    {"owned_by", "your-org"},  // 所属组织
                }
                // 可以添加更多支持的模型...
            }}
        };
        res.set_content(response.dump(), "application/json");
    });    



    server.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        try {

            std::cout << "Request received" << std::endl;
            std::cout << "请求体: " << req.body << std::endl;


            auto j = json::parse(req.body);
            bool stream = j.value("stream", false);
            
            // 检查messages是否存在
            if(!j.contains("messages") || j["messages"].empty()) {
                throw runtime_error("No messages provided");
            }
    
            string prompt_text;
            bool has_image = false;
            std::string image_base64;

            auto& content = j["messages"].back()["content"];
            if (content.is_string()) {
                // 纯文本格式：直接读取字符串
                prompt_text = content.get<string>();
            } 
            else if (content.is_array()){
                // 多模态格式
                for (const auto& content_item : j["messages"].back()["content"]) {
                    if(content_item["type"] == "text") {
                        prompt_text = content_item["text"];
                    } 
                    else if(content_item["type"] == "image_url") {
                        // 获取base64编码的图片数据（去掉可能的data:image/...前缀）
                        std::string url = content_item["image_url"]["url"];
                        size_t pos = url.find("base64,");
                        if(pos != std::string::npos) {
                            image_base64 = url.substr(pos + 7); // 跳过"base64,"前缀
                            has_image = true;
                        } else {
                            throw runtime_error("Invalid image format: expected base64 encoded data");
                        }
                    }
                }
            }

            // 如果没有文本提示，使用默认提示
            if(prompt_text.empty()) {
                prompt_text = "请帮我解释一下这个图片的内容。";
            }
    
            float img_vec[196 * 1536] = {0};
            if(has_image) {
                std::vector<uchar> image_data = base64_decode(image_base64);
                cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
                if(img.empty()) {
                    throw runtime_error("Failed to decode base64 image data");
                }

                cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
                cv::Mat processed_img = preprocess_image(img, cv::Size(392, 392));
                
                // 提取图像特征
                if(run_imgenc(&rknn_app_ctx, processed_img.data, img_vec) != 0) {
                    throw runtime_error("Failed to run image encoder");
                }
            }
    
            // 构建Prompt
            string image_prefix = "<image>";
            string prompt = PROMPT_TEXT_PREFIX + image_prefix + prompt_text + PROMPT_TEXT_POSTFIX;
            // std::cout << "full_prompt: " << prompt << std::endl;
            
            // 调用RKLLM
            RKLLMInput input;
            if(has_image) {  //多模态输入
                input.input_type = RKLLM_INPUT_MULTIMODAL;
                input.multimodal_input.prompt = (char*)prompt.c_str();
                input.multimodal_input.image_embed = img_vec;
                input.multimodal_input.n_image_tokens = 196;
            }else{            //文本输入
                input.input_type = RKLLM_INPUT_PROMPT;
                input.prompt_input = (char*)prompt.c_str();
            }
     
            string assistant_reply;
            RKLLMInferParam infer_param;
            memset(&infer_param, 0, sizeof(RKLLMInferParam));
            infer_param.mode = RKLLM_INFER_GENERATE;
            rkllm_run(llmHandle, &input, &infer_param, &assistant_reply);
            
            //返回OpenAI兼容格式响应
            json response = {
                {"id", "chatcmpl-B9MHDbslfkBeAs8l4bebGdFOJ6PeG"},
                {"object", "chat.completion"},
                {"created", time(nullptr)},
                {"model", "Qwen2-VL-2B"},
                {"choices", {
                    {
                        {"index", 0},
                        {"message", {
                            {"role", "assistant"},
                            {"content", assistant_reply}
                        }},
                        {"finish_reason", "stop"}
                    }
                }},
                {"usage", {
                    {"prompt_tokens", 0},  
                    {"completion_tokens", 0},
                    {"total_tokens", 0}
                }}
            };
            std::cout << "输出响应: " << response << std::endl;
            res.set_content(response.dump(), "application/json");
            
        } catch(const exception& e) {
            res.status = 400;
            res.set_content(json{
                {"error", {
                    {"message", e.what()},
                    {"type", "invalid_request_error"}
                }}
            }.dump(), "application/json");
        }
    });


    cout << "Server running on port " << argv[3] << endl;
    if (!server.listen("0.0.0.0", atoi(argv[3]))) {
        cerr << "Failed to start server on port " << argv[3] << endl;
        return -1;
    }

    // 清理资源
    curl_global_cleanup();
    release_imgenc(&rknn_app_ctx);
    rkllm_destroy(llmHandle);
    return 0;
}