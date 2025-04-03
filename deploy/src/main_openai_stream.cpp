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

// #define PROMPT_TEXT_PREFIX "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
// #define PROMPT_TEXT_POSTFIX "<|im_end|>\n<|im_start|>assistant\n"

#define PROMPT_TEXT_PREFIX "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
#define PROMPT_TEXT_POSTFIX "<|im_start|>assistant\n"



using namespace std;
using json = nlohmann::json;

LLMHandle llmHandle = nullptr;
rknn_app_context_t rknn_app_ctx;

// 全局变量用于保持流状态
struct StreamContext {
    std::mutex mutex;
    std::queue<std::string> chunks;
    bool finished = false;
};


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

// 回调函数处理流式输出，
void stream_callback(RKLLMResult *result, void *userdata, LLMCallState state) {


    StreamContext* ctx = static_cast<StreamContext*>(userdata);
    json response_chunk;
    std::string chunk_data;
    
    if (state == RKLLM_RUN_FINISH) {
        // 发送结束标记
        response_chunk = {
            {"choices", {
                {
                    {"delta", {
                        {"content", ""}
                    }},
                    {"finish_reason", "stop"},
                    {"index", 0}
                }
            }},
            {"created", time(nullptr)},
            {"id", "chatcmpl-stream-" + to_string(time(nullptr))},
            {"model", "Qwen2-VL-2B"},
            {"object", "chat.completion.chunk"}
        };
        chunk_data = "data: " + response_chunk.dump() + "\n\n";
        std::lock_guard<std::mutex> lock(ctx->mutex);

        // 发送最后一个空内容块
        ctx->chunks.push(chunk_data);
        
        //发送标准结束标记（兼容OpenAI）
        ctx->chunks.push("data: [DONE]\n\n");

        ctx->finished = true;
    }
    else if (state == RKLLM_RUN_ERROR) {
        // 发送错误信息
        response_chunk = {
            {"error", {
                {"message", "Error occurred during generation"},
                {"type", "generation_error"}
            }}
        };
        chunk_data = "data: " + response_chunk.dump() + "\n\n";
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ctx->chunks.push(chunk_data);
        ctx->chunks.push("data: [DONE]\n\n");
        ctx->finished = true;
    }
    else if (state == RKLLM_RUN_NORMAL) {
        printf("%s", result->text);
        // 发送正常内容块
        response_chunk = {
            {"choices", {
                {
                    {"delta", {
                        {"content", result->text}
                    }},
                    {"index", 0}
                }
            }},
            {"created", time(nullptr)},
            {"id", "chatcmpl-stream-" + to_string(time(nullptr))},
            {"model", "Qwen2-VL-2B"},
            {"object", "chat.completion.chunk"}
        };
        chunk_data = "data: " + response_chunk.dump() + "\n\n";
        std::lock_guard<std::mutex> lock(ctx->mutex);
        // 将流式输出数据添加到ctx队列中
        ctx->chunks.push(chunk_data);
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

// 纯文本输入时，根据关键字引用历史图片
bool is_referencing_image(const std::string& text) {
    // 所有可能引用图片的关键词
    const std::string keywords[] = {
        "图片", "照片", "图像", "图示", "图案", "图表", "插图",
        "上一张"
    };

    // 检查文本是否包含任一关键词
    for (const auto& word : keywords) {
        if (text.find(word) != std::string::npos) {
            return true;
        }
    }
    return false;
}




int main(int argc, char** argv) {
    if(argc < 4) {
        cerr << "Usage: " << argv[0] << " <encoder_model_path> <llm_model_path> <port>\n";
        return -1;
    }

    // 初始化RKLLM
    RKLLMParam llm_param = rkllm_createDefaultParam();
    llm_param.model_path = argv[2];
    llm_param.max_new_tokens = 8192;
    llm_param.max_context_len = 4096;
    llm_param.skip_special_token = true;
    llm_param.img_start = "<|vision_start|>";
    llm_param.img_end = "<|vision_end|>";
    llm_param.img_content = "<|image_pad|>";

    if(rkllm_init(&llmHandle, &llm_param, stream_callback) != 0) {
        cerr << "Failed to initialize RKLLM\n";
        return -1;
    }

    // 初始化图像编码器
    if(init_imgenc(argv[1], &rknn_app_ctx) != 0) {
        cerr << "Failed to initialize image encoder\n";
        return -1;
    }


    // 启动HTTP服务器
    httplib::Server server;
    
    // 模型列表接口
    server.Get("/v1/models", [](const httplib::Request& req, httplib::Response& res) {
        json response = {
            {"object", "list"},
            {"data", {
                {
                    {"id", "RK3588-Qwen2-VL-2B"},
                    {"object", "model"},
                    {"created", 1686935000},
                    {"owned_by", "your-org"},
                }
            }}
        };
        res.set_content(response.dump(), "application/json");
    });
    
    // 处理POST请求
    server.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            // 打印请求体
            std::cout << "请求体: " << req.body << std::endl;

            // 解析JSON请求
            auto j = json::parse(req.body);
            bool stream = j.value("stream", false);
            if(!stream) {
                throw runtime_error("Only streaming output is supported");
            }
            
            // 检查messages是否存在
            if(!j.contains("messages") || j["messages"].empty()) {
                throw runtime_error("No messages provided");
            }
            
       
            // 获取prompt和图片
            string prompt_text;
            bool has_image = false;
            std::string image_base64;
            string image_prefix = "<image>";

            // 存储对话历史中的最新图片信息
            struct {
                std::string base64;
                std::string description; // 可选的图片描述
            } latest_image_info;


            // 先找出多模态请求文本中最新图片的base64编码以及索引位置（历史和当前user）
            size_t latest_image_index = 0;
            bool found_latest_image = false;
            for (size_t i = 0; i < j["messages"].size(); ++i) {
                const auto& msg = j["messages"][i];
                // 检查用户多模态图片消息
                if (msg["role"] == "user" && msg["content"].is_array()) {
                    for (const auto& content_item : msg["content"]) {
                        if (content_item["type"] == "image_url") {
                            std::string url = content_item["image_url"]["url"];
                            size_t pos = url.find("base64,"); 
                            if (pos != std::string::npos) {
                                latest_image_info.base64 = url.substr(pos + 7);
                                latest_image_index = i;  // 记录位置索引
                                found_latest_image = true;
                            }
                        }
                    }
                }
            }


            // 统一处理所有消息
            for (size_t i = 0; i < j["messages"].size(); ++i) {
                const auto& msg = j["messages"][i];

                // 每次迭代重置has_image, 支持多模态转纯文本对话
                has_image = false;

                // 每次迭代都清理历史的image占位符,避免干扰
                prompt_text = std::regex_replace(prompt_text, std::regex("<image>"), "");

                if (msg["role"] == "user") {
                    prompt_text += "<|im_start|>user\n";
                    
                    bool current_has_image = false;
                    std::string user_text;
                    // 检查当前用户索引是否为最新图片所在索引
                    bool is_latest_image_msg = (i == latest_image_index);
                    
                    // 处理多模态消息
                    if (msg["content"].is_array()) {
                        for (const auto& content_item : msg["content"]) {
                            if (content_item["type"] == "image_url") {
                                std::string url = content_item["image_url"]["url"];
                                // 检查是否包含base64编码的图片数据
                                size_t pos = url.find("base64,"); 
                                if (pos != std::string::npos) {
                                    current_has_image = true;
                                    // 判断当前是否为最新的图片，是则后续输入编码器
                                    if (is_latest_image_msg) {
                                        image_base64 = url.substr(pos + 7);
                                        has_image = true;
                                    }
                                }
                            }
                            // 添加多模态的文本(历史和当前都包括，无<image>占位符)
                            else if (content_item["type"] == "text") {
                                user_text = content_item["text"];
                            }
                        }
                    }
                    // 处理纯文本消息 
                    else {
                        // 读取纯文本请求中的content直接获得text
                        user_text = msg["content"].get<std::string>();
                        // 检查纯文本关键字触发关联历史图片
                        if (is_referencing_image(user_text) && found_latest_image) {
                            current_has_image = true;
                            has_image = true;
                            // 强制使用最新图片
                            if (!latest_image_info.base64.empty()) {
                                image_base64 = latest_image_info.base64;
                            }
                        }
                    }
            
                    // 最新为多模态输入，文本添加<image>占位符
                    // 最新为纯文本输入且触发关联历史图片，文本添加<image>占位符
                    if ((is_latest_image_msg && current_has_image) || (msg["content"].is_string() && current_has_image)) {
                        prompt_text += image_prefix;
                    }
                    
                    // 多模态输入下，只传送图片场景下，添加默认文本
                    prompt_text += user_text.empty() ? 
                        (current_has_image ? "请描述这张图片" : "") : 
                        user_text;
                    
                    prompt_text += "<|im_end|>\n";
                }
                // 添加历史模型回复上下文
                else if (msg["role"] == "assistant") {
                    std::string assistant_text = msg["content"].get<std::string>();
                    assistant_text = std::regex_replace(assistant_text, std::regex("<image>"), "");
                    prompt_text += "<|im_start|>assistant\n" + assistant_text + "<|im_end|>\n";
                }
            }



            // 图像编码处理
            float img_vec[196 * 1536] = {0};
            if(has_image) {
                std::vector<uchar> image_data = base64_decode(image_base64);
                cv::Mat img = cv::imdecode(image_data, cv::IMREAD_COLOR);
                if(img.empty()) {
                    throw runtime_error("Failed to decode base64 image data");
                }

                cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
                cv::Mat processed_img = preprocess_image(img, cv::Size(392, 392));
                
                if(run_imgenc(&rknn_app_ctx, processed_img.data, img_vec) != 0) {
                    throw runtime_error("Failed to run image encoder");
                }
            }
    
            string prompt = PROMPT_TEXT_PREFIX + prompt_text + PROMPT_TEXT_POSTFIX;
            printf("prompt_text: %s\n", prompt.c_str());
            
            // 设置流式响应头
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");

            // 创建流上下文
            auto ctx = std::make_shared<StreamContext>();
            
            // HTTP线程处理函数
            res.set_chunked_content_provider(
                "text/event-stream",
                [ctx](size_t offset, httplib::DataSink& sink) {
                    std::unique_lock<std::mutex> lock(ctx->mutex);
                    
                    // 主线程持续从ctx->chunks检查队列并返回数据（轮询机制）
                    while (!ctx->chunks.empty()) {
                        std::string chunk = ctx->chunks.front();
                        ctx->chunks.pop();
                        
                        lock.unlock();  // 释放锁，避免阻塞
                        bool write_ok = sink.write(chunk.data(), chunk.size());
                        lock.lock();   // 重新加锁
                        
                        if (!write_ok) return false;  // 写入失败
                    }
                    
                    // 检查是否应该结束
                    if (ctx->finished) {
                        sink.done();  // 主动关闭流
                        return false;
                    }
                    
                    return true;  // 继续等待数据
                }
            );
            

            // 子线程生成LLM结果和共享上下文（ctx）的异步通信流式输出
            std::thread([&, ctx, prompt, has_image]() {
                // 1. 准备输入（根据是否有图片选择多模态或普通文本输入）
                RKLLMInput input;
                if (has_image) {
                    input.input_type = RKLLM_INPUT_MULTIMODAL;
                    input.multimodal_input.prompt = (char*)prompt.c_str();
                    input.multimodal_input.image_embed = img_vec;
                    input.multimodal_input.n_image_tokens = 196;
                } else {
                    input.input_type = RKLLM_INPUT_PROMPT;
                    input.prompt_input = (char*)prompt.c_str();
                }

                // 2. 运行LLM
                RKLLMInferParam infer_param;
                memset(&infer_param, 0, sizeof(RKLLMInferParam));
                infer_param.mode = RKLLM_INFER_GENERATE;
    
                // 3. 调用LLM生成，（回调里阻塞）
                rkllm_run(llmHandle, &input, &infer_param, ctx.get());

                // 4. 生成完成后，确保结束标记已处理
                std::lock_guard<std::mutex> lock(ctx->mutex);
                if (!ctx->finished) {  // 双重检查
                    ctx->chunks.push("data: [DONE]\n\n");
                    ctx->finished = true;
                }
            }).detach();  // detach线程，让其独立于HTTP主线程在后台持续运行
            
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
    // curl_global_cleanup();
    release_imgenc(&rknn_app_ctx);
    rkllm_destroy(llmHandle);
    return 0;
}