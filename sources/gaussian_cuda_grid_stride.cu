#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

struct GaussianParams{
  int width;
  int height;
  int radius;
  float std_dev;

  float* in_r;
  float* in_g;
  float* in_b;

  float* out_r;
  float* out_g;
  float* out_b;
};

__device__
float ComputeWeight(float center_x, float center_y, float curr_x, float curr_y, float std_dev){
  float x = center_x - curr_x;
  float y = center_y - curr_y;

  float e_pow = (x*x + y*y) / (2.0f * std_dev * std_dev) * -1.0f;
  float e = expf(e_pow);
  float frac = 1.0f / sqrtf(2.0f * M_PIf * (std_dev * std_dev));

  return frac * e;
}

__global__
void GaussianBlur(GaussianParams *g_param){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int pixelCount = g_param->width * g_param->height;

  for (int i = index; i < pixelCount; i += stride){
    int curr_x = i % g_param->width;
    int curr_y = i / g_param->width;

    float totR = 0, totG = 0, totB = 0;
    float baseMax = 0.0f;

    int minX = max(curr_x - g_param->radius, 0);
    int maxX = min(curr_x + g_param->radius + 1, g_param->width);

    int minY = max(curr_y - g_param->radius, 0);
    int maxY = min(curr_y + g_param->radius + 1, g_param->height);

    for (int y = minY; y < maxY; y++){
      for (int x = minX; x < maxX; x++){
        float weight = ComputeWeight(curr_x, curr_y, (float)x, (float)y, g_param->std_dev);
        int idx = y * g_param->width + x;

        totR += g_param->in_r[idx] * weight;
        totG += g_param->in_g[idx] * weight;
        totB += g_param->in_b[idx] * weight;

        baseMax += weight;
      }
    }

    if (baseMax > 1e-6f) {
      float ceilVal = 1.0f / baseMax;
      g_param->out_r[i] = totR * ceilVal;
      g_param->out_g[i] = totG * ceilVal;
      g_param->out_b[i] = totB * ceilVal;
    } else {
      g_param->out_r[i] = g_param->in_r[i];
      g_param->out_g[i] = g_param->in_g[i];
      g_param->out_b[i] = g_param->in_b[i];
    }
  }
}

//Calculate RMSE from 2 images
double calculateRMSE3(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Images are empty." << std::endl;
        return -1.0;
    }
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        std::cerr << "Error: Images must have the same dimensions and type." << std::endl;
        return -1.0;
    }

    cv::Mat float_img1, float_img2;
    img1.convertTo(float_img1, CV_32FC(img1.channels()), 1.0/255.0);
    img2.convertTo(float_img2, CV_32FC(img2.channels()), 1.0/255.0);

    cv::Mat diff;
    cv::absdiff(float_img1, float_img2, diff);

    cv::Mat squared_diff;
    cv::pow(diff, 2, squared_diff);

    cv::Scalar mse = cv::mean(squared_diff);

    double ave_mse = (mse.val[0] + mse.val[1] + mse.val[2]) / 3.0;

    std::cout << "MSE: " << ave_mse << std::endl;

    double rmse = std::sqrt(ave_mse);

    return rmse;
}

double calculateRMSE(float *a_r, float *a_g, float *a_b,
                      float *b_r, float *b_g, float *b_b,
                      int len)
{
  double diff_r, diff_g, diff_b;
  double sq_diff_r = 0.0;
  double sq_diff_g = 0.0;
  double sq_diff_b = 0.0;

  for(int i=0; i<len; i++)
  {
    diff_r = a_r[i] - b_r[i];
    diff_g = a_g[i] - b_g[i];
    diff_b = a_b[i] - b_b[i];

    sq_diff_r += diff_r * diff_r;
    sq_diff_g += diff_g * diff_g;
    sq_diff_b += diff_b * diff_b;
  }

  double avg_sq_diff_r = sq_diff_r / len;
  double avg_sq_diff_g = sq_diff_g / len;
  double avg_sq_diff_b = sq_diff_b / len;

  double mse = (avg_sq_diff_r + avg_sq_diff_g + avg_sq_diff_b) / 3.0;
  std::cout << "MSE: " << mse << std::endl;

  double rmse = std::sqrt(mse);
  return rmse;
}

double RSME(GaussianParams *g_param, const cv::Mat& img2){
  int array_byte_size = g_param->width * g_param->height * sizeof(float);

  float *comp_r = (float*)malloc(array_byte_size);
  float *comp_g = (float*)malloc(array_byte_size);
  float *comp_b = (float*)malloc(array_byte_size);

   // Copy pixel data
  for (int y = 0; y < img2.rows; ++y) {
    for (int x = 0; x < img2.cols; ++x) {
      const cv::Vec3b& p = img2.at<cv::Vec3b>(y, x);

      comp_r[y * img2.cols + x] = p[2] / 255.0f;
      comp_g[y * img2.cols + x] = p[1] / 255.0f;
      comp_b[y * img2.cols + x] = p[0] / 255.0f;
    }
  }

  return calculateRMSE(comp_r, comp_g, comp_b,
      g_param->out_r, g_param->out_g, g_param->out_b,
      img2.cols * img2.rows);
}

double RSME_FromFile(GaussianParams *g_param, const std::string& filename){
  int array_byte_size = g_param->width * g_param->height * sizeof(float);

  float *comp_r = (float*)malloc(array_byte_size);
  float *comp_g = (float*)malloc(array_byte_size);
  float *comp_b = (float*)malloc(array_byte_size);

  int index = 0;

  std::ifstream file(filename);
  if (!file.is_open()) {
      std::cerr << "Error opening file!" << std::endl;
      return -1;
  }

  std::string line;
  while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string field;
      std::vector<std::string> row_data;

      int col = 0;
      // Extract fields separated by commas
      while (std::getline(ss, field, ',')) {
          switch(col){
            case 0:
              comp_r[index] = std::stof(field);
              break;
            case 1:
              comp_g[index] = std::stof(field);
              break;
            case 2:
              comp_b[index] = std::stof(field);
              break;
          }
          col++;
      }

      index++;
  }

  file.close();
  return calculateRMSE(comp_r, comp_g, comp_b,
      g_param->out_r, g_param->out_g, g_param->out_b,
      g_param->width * g_param->height);
}

int main()
{
  GaussianParams* gaussianParams  = nullptr;
  cudaMallocManaged(&gaussianParams, sizeof(GaussianParams));

  // IMAGE USED = 512x512
  std::string imagePath = "512x512_sample.png";
  //Baseline image from C
  std::string compPath = "512x512_outputImage.jpg";

  cv::Mat hostImage = cv::imread(imagePath, cv::IMREAD_COLOR);

  if (hostImage.empty()) {
      std::cerr << "Error: Could not read image!" << std::endl;
      return -1;
  }

  gaussianParams->width  = hostImage.cols;
  gaussianParams->height = hostImage.rows;
  gaussianParams->radius = 20;
  gaussianParams->std_dev = 20.0f;

  std::cout << "Image loaded successfully!" << std::endl;
  std::cout << "Image dimensions: " << gaussianParams->width << "x" << gaussianParams->height << std::endl;

  int array_byte_size = gaussianParams->width * gaussianParams->height * sizeof(float);

  cudaMallocManaged(&gaussianParams->in_r, array_byte_size);
  cudaMallocManaged(&gaussianParams->in_g, array_byte_size);
  cudaMallocManaged(&gaussianParams->in_b, array_byte_size);

  cudaMallocManaged(&gaussianParams->out_r, array_byte_size);
  cudaMallocManaged(&gaussianParams->out_g, array_byte_size);
  cudaMallocManaged(&gaussianParams->out_b, array_byte_size);

  // Copy pixel data
  for (int y = 0; y < hostImage.rows; ++y) {
    for (int x = 0; x < hostImage.cols; ++x) {
      cv::Vec3b& p = hostImage.at<cv::Vec3b>(y, x);

      gaussianParams->in_r[y * hostImage.cols + x] = p[2] / 255.0f;
      gaussianParams->in_g[y * hostImage.cols + x] = p[1] / 255.0f;
      gaussianParams->in_b[y * hostImage.cols + x] = p[0] / 255.0f;
    }
  }

  size_t threads = 256;
  size_t blocks = ((gaussianParams->width * gaussianParams->height) + threads - 1) / threads;

  std::cout << "Will Process Image "<< std::endl;
  for(int i = 0; i < 10; i++) {
    GaussianBlur<<<blocks, threads>>>(gaussianParams);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
  }
  cudaDeviceSynchronize();
  std::cout << "Finished processing." << std::endl;

  // Copy back output
  //float *host_out_r = (float*)malloc(array_byte_size);
  //float *host_out_g = (float*)malloc(array_byte_size);
  //float *host_out_b = (float*)malloc(array_byte_size);

 // cudaMemcpy(host_out_r, gaussianParams->out_r, array_byte_size, cudaMemcpyDeviceToHost);
 // cudaMemcpy(host_out_g, gaussianParams->out_g, array_byte_size, cudaMemcpyDeviceToHost);
 // cudaMemcpy(host_out_b, gaussianParams->out_b, array_byte_size, cudaMemcpyDeviceToHost);

  // Save output image
  cv::Mat outimage(hostImage.rows, hostImage.cols, CV_8UC3);

  for (int y = 0; y < outimage.rows; ++y) {
    for (int x = 0; x < outimage.cols; ++x) {
      int idx = y * outimage.cols + x;
      outimage.at<cv::Vec3b>(y, x) =
        cv::Vec3b(
          (uchar)(std::min(1.f, std::max(0.f, gaussianParams->out_b[idx])) * 255),
          (uchar)(std::min(1.f, std::max(0.f, gaussianParams->out_g[idx])) * 255),
          (uchar)(std::min(1.f, std::max(0.f, gaussianParams->out_r[idx])) * 255)
        );
    }
  }

  cv::imwrite("512x512_outputImage.jpg", outimage);
  std::cout << "512x512_outputImage.jpg saved." << std::endl;

  ////Validation
  cv::Mat compImage = cv::imread(compPath, cv::IMREAD_COLOR);
  cv::Mat outputImage = cv::imread("512x512_outputImage.jpg", cv::IMREAD_COLOR);

  std::cout << "Validation from image" << std::endl;
  auto rmse = RSME(gaussianParams, compImage);
  std::cout << "RMSE from image: " << rmse << std::endl;

  std::cout << "Validation from file" << std::endl;
  auto rmse2 = RSME_FromFile(gaussianParams, "output512.csv");
  std::cout << "RMSE from file " << rmse2 << std::endl;

  cudaFree(gaussianParams->in_r);
  cudaFree(gaussianParams->in_g);
  cudaFree(gaussianParams->in_b);

  cudaFree(gaussianParams->out_r);
  cudaFree(gaussianParams->out_g);
  cudaFree(gaussianParams->out_b);

  cudaFree(gaussianParams);

  return 0;
}
