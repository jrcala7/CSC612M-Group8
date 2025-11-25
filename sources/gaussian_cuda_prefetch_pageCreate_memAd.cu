#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>
#include <fstream>

//#define M_PI 3.14159265358979323846
//#define M_PIf 3.14159265358979323846f


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

  float x2 = x*x;
  float y2 = y*y;
  float std2 = std_dev*std_dev;

  float e_pow = (x2 + y2) / (2* std2) * -1;
  float e = expf(e_pow);

  //float frac = 1.0f / sqrtf(2.0f * M_PIf * std2);
  float frac = 1.0f / sqrtf(2.0f * M_PIf * std2);

  return frac * e;
}

//CUDA Gaussian Blur kernel
__global__
void GaussianBlur(GaussianParams *g_param){
  //std::cout << "Image dimensions: " << g_param->width << "x" << g_param->height << std::endl;

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int pixelCount = g_param->width * g_param->height;
  for (int i = index; i < pixelCount; i += stride){
          int curr_x = i % g_param->width;
          int curr_y = i / g_param->width;

          //int index = curr_y * g_param->width + curr_x;
          //float r = g_param->in_r[index];
          //float g = g_param->in_g[index];
          //float b = g_param->in_b[index];

          float totR = 0;
          float totG = 0;
          float totB = 0;

          float baseMax = 0.f;

          //To factor in the corners
          int minX = max(curr_x - g_param->radius, 0);
          int maxX = min(curr_x + g_param->radius + 1, g_param->width);
          int minY = max(curr_y - g_param->radius, 0);
          int maxY = min(curr_y + g_param->radius + 1, g_param->height);
          /////////////////////////////////////////////////////////////////

          //Compute for the summation of colors here
          for(int y = minY; y < maxY; y++){
            for(int x = minX; x < maxX; x++){
              float weight = ComputeWeight(curr_x, curr_y, (float)x, (float)y, g_param->std_dev);

              int curr_index = y * g_param->width + x;

              totR += g_param->in_r[curr_index] * weight;
              totG += g_param->in_g[curr_index] * weight;
              totB += g_param->in_b[curr_index] * weight;

              //For normalization later
              baseMax += 1.f *weight;
            }
          }
          ///////////////////////////

          //Normalize the colors here
          //float ceilVal = 1.f / baseMax;

          //totR *= ceilVal;
          //totG *= ceilVal;
          //totB *= ceilVal;
          ///////////////////////////

          //g_param->out_r[index] = totR;
          //g_param->out_g[index] = totG;
          //g_param->out_b[index] = totB;

          if (baseMax > 1e-6f) {
              float ceilVal = 1.f / baseMax;

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

double RMSE(GaussianParams *g_param, const cv::Mat& img2){
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

double RMSE_FromFile(GaussianParams *g_param, const std::string& filename){
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
  //GaussianParams* gaussianParams = new GaussianParams;
  GaussianParams* gaussianParams  = nullptr;


  cudaMallocManaged(&gaussianParams, sizeof(GaussianParams));

  //You need to upload the image to load manually on the left side of the collab
  //Google Collab deletes files on close >..>
  std::string imagePath = "512x512_sample.png"; //<-- CHANGE FILENAME AS NEEDED
  //Baseline image from C
  std::string compPath = "512x512_sample_outputImage.jpg";

  cv::Mat hostImage = cv::imread(imagePath, cv::IMREAD_COLOR);

  if (hostImage.empty()) {
      std::cerr << "Error: Could not read the image from " << imagePath << std::endl;
      return -1;
  }

  //Set the parameters here
  gaussianParams->width = hostImage.cols;

  gaussianParams->height = hostImage.rows;
  gaussianParams->radius = 20;
  gaussianParams->std_dev = 20.0f;

  std::cout << "Image loaded successfully!" << std::endl;
  std::cout << "Image dimensions: " << gaussianParams->width << "x" << gaussianParams->height << std::endl;

  int array_byte_size = hostImage.cols * hostImage.rows * sizeof(float);
  std::cout << "Array size: " << array_byte_size << std::endl;




  cudaMallocManaged(&gaussianParams->in_r, array_byte_size);
  cudaMallocManaged(&gaussianParams->in_g, array_byte_size);
  cudaMallocManaged(&gaussianParams->in_b, array_byte_size);

  cudaMallocManaged(&gaussianParams->out_r, array_byte_size);
  cudaMallocManaged(&gaussianParams->out_g, array_byte_size);
  cudaMallocManaged(&gaussianParams->out_b, array_byte_size);


  ///Load and Read Pixel Data
  for (int y = 0; y < hostImage.rows; ++y) {
    for (int x = 0; x < hostImage.cols; ++x) {

       // Access pixel at (x, y) coordinates
          cv::Vec3b& pixel = hostImage.at<cv::Vec3b>(y, x);

          // Its in (BGR order)
          uchar blue = pixel[0];
          uchar green = pixel[1];
          uchar red = pixel[2];

          gaussianParams->in_r[y * hostImage.cols + x] = static_cast<float>(red) / 255.0f;
          gaussianParams->in_g[y * hostImage.cols + x] = static_cast<float>(green) / 255.0f;
          gaussianParams->in_b[y * hostImage.cols + x] = static_cast<float>(blue) / 255.0f;

    }
}

  // Memory Advise
   cudaMemAdvise(gaussianParams->in_r, array_byte_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
   cudaMemAdvise(gaussianParams->in_r, array_byte_size, cudaMemAdviseSetReadMostly, cudaCpuDeviceId);
   cudaMemAdvise(gaussianParams->in_g, array_byte_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
   cudaMemAdvise(gaussianParams->in_g, array_byte_size, cudaMemAdviseSetReadMostly, cudaCpuDeviceId);
   cudaMemAdvise(gaussianParams->in_b, array_byte_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
   cudaMemAdvise(gaussianParams->in_b, array_byte_size, cudaMemAdviseSetReadMostly, cudaCpuDeviceId);

  int device = -1;
  cudaGetDevice(&device);

  // Page Creation


  cudaMemPrefetchAsync(gaussianParams->in_r,array_byte_size, cudaCpuDeviceId, NULL);
  cudaMemPrefetchAsync(gaussianParams->in_g,array_byte_size, cudaCpuDeviceId, NULL);
  cudaMemPrefetchAsync(gaussianParams->in_b,array_byte_size, cudaCpuDeviceId, NULL);

  cudaMemPrefetchAsync(gaussianParams->out_r,array_byte_size, device, NULL);
  cudaMemPrefetchAsync(gaussianParams->out_g,array_byte_size, device, NULL);
  cudaMemPrefetchAsync(gaussianParams->out_b,array_byte_size, device, NULL);



  // Prefetch data from CPU-GPU
  //cudaMemPrefetchAsync(gaussianParams, sizeof(GaussianParams), device, NULL);


  cudaMemPrefetchAsync(gaussianParams->in_r,array_byte_size, device, NULL);
  cudaMemPrefetchAsync(gaussianParams->in_g,array_byte_size, device, NULL);
  cudaMemPrefetchAsync(gaussianParams->in_b,array_byte_size, device, NULL);


  size_t threads = 256;
  size_t blocks = ((hostImage.cols * hostImage.rows) + threads - 1) / threads;

  std::cout << "Will Process Image "<< std::endl;
  //GaussianBlur(gaussianParams);
  for(int i = 0; i < 10; i++) {
    GaussianBlur<<<blocks, threads>>>(gaussianParams);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
  }
  cudaDeviceSynchronize();
  std::cout << "Done Process Image "<< std::endl;



  cudaMemPrefetchAsync(gaussianParams->out_r,array_byte_size, cudaCpuDeviceId, NULL);
  cudaMemPrefetchAsync(gaussianParams->out_g,array_byte_size, cudaCpuDeviceId, NULL);
  cudaMemPrefetchAsync(gaussianParams->out_b,array_byte_size, cudaCpuDeviceId, NULL);


 // float *host_out_r = (float*)malloc(array_byte_size);
 // float *host_out_g = (float*)malloc(array_byte_size);
 // float *host_out_b = (float*)malloc(array_byte_size);

 // cudaMemcpy(host_out_r, gaussianParams->out_r, array_byte_size, cudaMemcpyDeviceToHost);
 // cudaMemcpy(host_out_g, gaussianParams->out_g, array_byte_size, cudaMemcpyDeviceToHost);
 // cudaMemcpy(host_out_b, gaussianParams->out_b, array_byte_size, cudaMemcpyDeviceToHost);

  //Save image
  cv::Mat outimage(hostImage.rows, hostImage.cols, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int y = 0; y < outimage.rows; ++y) {
      for (int x = 0; x < outimage.cols; ++x) {

          int index = y * outimage.cols + x;

          float b_val = std::min(1.0f, std::max(0.0f, gaussianParams->out_b[index]));
          float g_val = std::min(1.0f, std::max(0.0f, gaussianParams->out_g[index]));
          float r_val = std::min(1.0f, std::max(0.0f, gaussianParams->out_r[index]));

          outimage.at<cv::Vec3b>(y, x) =
            cv::Vec3b(static_cast<uchar>(b_val * 255.0f),
                      static_cast<uchar>(g_val * 255.0f),
                      static_cast<uchar>(r_val * 255.0f));
      }
  }

  //std::cout << "Sample output pixel: R=" << host_out_r[0]
  //        << " G=" << host_out_g[0]
   //       << " B=" << host_out_b[0] << std::endl;

  //Download the output image manually from the left side
  bool success = cv::imwrite("512x512_outputImage_cuda_abcd.jpg", outimage);

  std::cout << "Image Saved " << std::endl;

  // Free memory
  //free(host_out_r);
  //free(host_out_g);
  //free(host_out_b);


  ////Validation
  cv::Mat compImage = cv::imread(compPath, cv::IMREAD_COLOR);
  cv::Mat outputImage = cv::imread("512x512_outputImage_cuda_abcd.jpg", cv::IMREAD_COLOR);

  std::cout << "Validation from image" << std::endl;
  auto rmse = RMSE(gaussianParams, compImage);
  std::cout << "RMSE from image: " << rmse << std::endl;

  std::cout << "Validation from file" << std::endl;
  auto rmse2 = RMSE_FromFile(gaussianParams, "output512.csv");
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