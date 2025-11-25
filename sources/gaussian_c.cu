#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>

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

float ComputeWeight(float center_x, float center_y, float curr_x, float curr_y, float std_dev){
  float x = center_x - curr_x;
  float y = center_y - curr_y;

  float x2 = x*x;
  float y2 = y*y;
  float std2 = std_dev*std_dev;

  float e_pow = (x2 + y2) / (2* std2) * -1;
  float e = std::exp(e_pow);

  float frac = 1 / (std::sqrt(2 * M_PI * std2));

  return frac * e;
}



void GaussianBlur(GaussianParams *g_param){
  std::cout << "Image dimensions: " << g_param->width << "x" << g_param->height << std::endl;

  for (int curr_y = 0; curr_y < g_param->height; ++curr_y) {
      for (int curr_x = 0; curr_x < g_param->width; ++curr_x) {

          int index = curr_y * g_param->width + curr_x;
          float r = g_param->in_r[index];
          float g = g_param->in_g[index];
          float b = g_param->in_b[index];

          float totR = 0;
          float totG = 0;
          float totB = 0;

          float baseMax = 0.f;

          //To factor in the corners
          float minX = std::max(curr_x - g_param->radius, 0);
          float maxX = std::min(curr_x + g_param->radius, g_param->width);

          float minY = std::max(curr_y - g_param->radius, 0);
          float maxY = std::min(curr_y + g_param->radius, g_param->height);
          /////////////////////////////////////////////////////////////////

          //Compute for the summation of colors here
          for(int y = minY; y < maxY; y++){
            for(int x = minX; x < maxX; x++){
              float weight = ComputeWeight(curr_x, curr_y, x, y, g_param->std_dev);
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
          float ceilVal = 1.f / baseMax;

          totR *= ceilVal;
          totG *= ceilVal;
          totB *= ceilVal;
          ///////////////////////////

          g_param->out_r[index] = totR;
          g_param->out_g[index] = totG;
          g_param->out_b[index] = totB;
      }
  }
}

void WriteToCSV(GaussianParams *g_param, std::string filename){
  std::ofstream outputFile(filename);

   if (outputFile.is_open()) {
        int len = g_param->width * g_param->height;
        std::setprecision(10);
        for (int i = 0; i < len; i++) {
          //RGB
            outputFile << g_param->out_r[i] << "," << g_param->out_g[i] << "," << g_param->out_b[i];
            if (i < len - 1) {
                outputFile << "\n";
            }
        }
        std::setprecision(0);
        outputFile.close();
        std::cout << "C Kernel output written to: " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }
}

int main()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

  GaussianParams* gaussianParams = new GaussianParams;

  //You need to upload the image to load manually on the left side of the collab
  //Google Collab deletes files on close >..>
  std::string imagePath = "512x512_sample.png"; //<-- CHANGE FILENAME AS NEEDED
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

  float *r, *g, *b;
  r = (float*)malloc(array_byte_size);
  g = (float*)malloc(array_byte_size);
  b = (float*)malloc(array_byte_size);

  gaussianParams->in_r = (float*)malloc(array_byte_size);
  gaussianParams->in_g = (float*)malloc(array_byte_size);
  gaussianParams->in_b = (float*)malloc(array_byte_size);

  gaussianParams->out_r = (float*)malloc(array_byte_size);
  gaussianParams->out_g = (float*)malloc(array_byte_size);
  gaussianParams->out_b = (float*)malloc(array_byte_size);

  ///Load and Read Pixel Data
  for (int y = 0; y < hostImage.rows; ++y) {
      for (int x = 0; x < hostImage.cols; ++x) {
          // Access pixel at (x, y) coordinates
          cv::Vec3b& pixel = hostImage.at<cv::Vec3b>(y, x);

          // Its in (BGR order)
          uchar blue = pixel[0];
          uchar green = pixel[1];
          uchar red = pixel[2];

          int i_r = static_cast<int>(red);
          int i_g = static_cast<int>(green);
          int i_b = static_cast<int>(blue);

          r[y * hostImage.cols + x] = static_cast<float>(i_r/255.f);
          g[y * hostImage.cols + x] = static_cast<float>(i_g/255.f);
          b[y * hostImage.cols + x] = static_cast<float>(i_b/255.f);
      }
  }

  memcpy(gaussianParams->in_r, r, array_byte_size);
  memcpy(gaussianParams->in_g, g, array_byte_size);
  memcpy(gaussianParams->in_b, b, array_byte_size);

  auto z0 = high_resolution_clock::now();
  duration<double, std::milli> timeElapsed = z0 - z0;

  std::cout << "Initial Time Elapsed: " << timeElapsed.count() << " ms\n";

  std::cout << "Will Process Image "<< std::endl;
  for(int i = 0; i < 1; i++) {
    auto sTime = high_resolution_clock::now();
    GaussianBlur(gaussianParams);
    auto eTime = high_resolution_clock::now();
    duration<double, std::milli> exeTime = eTime - sTime;
    std::cout << "Run " << i << " Execution Time: " << exeTime.count() << " ms\n";
    timeElapsed = timeElapsed + exeTime;
  }
  std::cout << "Donee Process Image "<< std::endl;

  //Save image
  cv::Mat outimage(hostImage.rows, hostImage.cols, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int y = 0; y < outimage.rows; ++y) {
      for (int x = 0; x < outimage.cols; ++x) {

          int index = y * outimage.cols + x;

          outimage.at<cv::Vec3b>(y, x) =
            cv::Vec3b(gaussianParams->out_b[index] * 255,
                      gaussianParams->out_g[index] * 255,
                      gaussianParams->out_r[index] * 255);
      }
  }

  //Download the output image manually from the left side
  bool success = cv::imwrite("outputImage.jpg", outimage);
  WriteToCSV(gaussianParams, "output512.csv");

  std::cout << "Image Saved " << std::endl;
  std::cout << "Average Execution Time (10 Runs): " << timeElapsed.count() << " ms\n";
  return 0;
}