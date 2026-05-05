#ifndef GAUSSIAN_BLUR_TEST_FILE_HANDLING_CUH
#define GAUSSIAN_BLUR_TEST_FILE_HANDLING_CUH

#include <string>
#include <vector>

std::vector<unsigned char> read_file(const std::string &image_path);
std::vector<float> make_gaussian_mask(int R, float sigma);
std::string default_output_image_path(const std::string &image_path, int channels);
void write_output_image(const std::string &output_image_path,
                        const float *output,
                        int image_width,
                        int image_height,
                        int channels);

#endif  // GAUSSIAN_BLUR_TEST_FILE_HANDLING_CUH
