#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
int width = 1920;
int height = 1080;


void block_matching()
{
    // Read raw image data from file
    std::vector<Mat>frames;
    
    for (int i = 1; i <= 10; i++) 
    {
        unsigned short* data = new unsigned short[width * height];      
        std::string filename = "im6/img" + std::to_string(i) + ".raw";       
        FILE* file = fopen64(filename.c_str(), "rb");        
        fread(data, sizeof(unsigned short), width * height, file);        
        fclose(file);  

        // Convert raw image data to OpenCV Mat object
        Mat image(height, width, CV_16UC1, data);

        frames.push_back(image);

    }
    
    int block_size = 16;
    int search_range = 16;

    // Perform motion compensation on each pair of consecutive frames
    vector<Point2f> motion_vectors;
    for (int i = 1; i < frames.size(); i++) {
        Mat curr_frame = frames[i];
        Mat prev_frame = frames[i - 1];

        // Perform block matching
        vector<Point2f> curr_points, prev_points;
        for (int y = 0; y < curr_frame.rows - block_size; y += block_size) {
            for (int x = 0; x < curr_frame.cols - block_size; x += block_size) {
                Rect curr_roi(x, y, block_size, block_size);
                Mat curr_block = curr_frame(curr_roi);

                Point2f best_match = Point2f(x, y);
                double best_score = numeric_limits<double>::max();

                for (int dy = -search_range; dy <= search_range; dy++) {
                    for (int dx = -search_range; dx <= search_range; dx++) {
                        int x_offset = x + dx;
                        int y_offset = y + dy;

                        if (x_offset < 0 || x_offset >= curr_frame.cols - block_size ||
                            y_offset < 0 || y_offset >= curr_frame.rows - block_size) {
                            continue;
                        }

                        Rect prev_roi(x_offset, y_offset, block_size, block_size);
                        Mat prev_block = prev_frame(prev_roi);

                        double score = norm(curr_block, prev_block, NORM_L2SQR);

                        if (score < best_score) {
                            best_score = score;
                            best_match = Point2f(x_offset, y_offset);
                        }
                    }
                }

                curr_points.push_back(Point2f(x + block_size / 2, y + block_size / 2));
                prev_points.push_back(Point2f(best_match.x + block_size / 2, best_match.y + block_size / 2));
            }
        }

        // Estimate motion vector as mean displacement of block centers
        Point2f mean_curr_points(0.0, 0.0);
        Point2f mean_prev_points(0.0, 0.0);
        for (int j = 0; j < curr_points.size(); j++) {
            mean_curr_points = mean_curr_points + curr_points[j];
            mean_prev_points = mean_prev_points + prev_points[j];
        }
        mean_curr_points = mean_curr_points / (float)curr_points.size();
        mean_prev_points = mean_prev_points / (float)prev_points.size();
        Point2f mean_motion_vector = mean_curr_points - mean_prev_points;

        // Store motion vector
        motion_vectors.push_back(mean_motion_vector);

        // Define transformation matrix
        Mat transform = (Mat_<double>(2, 3) << 1, 0, mean_motion_vector.x, 0, 1, mean_motion_vector.y);

        // Warp previous frame to compensate for motion
        Mat warped_prev_frame;
        warpAffine(prev_frame, warped_prev_frame, transform, curr_frame.size());

        // Blend current and warped previous frame
        Mat blended_frame;
        addWeighted(curr_frame, 0.5, warped_prev_frame, 0.5, 0, blended_frame);
        // Display blended frame
        
        

	std::string save_filename = "deghosted_image.raw";
        FILE* save_file=fopen64(save_filename.c_str(), "wb");
        fwrite(blended_frame.data, sizeof(unsigned short), width * height, save_file);
        fclose(save_file);
    }
    

}


void weighted_averaging()
{
    int image_c=10;
    // Set the image size and color depth
    
    const int depth = 12;

    // Allocate memory for the raw image data
    unsigned short* idata = new unsigned short[width * height * image_c];

    // Read the raw image data from files
    for (int i = 0; i <10; i++) {
        std::string filename = "im6/img" + std::to_string(i) + ".raw";
        FILE* file=fopen64(filename.c_str(), "rb");
        fread(&idata[i * width * height], sizeof(unsigned short), width * height, file);
        fclose(file);
    }

    // Create a vector of Mat objects to hold the image data
    vector<Mat> framess(image_c);
    for (int i = 0; i < image_c; i++) {
       framess[i] = Mat(height, width, CV_16UC1, &idata[i * width * height]);
    }

    // Perform the averaging algorithm 
    Mat image_total = Mat::zeros(height, width, CV_32FC1);
    for (int i = 0; i < image_c; i++) {
        Mat tempImage;
        framess[i].convertTo(tempImage, CV_32FC1, 1.0 / ((1 << depth) - 1));
        image_total += tempImage;
    }
    Mat averageImage;
    image_total /= image_c;
    averageImage = image_total * ((1 << depth) - 1);
    averageImage.convertTo(averageImage, CV_16UC1);

    // Save the averaged image as .raw format
    std::string save_filename = "averaged_image.raw";
    FILE* save_file=fopen64(save_filename.c_str(), "wb");
    fwrite(averageImage.data, sizeof(unsigned short), width * height, save_file);
    fclose(save_file);

    // Free memory
    delete[] idata;
}

int main() 
{
	cout<<"block_matching processing"<<endl;
	block_matching();
	cout<<"block_matching completed"<<endl;
	weighted_averaging();
	cout<<"weighted averaging completed"<<endl;
	return 0;	
}
