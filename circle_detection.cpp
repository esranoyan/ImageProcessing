#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <climits>

// Define M_PI
#define M_PI 3.14159265358979323846

// Simple structure to represent an image
struct Image {
    int width;
    int height;
    std::vector<std::vector<unsigned char>> pixels;
};

// Structure to represent a circle
struct Circle {
    int x;      // center x-coordinate
    int y;      // center y-coordinate
    int radius; // radius of circle
    int votes;  // number of votes (for Hough Transform)
};

// Function to load a simple PPM image (P6 format)
Image loadPPM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return { 0, 0, {} };
    }
    // Read PPM header
    std::string format;
    file >> format;
    if (format != "P6") {
        std::cerr << "Unsupported format: " << format << std::endl;
        return { 0, 0, {} };
    }
    int width, height, maxVal;
    file >> width >> height >> maxVal;

    // Skip whitespace
    file.get();
    // Read pixel data
    std::vector<unsigned char> buffer(width * height * 3);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    // Convert RGB to grayscale
    Image img;
    img.width = width;
    img.height = height;
    img.pixels.resize(height, std::vector<unsigned char>(width, 0));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            // Simple grayscale conversion: average of RGB values
            img.pixels[y][x] = static_cast<unsigned char>(
                (buffer[idx] + buffer[idx + 1] + buffer[idx + 2]) / 3);
        }
    }
    return img;
}

// Function to save a grayscale image as PPM
void saveGrayscaleAsPPM(const Image& image, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    // Create a color output image (convert grayscale to RGB)
    std::vector<unsigned char> output(image.width * image.height * 3, 0);

    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            int idx = (y * image.width + x) * 3;
            output[idx] = image.pixels[y][x];     // Red channel
            output[idx + 1] = image.pixels[y][x]; // Green channel
            output[idx + 2] = image.pixels[y][x]; // Blue channel
        }
    }

    // Write PPM header
    file << "P6\n" << image.width << " " << image.height << "\n255\n";

    // Write pixel data
    file.write(reinterpret_cast<const char*>(output.data()), output.size());

    std::cout << "Image saved to: " << filename << std::endl;
}

// Function to save an image with detected circles
void saveImageWithCircles(const Image& image, const std::vector<Circle>& circles,
    const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    // Create a color output image
    std::vector<unsigned char> output(image.width * image.height * 3, 0);

    // Convert grayscale to RGB
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            int idx = (y * image.width + x) * 3;
            output[idx] = image.pixels[y][x];
            output[idx + 1] = image.pixels[y][x];
            output[idx + 2] = image.pixels[y][x];
        }
    }

    // Draw circles in red
    for (const auto& circle : circles) {
        // Draw the circle outline
        for (int theta = 0; theta < 360; theta++) {
            double radian = theta * M_PI / 180.0;
            int x = circle.x + static_cast<int>(circle.radius * cos(radian));
            int y = circle.y + static_cast<int>(circle.radius * sin(radian));

            if (x >= 0 && x < image.width && y >= 0 && y < image.height) {
                int idx = (y * image.width + x) * 3;
                output[idx] = 255;     // Red channel
                output[idx + 1] = 0;   // Green channel
                output[idx + 2] = 0;   // Blue channel
            }
        }

        // Mark the center with a small cross
        for (int dx = -2; dx <= 2; ++dx) {
            for (int dy = -2; dy <= 2; ++dy) {
                if ((dx == 0 || dy == 0) && !(dx == 0 && dy == 0)) {
                    int x = circle.x + dx;
                    int y = circle.y + dy;
                    if (x >= 0 && x < image.width && y >= 0 && y < image.height) {
                        int idx = (y * image.width + x) * 3;
                        output[idx] = 0;       // Red channel
                        output[idx + 1] = 255; // Green channel
                        output[idx + 2] = 0;   // Blue channel
                    }
                }
            }
        }
    }

    // Write PPM header
    file << "P6\n" << image.width << " " << image.height << "\n255\n";

    // Write pixel data
    file.write(reinterpret_cast<const char*>(output.data()), output.size());

    std::cout << "Image with circles saved to: " << filename << std::endl;
}

// Function to resize an image to a smaller dimension
Image resizeImage(const Image& original, int targetMaxDimension) {
    double scale = std::min(
        static_cast<double>(targetMaxDimension) / original.width,
        static_cast<double>(targetMaxDimension) / original.height
    );

    int newWidth = static_cast<int>(original.width * scale);
    int newHeight = static_cast<int>(original.height * scale);

    Image resized;
    resized.width = newWidth;
    resized.height = newHeight;
    resized.pixels.resize(newHeight, std::vector<unsigned char>(newWidth, 0));

    // Simple nearest neighbor resizing
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int sourceX = static_cast<int>(x / scale);
            int sourceY = static_cast<int>(y / scale);

            sourceX = std::min(sourceX, original.width - 1);
            sourceY = std::min(sourceY, original.height - 1);

            resized.pixels[y][x] = original.pixels[sourceY][sourceX];
        }
    }

    return resized;
}

// Apply Sobel edge detection
Image detectEdges(const Image& image) {
    Image edges;
    edges.width = image.width;
    edges.height = image.height;
    edges.pixels.resize(image.height, std::vector<unsigned char>(image.width, 0));

    // Sobel operators for x and y directions
    const int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    const int sobel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    // Calculate gradient magnitude
    for (int y = 1; y < image.height - 1; ++y) {
        for (int x = 1; x < image.width - 1; ++x) {
            int gx = 0, gy = 0;

            // Apply Sobel operators
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int pixel = image.pixels[y + dy][x + dx];
                    gx += pixel * sobel_x[dy + 1][dx + 1];
                    gy += pixel * sobel_y[dy + 1][dx + 1];
                }
            }
            // Calculate gradient magnitude
            int magnitude = static_cast<int>(sqrt(gx * gx + gy * gy));
            edges.pixels[y][x] = std::min(255, magnitude);
        }
    }
    return edges;
}

// Apply thresholding to create a binary edge image
Image thresholdImage(const Image& image, unsigned char threshold) {
    Image result;
    result.width = image.width;
    result.height = image.height;
    result.pixels.resize(image.height, std::vector<unsigned char>(image.width, 0));

    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            result.pixels[y][x] = (image.pixels[y][x] > threshold) ? 255 : 0;
        }
    }

    return result;
}

// Hough Transform for circle detection
std::vector<Circle> detectCircles(const Image& edges, int minRadius, int maxRadius, int threshold, int angleStep = 1) {
    // Create accumulator for Hough space
    const int radiusCount = maxRadius - minRadius + 1;
    std::vector<std::vector<std::vector<int>>> accumulator(
        edges.height,
        std::vector<std::vector<int>>(
            edges.width,
            std::vector<int>(radiusCount, 0)
        )
    );

    // Fill the accumulator
    for (int y = 0; y < edges.height; ++y) {
        for (int x = 0; x < edges.width; ++x) {
            // Only process edge pixels
            if (edges.pixels[y][x] > 0) {
                // For each potential radius
                for (int r_idx = 0; r_idx < radiusCount; ++r_idx) {
                    int radius = minRadius + r_idx;

                    // Vote for all possible circle centers
                    for (int theta = 0; theta < 360; theta += angleStep) {
                        double radian = theta * M_PI / 180.0;
                        int a = x - static_cast<int>(radius * cos(radian));
                        int b = y - static_cast<int>(radius * sin(radian));

                        if (a >= 0 && a < edges.width && b >= 0 && b < edges.height) {
                            accumulator[b][a][r_idx]++;
                        }
                    }
                }
            }
        }
    }

    // Find peaks in the accumulator
    std::vector<Circle> circles;
    for (int y = 0; y < edges.height; ++y) {
        for (int x = 0; x < edges.width; ++x) {
            for (int r_idx = 0; r_idx < radiusCount; ++r_idx) {
                int votes = accumulator[y][x][r_idx];

                if (votes > threshold) {
                    // Check if it's a local maximum
                    bool isLocalMax = true;

                    // Define the neighborhood for local maximum check
                    const int neighborhoodSize = 5;

                    for (int ny = std::max(0, y - neighborhoodSize); ny <= std::min(edges.height - 1, y + neighborhoodSize) && isLocalMax; ++ny) {
                        for (int nx = std::max(0, x - neighborhoodSize); nx <= std::min(edges.width - 1, x + neighborhoodSize) && isLocalMax; ++nx) {
                            for (int nr_idx = std::max(0, r_idx - 2); nr_idx <= std::min(radiusCount - 1, r_idx + 2) && isLocalMax; ++nr_idx) {
                                if (ny == y && nx == x && nr_idx == r_idx) continue;

                                if (accumulator[ny][nx][nr_idx] > votes) {
                                    isLocalMax = false;
                                }
                            }
                        }
                    }

                    if (isLocalMax) {
                        Circle circle;
                        circle.x = x;
                        circle.y = y;
                        circle.radius = minRadius + r_idx;
                        circle.votes = votes;
                        circles.push_back(circle);
                    }
                }
            }
        }
    }

    // Sort circles by number of votes (highest first)
    std::sort(circles.begin(), circles.end(),
        [](const Circle& a, const Circle& b) {
            return a.votes > b.votes;
        });

    return circles;
}

// Function to extract filename without extension
std::string getFilenameWithoutExtension(const std::string& path) {
    size_t lastSlash = path.find_last_of("/\\");
    size_t lastDot = path.find_last_of(".");

    // Extract filename without path
    std::string filename;
    if (lastSlash != std::string::npos) {
        filename = path.substr(lastSlash + 1);
    }
    else {
        filename = path;
    }

    // Remove extension if present
    if (lastDot != std::string::npos && (lastSlash == std::string::npos || lastDot > lastSlash)) {
        filename = filename.substr(0, lastDot - (lastSlash != std::string::npos ? lastSlash + 1 : 0));
    }

    return filename;
}

// Function to generate output filename
std::string generateOutputFilename(const std::string& inputFilename, const std::string& suffix) {
    std::string baseFilename = getFilenameWithoutExtension(inputFilename);
    std::string extension = ".ppm";
    return baseFilename + "_" + suffix + extension;
}

int main(int argc, char* argv[]) {
    // Option 1: Hardcoded image path 
    std::string inputFilename = "C:/Users/esran/Desktop/dama.ppm";
    std::string outputFilename = "detected_circles.ppm";

    //Command line arguments 
    if (argc > 1) {
        inputFilename = argv[1];
        if (argc > 2) {
            outputFilename = argv[2];
        }
    }

    // Load image
    Image originalImage = loadPPM(inputFilename);
    if (originalImage.width == 0 || originalImage.height == 0) {
        std::cerr << "Failed to load image." << std::endl;
        return 1;
    }

    std::cout << "Original image loaded: " << originalImage.width << "x" << originalImage.height << std::endl;

    // Save the grayscale original image
    saveGrayscaleAsPPM(originalImage, generateOutputFilename(inputFilename, "grayscale"));

    // Resize image to speed up processing (max dimension of 600px)
    const int maxDimension = 600;
    Image image = resizeImage(originalImage, maxDimension);
    std::cout << "Resized for processing: " << image.width << "x" << image.height << std::endl;

    // Save the resized image
    saveGrayscaleAsPPM(image, generateOutputFilename(inputFilename, "resized"));

    // Calculate sampling ratio for scaling detected circles back to original size
    double scaleRatio = static_cast<double>(originalImage.width) / image.width;

    // Detect edges
    Image edges = detectEdges(image);

    // Save the edge detection result
    saveGrayscaleAsPPM(edges, generateOutputFilename(inputFilename, "edges"));

    // Apply threshold to get binary edge image
    Image binaryEdges = thresholdImage(edges, 50);

    // Save the thresholded binary edge image
    saveGrayscaleAsPPM(binaryEdges, generateOutputFilename(inputFilename, "binary_edges"));

    // Parameters for circle detection
    int minRadius = 10;
    int maxRadius = std::min(image.width, image.height) / 4;
    int threshold = 50;  // Minimum number of votes to consider a circle

    // Reduce the angular step to improve performance
    int angleStep = 5;  // Check every 5 degrees instead of every 1 degree

    std::cout << "Detecting circles with radius between " << minRadius << " and " << maxRadius << std::endl;
    std::cout << "Processing... (this may take a moment)" << std::endl;

    // Detect circles with optimized angle step
    std::vector<Circle> circles = detectCircles(binaryEdges, minRadius, maxRadius, threshold, angleStep);

    std::cout << "Detected " << circles.size() << " circles." << std::endl;

    // Take the top N circles to avoid too many false positives
    const int maxCircles = 200;
    if (circles.size() > maxCircles) {
        circles.resize(maxCircles);
 

    // Save image with circles in resized dimensions
    saveImageWithCircles(image, circles, generateOutputFilename(inputFilename, "circles_resized"));

    // Scale circles back to original image size
    std::vector<Circle> originalSizeCircles;
    for (const auto& circle : circles) {
        Circle scaledCircle;
        scaledCircle.x = static_cast<int>(circle.x * scaleRatio);
        scaledCircle.y = static_cast<int>(circle.y * scaleRatio);
        scaledCircle.radius = static_cast<int>(circle.radius * scaleRatio);
        scaledCircle.votes = circle.votes;
        originalSizeCircles.push_back(scaledCircle);
    }

    // Print detected circles
    for (size_t i = 0; i < originalSizeCircles.size(); ++i) {
        std::cout << "Circle " << (i + 1) << ": center=(" << originalSizeCircles[i].x << ", " << originalSizeCircles[i].y
            << "), radius=" << originalSizeCircles[i].radius << ", votes=" << originalSizeCircles[i].votes << std::endl;
    }

    // Save output image with detected circles
    saveImageWithCircles(originalImage, originalSizeCircles, outputFilename);
    std::cout << "Final output image saved to: " << outputFilename << std::endl;

    return 0;
}