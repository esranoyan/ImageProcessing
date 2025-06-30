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

// Structure to represent a line
struct Line {
    double rho;    // distance from origin
    double theta;  // angle in radians
    int votes;     // number of votes (for Hough Transform)

    // Helper function to get line endpoints
    std::pair<std::pair<int, int>, std::pair<int, int>> getEndpoints(int imgWidth, int imgHeight) const {
        double cosTheta = cos(theta);
        double sinTheta = sin(theta);

        int x1 = 0, y1 = 0, x2 = 0, y2 = 0;

        if (abs(sinTheta) > 0.001) { // Not vertical line
            x1 = 0;
            y1 = (int)(rho / sinTheta);
            x2 = imgWidth - 1;
            y2 = (int)((rho - x2 * cosTheta) / sinTheta);
        }
        else { // Vertical line
            x1 = x2 = (int)(rho / cosTheta);
            y1 = 0;
            y2 = imgHeight - 1;
        }

        // Clip to image boundaries
        if (y1 < 0) { y1 = 0; x1 = (int)((rho - y1 * sinTheta) / cosTheta); }
        if (y1 >= imgHeight) { y1 = imgHeight - 1; x1 = (int)((rho - y1 * sinTheta) / cosTheta); }
        if (y2 < 0) { y2 = 0; x2 = (int)((rho - y2 * sinTheta) / cosTheta); }
        if (y2 >= imgHeight) { y2 = imgHeight - 1; x2 = (int)((rho - y2 * sinTheta) / cosTheta); }

        if (x1 < 0) { x1 = 0; y1 = (int)((rho - x1 * cosTheta) / sinTheta); }
        if (x1 >= imgWidth) { x1 = imgWidth - 1; y1 = (int)((rho - x1 * cosTheta) / sinTheta); }
        if (x2 < 0) { x2 = 0; y2 = (int)((rho - x2 * cosTheta) / sinTheta); }
        if (x2 >= imgWidth) { x2 = imgWidth - 1; y2 = (int)((rho - x2 * cosTheta) / sinTheta); }

        return { {x1, y1}, {x2, y2} };
    }
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

// Function to save an image with detected lines
void saveImageWithLines(const Image& image, const std::vector<Line>& lines,
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

    // Draw lines in red
    for (const auto& line : lines) {
        auto endpoints = line.getEndpoints(image.width, image.height);
        int x1 = endpoints.first.first;
        int y1 = endpoints.first.second;
        int x2 = endpoints.second.first;
        int y2 = endpoints.second.second;

        // Bresenham's line algorithm
        int dx = abs(x2 - x1);
        int dy = abs(y2 - y1);
        int sx = x1 < x2 ? 1 : -1;
        int sy = y1 < y2 ? 1 : -1;
        int err = dx - dy;

        int x = x1, y = y1;

        while (true) {
            if (x >= 0 && x < image.width && y >= 0 && y < image.height) {
                int idx = (y * image.width + x) * 3;
                output[idx] = 255;     // Red channel
                output[idx + 1] = 0;   // Green channel
                output[idx + 2] = 0;   // Blue channel
            }
            
            if (x == x2 && y == y2) break;

            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x += sx;
            }
            if (e2 < dx) {
                err += dx;
                y += sy;
            }
        }
    }

    // Write PPM header
    file << "P6\n" << image.width << " " << image.height << "\n255\n";

    // Write pixel data
    file.write(reinterpret_cast<const char*>(output.data()), output.size());

    std::cout << "Image with lines saved to: " << filename << std::endl;
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

// Apply Gaussian smoothing
Image gaussianSmooth(const Image& image, double sigma = 1.0) {
    Image smoothed;
    smoothed.width = image.width;
    smoothed.height = image.height;
    smoothed.pixels.resize(image.height, std::vector<unsigned char>(image.width, 0));

    int kernelSize = (int)(6 * sigma) | 1; // Ensure odd number
    std::vector<double> kernel(kernelSize);
    double sum = 0;
    int center = kernelSize / 2;

    // Create Gaussian kernel
    for (int i = 0; i < kernelSize; i++) {
        double x = i - center;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize kernel
    for (int i = 0; i < kernelSize; i++) {
        kernel[i] /= sum;
    }

    // Horizontal convolution
    std::vector<std::vector<double>> temp(image.height, std::vector<double>(image.width, 0));
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            double sum = 0;
            for (int k = 0; k < kernelSize; k++) {
                int idx = x + k - center;
                if (idx >= 0 && idx < image.width) {
                    sum += image.pixels[y][idx] * kernel[k];
                }
            }
            temp[y][x] = sum;
        }
    }

    // Vertical convolution
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            double sum = 0;
            for (int k = 0; k < kernelSize; k++) {
                int idx = y + k - center;
                if (idx >= 0 && idx < image.height) {
                    sum += temp[idx][x] * kernel[k];
                }
            }
            smoothed.pixels[y][x] = (unsigned char)std::min(255.0, std::max(0.0, sum));
        }
    }

    return smoothed;
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

// Hough Transform for line detection
std::vector<Line> detectLines(const Image& edges, int threshold, double thetaStep = 1.0) {
    // Parameters for Hough space
    int maxRho = (int)(sqrt(edges.width * edges.width + edges.height * edges.height));
    int thetaSteps = (int)(180.0 / thetaStep);

    // Create accumulator
    std::vector<std::vector<int>> accumulator(2 * maxRho, std::vector<int>(thetaSteps, 0));

    // Fill accumulator
    for (int y = 0; y < edges.height; ++y) {
        for (int x = 0; x < edges.width; ++x) {
            if (edges.pixels[y][x] > 0) { // Edge pixel
                for (int thetaIdx = 0; thetaIdx < thetaSteps; ++thetaIdx) {
                    double theta = (thetaIdx * thetaStep) * M_PI / 180.0;
                    double rho = x * cos(theta) + y * sin(theta);
                    int rhoIdx = (int)(rho + maxRho);

                    if (rhoIdx >= 0 && rhoIdx < 2 * maxRho) {
                        accumulator[rhoIdx][thetaIdx]++;
                    }
                }
            }
        }
    }

    // Find peaks in accumulator
    std::vector<Line> lines;
    for (int rhoIdx = 0; rhoIdx < 2 * maxRho; ++rhoIdx) {
        for (int thetaIdx = 0; thetaIdx < thetaSteps; ++thetaIdx) {
            int votes = accumulator[rhoIdx][thetaIdx];

            if (votes > threshold) {
                // Check if it's a local maximum
                bool isLocalMax = true;
                const int neighborhoodSize = 3;

                for (int dr = -neighborhoodSize; dr <= neighborhoodSize && isLocalMax; ++dr) {
                    for (int dt = -neighborhoodSize; dt <= neighborhoodSize && isLocalMax; ++dt) {
                        int checkRho = rhoIdx + dr;
                        int checkTheta = thetaIdx + dt;

                        if (checkRho >= 0 && checkRho < 2 * maxRho &&
                            checkTheta >= 0 && checkTheta < thetaSteps) {
                            if (checkRho == rhoIdx && checkTheta == thetaIdx) continue;

                            if (accumulator[checkRho][checkTheta] > votes) {
                                isLocalMax = false;
                            }
                        }
                    }
                }

                if (isLocalMax) {
                    Line line;
                    line.rho = rhoIdx - maxRho;
                    line.theta = (thetaIdx * thetaStep) * M_PI / 180.0;
                    line.votes = votes;
                    lines.push_back(line);
                }
            }
        }
    }

    // Sort lines by votes (highest first)
    std::sort(lines.begin(), lines.end(),
        [](const Line& a, const Line& b) {
            return a.votes > b.votes;
        });

    return lines;
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
    std::string inputFilename = "C:/Users/esran/Desktop/high_road.ppm";
    std::string outputFilename = "detected_lines.ppm";

    // Command line arguments 
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

    // Calculate sampling ratio for scaling detected lines back to original size
    double scaleRatio = static_cast<double>(originalImage.width) / image.width;

    // Apply Gaussian smoothing
    Image smoothed = gaussianSmooth(image, 1.0);
    saveGrayscaleAsPPM(smoothed, generateOutputFilename(inputFilename, "smoothed"));

    // Detect edges
    Image edges = detectEdges(smoothed);
    saveGrayscaleAsPPM(edges, generateOutputFilename(inputFilename, "edges"));

    // Apply threshold to get binary edge image
    Image binaryEdges = thresholdImage(edges, 50);
    saveGrayscaleAsPPM(binaryEdges, generateOutputFilename(inputFilename, "binary_edges"));

    // Parameters for line detection
    int threshold = 80;  // Minimum number of votes to consider a line
    double thetaStep = 1.0; // Angular resolution in degrees

    std::cout << "Detecting lines..." << std::endl;
    std::cout << "Processing... (this may take a moment)" << std::endl;

    // Detect lines
    std::vector<Line> lines = detectLines(binaryEdges, threshold, thetaStep);

    std::cout << "Detected " << lines.size() << " lines." << std::endl;

    // Take the top N lines to avoid too many false positives
    const int maxLines = 20;
    if (lines.size() > maxLines) {
        lines.resize(maxLines);
    }

    // Save image with lines in resized dimensions
    saveImageWithLines(image, lines, generateOutputFilename(inputFilename, "lines_resized"));

    // Scale lines back to original image size
    std::vector<Line> originalSizeLines;
    for (const auto& line : lines) {
        Line scaledLine;
        scaledLine.rho = line.rho * scaleRatio;
        scaledLine.theta = line.theta;
        scaledLine.votes = line.votes;
        originalSizeLines.push_back(scaledLine);
    }

    // Print detected lines
    for (size_t i = 0; i < originalSizeLines.size(); ++i) {
        double thetaDegrees = originalSizeLines[i].theta * 180.0 / M_PI;
        std::cout << "Line " << (i + 1) << ": rho=" << originalSizeLines[i].rho
            << ", theta=" << thetaDegrees << " degrees"
            << ", votes=" << originalSizeLines[i].votes << std::endl;
    }

    // Save output image with detected lines
    saveImageWithLines(originalImage, originalSizeLines, outputFilename);
    std::cout << "Final output image saved to: " << outputFilename << std::endl;

    return 0;
}

