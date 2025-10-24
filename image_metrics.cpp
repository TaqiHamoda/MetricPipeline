#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>
#include <eigen3/Eigen/Dense>
// #include <Eigen/Dense> // For Eigen linear algebra

// Ensure M_PI is defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper for standard deviation
double calculate_std(const std::vector<double>& v, double mean) {
    if (v.empty()) return 0.0;
    double sq_sum = 0.0;
    for (double d : v) {
        sq_sum += (d - mean) * (d - mean);
    }
    return std::sqrt(sq_sum / v.size());
}

// Helper for mean
double calculate_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

// We must wrap our C++ functions in extern "C" for ctypes to find them
extern "C" {

    /**
     * @brief Calculates the slant angle using Eigen for least squares fitting.
     */
    void calculate_slant_c(
        double* depth_data, int rows, int cols,
        double fx, double fy, double cx, double cy,
        double* out_slant_angle) 
    {
        std::vector<double> X, Y, Z;
        
        // 1. Calculate 3D coordinates and filter non-zero depth
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                double z = depth_data[r * cols + c];
                if (z > 0) {
                    Z.push_back(z);
                    X.push_back(z * (c - cx) / fx);
                    Y.push_back(z * (r - cy) / fy);
                }
            }
        }

        int n = Z.size();
        if (n < 9) { // Not enough points to solve
            *out_slant_angle = 0.0;
            return;
        }

        // 2. Create the design matrix A and vector b (which is Z)
        Eigen::MatrixXd A(n, 9);
        Eigen::VectorXd b(n);

        for (int i = 0; i < n; ++i) {
            double x = X[i];
            double y = Y[i];
            
            A(i, 0) = 1.0;
            A(i, 1) = x;
            A(i, 2) = y;
            A(i, 3) = x * x;
            A(i, 4) = x * y;
            A(i, 5) = y * y;
            A(i, 6) = x * x * y;
            A(i, 7) = x * y * y;
            A(i, 8) = y * y * y;
            
            b(i) = Z[i];
        }

        // 3. Perform least squares fitting: A * coeffs = b
        // We use ColPivHouseholderQr which is robust and good for overdetermined systems.
        Eigen::VectorXd coeffs = A.colPivHouseholderQr().solve(b);

        if (coeffs.size() != 9) {
             *out_slant_angle = 0.0; // Solve failed
             return;
        }

        // 4. Calculate angle
        double n_x = coeffs(1);
        double n_y = coeffs(2);
        double n_z = -1.0;

        double normal_magnitude = std::sqrt(n_x * n_x + n_y * n_y + n_z * n_z);
        *out_slant_angle = std::acos(n_z / (normal_magnitude + 1e-6)) * 180.0 / M_PI;
    }

    /**
     * @brief Calculates UCIQE from a pre-converted LAB image.
     */
    void calculate_uciqe_c(
        unsigned char* lab_data, int rows, int cols,
        double* out_uciqe)
    {
        int n = rows * cols;
        if (n == 0) {
            *out_uciqe = 0.0;
            return;
        }

        std::vector<double> l_channel(n);
        std::vector<double> a_channel(n);
        std::vector<double> b_channel(n);
        std::vector<double> chroma(n);
        double chroma_sum = 0.0;
        double satur_sum = 0.0;

        for (int i = 0; i < n; ++i) {
            l_channel[i] = static_cast<double>(lab_data[i * 3 + 0]);
            a_channel[i] = static_cast<double>(lab_data[i * 3 + 1]);
            b_channel[i] = static_cast<double>(lab_data[i * 3 + 2]);
            
            // Term 1 (sc) and Term 3 (us) part 1
            chroma[i] = std::sqrt(a_channel[i] * a_channel[i] + b_channel[i] * b_channel[i]);
            chroma_sum += chroma[i];
            
            if (l_channel[i] != 0) {
                satur_sum += chroma[i] / l_channel[i];
            }
        }

        // --- 1st Term: sc (Chroma standard deviation) ---
        double chroma_mean = chroma_sum / n;
        double sc = calculate_std(chroma, chroma_mean);

        // --- 2nd Term: conl (Luminance contrast) ---
        int top_k = static_cast<int>(0.01 * n);
        if (top_k == 0) top_k = 1; // Avoid division by zero for tiny images

        std::vector<double> sl = l_channel; // Copy for sorting
        std::partial_sort(sl.begin(), sl.begin() + top_k, sl.end());
        double bottom_sum = 0.0;
        for(int i = 0; i < top_k; ++i) bottom_sum += sl[i];

        std::partial_sort(sl.begin(), sl.begin() + top_k, sl.end(), std::greater<double>());
        double top_sum = 0.0;
        for(int i = 0; i < top_k; ++i) top_sum += sl[i];
        
        double conl = (top_sum / top_k) - (bottom_sum / top_k);

        // --- 3rd Term: us (Saturation) ---
        double satur = satur_sum / n;

        // --- Final Calculation ---
        *out_uciqe = 0.4680 * sc + 0.2745 * conl + 0.2576 * satur;
    }

    /**
     * @brief Calculates EME or LogAMEE for a single channel.
     */
    void calculate_channel_eme_c(
        unsigned char* ch_data, int rows, int cols, 
        int blocksize, bool is_logamee, 
        double gamma, double k,
        double* out_eme)
    {
        int num_x = static_cast<int>(std::ceil(static_cast<double>(rows) / blocksize));
        int num_y = static_cast<int>(std::ceil(static_cast<double>(cols) / blocksize));

        if (num_x == 0 || num_y == 0) {
            *out_eme = 0.0;
            return;
        }

        double eme_sum = 0.0;
        double w = 1.0 / (num_x * num_y);

        for (int i = 0; i < num_x; ++i) {
            int xlb = i * blocksize;
            int xrb = std::min((i + 1) * blocksize, rows);

            for (int j = 0; j < num_y; ++j) {
                int ylb = j * blocksize;
                int yrb = std::min((j + 1) * blocksize, cols);

                // Find min/max in block
                unsigned char blockmin = 255;
                unsigned char blockmax = 0;
                for (int r = xlb; r < xrb; ++r) {
                    for (int c = ylb; c < yrb; ++c) {
                        unsigned char val = ch_data[r * cols + c];
                        if (val < blockmin) blockmin = val;
                        if (val > blockmax) blockmax = val;
                    }
                }

                double fmin = static_cast<double>(blockmin);
                double fmax = static_cast<double>(blockmax);

                if (is_logamee) {
                    // Avoid division by zero
                    if (k - fmin == 0) continue;
                    
                    double top = k * (fmax - fmin) / (k - fmin);
                    double bottom = fmax + fmin - fmax * fmin / gamma;

                    if (bottom == 0) continue;
                    double m = top / bottom;

                    if (m != 0) {
                        eme_sum += m * std::log(m);
                    }
                } else {
                    if (fmin == 0) fmin = 1;
                    if (fmax == 0) fmax = 1; // Though fmax >= fmin, good to be safe
                    eme_sum += 2 * w * std::log(fmax / fmin);
                }
            }
        }

        if (is_logamee) {
            *out_eme = gamma - gamma * std::pow(1.0 - eme_sum / gamma, w);
        } else {
            *out_eme = eme_sum;
        }
    }

    /**
     * @brief Calculates the UICM part of the UIQM metric.
     * Takes pre-sorted and pre-trimmed arrays.
     */
    void calculate_uicm_c(
        double* rgl_trimmed, double* ybl_trimmed, int n, 
        double* out_uicm)
    {
        if (n == 0) {
            *out_uicm = 0.0;
            return;
        }
        
        std::vector<double> rgl(rgl_trimmed, rgl_trimmed + n);
        std::vector<double> ybl(ybl_trimmed, ybl_trimmed + n);

        // Calculate means
        double urg = calculate_mean(rgl);
        double uyb = calculate_mean(ybl);
        
        // Calculate variances (from mean)
        double s2rg = calculate_std(rgl, urg) * calculate_std(rgl, urg); // s^2
        double s2yb = calculate_std(ybl, uyb) * calculate_std(ybl, uyb); // s^2

        // Final calc
        double norm = std::sqrt(urg * urg + uyb * uyb);
        double s_sum_sqrt = std::sqrt(s2rg + s2yb);
        *out_uicm = -0.0268 * norm + 0.1586 * s_sum_sqrt;
    }
} // extern "C"