#pragma once

#include <fstream>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>  // Pour remove_if
#include <cctype> 


struct ObservationData {
    int T;              // Time steps (nombre de jours)
    int K;              // Observation dimension (nombre d'assets)
    float* observations; // Raw pointer [T * K] row-major
    
    ObservationData() : T(0), K(0), observations(nullptr) {}
    
    ~ObservationData() {
        if (observations != nullptr) {
            delete[] observations;
        }
    }
    
    // Disable copy
    ObservationData(const ObservationData&) = delete;
    ObservationData& operator=(const ObservationData&) = delete;
};


// Simple JSON parser for just T and K
bool parse_dimensions(const char* json_path, int& T, int& K) {
    std::ifstream file(json_path);
    if (!file.is_open()) return false;
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("\"T\"") != std::string::npos) {
            size_t colon = line.find(":");
            if (colon != std::string::npos) {
                std::string num = line.substr(colon + 1);
                // Remove spaces, comma
                num.erase(std::remove_if(num.begin(), num.end(), 
                         [](char c) { return !std::isdigit(c); }), num.end());
                T = std::stoi(num);
            }
        }
        if (line.find("\"K\"") != std::string::npos) {
            size_t colon = line.find(":");
            if (colon != std::string::npos) {
                std::string num = line.substr(colon + 1);
                num.erase(std::remove_if(num.begin(), num.end(), 
                         [](char c) { return !std::isdigit(c); }), num.end());
                K = std::stoi(num);
            }
        }
    }
    
    file.close();
    return (T > 0 && K > 0);
}


bool load_observations(ObservationData& data, 
                      const char* bin_path, 
                      const char* json_path) {
    // 1. Parse dimensions from JSON
    if (!parse_dimensions(json_path, data.T, data.K)) {
        std::cerr << "Failed to parse dimensions from " << json_path << std::endl;
        return false;
    }
    
    std::cout << "Loading observations: T=" << data.T 
              << ", K=" << data.K << std::endl;
    
    // 2. Allocate and read binary data
    size_t total_size = data.T * data.K;
    data.observations = new float[total_size];
    
    std::ifstream bin_file(bin_path, std::ios::binary);
    if (!bin_file.is_open()) {
        std::cerr << "Failed to open " << bin_path << std::endl;
        delete[] data.observations;
        data.observations = nullptr;
        return false;
    }
    
    bin_file.read(reinterpret_cast<char*>(data.observations), 
                  total_size * sizeof(float));
    
    if (!bin_file) {
        std::cerr << "Failed to read all data" << std::endl;
        delete[] data.observations;
        data.observations = nullptr;
        return false;
    }
    
    bin_file.close();
    
    std::cout << "✓ Loaded " << total_size << " observations" << std::endl;
    return true;
}


// Helper: access observation at time t for asset k
inline float get_observation(const ObservationData& data, int t, int k) {
    return data.observations[t * data.K + k];
}