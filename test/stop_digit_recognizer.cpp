#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;

static const char* PID_FILE = ".digit_recognizer.pid";
static const char* STOP_FILE = ".digit_recognizer.stop";

int main() {
    if (!fs::exists(PID_FILE)) {
        std::cerr << "No running digit_recognizer found ("
                  << PID_FILE << " does not exist)." << std::endl;
        return 1;
    }

    std::ifstream pf(PID_FILE);
    int pid = 0;
    pf >> pid;
    pf.close();

    std::cout << "Sending stop signal to digit_recognizer (PID " << pid << ")..." << std::endl;

    {
        std::ofstream sf(STOP_FILE);
        sf << "stop" << std::endl;
    }

    // Wait for digit_recognizer to clean up and remove the PID file
    int wait_seconds = 30;
    for (int i = 0; i < wait_seconds; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (!fs::exists(PID_FILE)) {
            std::cout << "digit_recognizer stopped successfully." << std::endl;
            // Clean up stop file in case it wasn't removed
            std::error_code ec;
            fs::remove(STOP_FILE, ec);
            return 0;
        }
    }

    std::cerr << "Warning: digit_recognizer did not exit within "
              << wait_seconds << " seconds." << std::endl;
    std::cerr << "You may need to kill PID " << pid << " manually." << std::endl;

    // Clean up signal files
    std::error_code ec;
    fs::remove(STOP_FILE, ec);
    fs::remove(PID_FILE, ec);
    return 1;
}
