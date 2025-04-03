#include "simd/feature_check.hpp"
#include <iomanip>
#include <iostream>

int main()
{
    std::cout << "===== Supported Features =====" << std::endl;

    std::vector<std::string> features = simd::get_supported_features();

    std::cout << "Found " << features.size()
              << " supported features:" << std::endl;

    const int columns = 3;
    for (size_t i = 0; i < features.size(); ++i)
    {
        std::cout << std::setw(20) << std::left << features[i];

        if ((i + 1) % columns == 0 || i == features.size() - 1)
        {
            std::cout << std::endl;
        }
    }

    std::cout << std::endl;
}
