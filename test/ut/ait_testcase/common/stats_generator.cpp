/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <unordered_map>
#include <utility>
#include "tools.h"
#include "securec.h"
#include "stats_generator.h"

uint16_t Float32ToFloat16Binary(float value)
{
    uint32_t floatBits;
    memcpy_s(&floatBits, sizeof(floatBits), &value, sizeof(floatBits));
    uint16_t result = static_cast<uint16_t>(floatBits >> 16);
    return result;
}

uint16_t Float32ToBFloat16Binary(float value)
{
    uint32_t floatBits;
    memcpy_s(&floatBits, sizeof(floatBits), &value, sizeof(floatBits));
    uint16_t sign = (floatBits >> 31) & 0x0001;
    uint16_t exponent = ((floatBits >> 23) & 0x00FF) - 1;
    uint16_t mantissa = (floatBits >> 16) & 0x007F;
    uint16_t bfloat16 = (sign << 15) | (exponent << 7) | mantissa;
    return bfloat16;
}

std::vector<uint16_t> GenerateVectorHalfPrecFloats(size_t dataSize, Mki::TensorDType dtype)
{
    std::vector<uint16_t> halfFloats(dataSize);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f); // 范围

    for (size_t i = 0; i < dataSize; ++i) {
        float randomFloat = dis(gen);
        if (dtype == Mki::TensorDType::TENSOR_DTYPE_FLOAT16) {
            halfFloats[i] = Float32ToFloat16Binary(randomFloat);
        } else if (dtype == Mki::TensorDType::TENSOR_DTYPE_BF16) {
            halfFloats[i] = Float32ToBFloat16Binary(randomFloat);
        }
    }
    return halfFloats;
}

std::unordered_map<Mki::TensorDType, std::pair<int, int>> halfPrecisionMap = {
    {Mki::TensorDType::TENSOR_DTYPE_FLOAT16, std::make_pair(5, 10)},
    {Mki::TensorDType::TENSOR_DTYPE_BF16, std::make_pair(8, 7)}
};

std::unique_ptr<LLM::StatisticsBase> CalStatsHalfPrec(std::vector<uint16_t>& random_nums, Mki::TensorDType dtype)
{
    auto statistics = std::make_unique<LLM::Statistics<std::string>>();
    size_t numSize = random_nums.size();

    int exp = 0;
    int man = 0;
    auto it = halfPrecisionMap.find(dtype);
    if (it != halfPrecisionMap.end()) {
        exp = it->second.first;
        man = it->second.second;
    } else {
        std::cout << "Unsupported dtype: " << Mki::GetDTypeStr(dtype) << std::endl;
        return std::make_unique<LLM::Statistics<std::string>>();
    }

    float fmax = std::numeric_limits<float>::lowest();
    float fmin = std::numeric_limits<float>::max();
    std::vector<float> float_vector(numSize);
    for (auto each : random_nums) {
        float_vector.push_back(Mki::ConvertToFloat32(each, exp, man));
    }

    double fsum = std::accumulate(std::begin(float_vector), std::end(float_vector), 0.0);
    double fnormsqsum = 0.0;
    std::for_each (std::begin(float_vector), std::end(float_vector), [&](const float d) {
        fmax = std::max(fmax, d);
        fmin = std::min(fmin, d);
        fnormsqsum  +=  d * d;
    });

    statistics->maxValue_ = std::to_string(fmax);
    statistics->minValue_ = std::to_string(fmin);
    statistics->average_ = std::to_string(fsum / numSize);
    statistics->l2norm_ = std::to_string(std::sqrt(fnormsqsum));

    return statistics;
}

std::vector<std::complex<float>> GenerateVectorComplex64(size_t numComplexes)
{
    std::vector<std::complex<float>> randomComplexes(numComplexes);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f); // 范围

    // 生成随机复数
    for (auto& c : randomComplexes) {
        c = std::complex<float>(dis(gen), dis(gen));
    }

    return randomComplexes;
}

std::unique_ptr<LLM::StatisticsBase> CalStatsComplex64(std::vector<std::complex<float>>& random_nums,
    uint8_t decimalPlaces)
{
    auto statistics = std::make_unique<LLM::Statistics<std::string>>();
    size_t numSize = random_nums.size();

    auto fmax = std::complex<float>(0, 0);
    auto fmin = std::complex<float>(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    std::complex<double> fsum = std::complex<double>(0, 0);
    std::for_each (std::begin(random_nums), std::end(random_nums), [&](const std::complex<float> d) {
        fsum  += d;
    });
    double fnormsqsum = 0;
    std::for_each (std::begin(random_nums), std::end(random_nums), [&](const std::complex<float> d) {
        fmax = std::norm(fmax) > std::norm(d) ? fmax : d;
        fmin = std::norm(fmin) < std::norm(d) ? fmin : d;
        fnormsqsum  += std::norm(d);
    });

    auto roundComplexStr = [decimalPlaces](std::complex<float> value) -> std::string {
        std::string realStr = RoundStrNum(std::to_string(value.real()), decimalPlaces);
        std::string imagStr = RoundStrNum(std::to_string(value.imag()), decimalPlaces);
        return "(" + realStr + "," + imagStr + ")";
    };

    statistics->maxValue_ = roundComplexStr(fmax);
    statistics->minValue_ = roundComplexStr(fmin);
    std::string realStr = RoundStrNum(std::to_string(fsum.real() / numSize), decimalPlaces);
    std::string imagStr = RoundStrNum(std::to_string(fsum.imag() / numSize), decimalPlaces);
    statistics->average_ = "(" + realStr + "," + imagStr + ")";
    statistics->l2norm_ = std::to_string(std::sqrt(fnormsqsum)); // l2norm为非复数格式

    return statistics;
}