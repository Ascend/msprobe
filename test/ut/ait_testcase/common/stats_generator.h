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

#ifndef STATS_GENERATOR_H
#define STATS_GENERATOR_H

#include <vector>
#include <thread>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
#include <random>
#include <map>
#include <complex>
#include <unordered_map>
#include "atb_probe.h"
#include "Statistics.h"


uint16_t Float32ToFloat16Binary(float value);
uint16_t Float32ToBFloat16Binary(float value);
std::vector<uint16_t> GenerateVectorHalfPrecFloats(size_t dataSize, Mki::TensorDType dtype);
std::vector<std::complex<float>> GenerateVectorComplex64(size_t numComplexes);
std::unique_ptr<LLM::StatisticsBase> CalStatsHalfPrec(std::vector<uint16_t>& random_nums, Mki::TensorDType dtype);
std::unique_ptr<LLM::StatisticsBase> CalStatsComplex64(std::vector<std::complex<float>>& random_nums,
                                                       uint8_t decimalPlaces);

template<typename T>
std::vector<T> GenerateVectorNorm(size_t dataSize)
{
    std::vector<T> random_values(dataSize);

    // 使用随机设备获取种子
    std::random_device rd;
    std::mt19937 gen(rd());

    // 根据类型 T 选择合适的分布
    // 此处假设 T 是浮点数或可以隐式转换为浮点数的类型
    auto dis = std::uniform_real_distribution<
            typename std::conditional<
                    std::is_floating_point<T>::value,
                    T,
                    double // 如果 T 不是浮点数，则使用 double 作为中间类型
            >::type
    >(-1000.0, 1000.0);

    // 生成随机数
    for (size_t i = 0; i < dataSize; ++i) {
        random_values[i] = dis(gen);
    }

    return random_values;
}

template<typename T>
std::unique_ptr<LLM::StatisticsBase> CalStatsNorm(std::vector<T>& random_nums)
{
    auto statistics = std::make_unique<LLM::Statistics<std::string>>();
    size_t numSize = random_nums.size();
    double fmax = std::numeric_limits<double>::lowest();
    double fmin = std::numeric_limits<double>::max();
    double fsum = std::accumulate(std::begin(random_nums), std::end(random_nums), 0.0);
    double fnormsqsum = 0.0;
    std::for_each (std::begin(random_nums), std::end(random_nums), [&](const double d) {
        fmax = std::max(fmax, d);
        fmin = std::min(fmin, d);
        fnormsqsum  += d * d;
    });

    statistics->maxValue_ = std::to_string(fmax);
    statistics->minValue_ = std::to_string(fmin);
    statistics->average_ = std::to_string(fsum / numSize);
    statistics->l2norm_ = std::to_string(std::sqrt(fnormsqsum));
    return statistics;
}

#endif // STATS_GENERATOR_H