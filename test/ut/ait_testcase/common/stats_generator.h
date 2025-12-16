/* -------------------------------------------------------------------------
 *  This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */


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
#include <type_traits>
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
    std::random_device rd;
    std::mt19937 gen(rd());

    if constexpr (std::is_integral<T>::value) {
        using range_type = typename std::conditional<
                            std::is_signed<T>::value,
                            int8_t,
                            uint8_t
                        >::type;
        range_type lower_bound = std::numeric_limits<range_type>::min();
        range_type upper_bound = std::numeric_limits<range_type>::max();

        std::uniform_int_distribution<range_type> dis(lower_bound, upper_bound);
        for (auto& val : random_values) {
            val = static_cast<T>(dis(gen));
        }
    } else {
        // 浮点类型处理（保持原逻辑但自动转换）
        int lower_bound = -1000;
        int upper_bound = 1000;
        using distribution_type = std::uniform_real_distribution<T>;
        distribution_type dis(lower_bound, upper_bound);
        
        for (auto& val : random_values) {
            val = dis(gen);
        }
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