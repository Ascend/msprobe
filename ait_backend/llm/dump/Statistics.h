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


#ifndef STATISTICS_H
#define STATISTICS_H

#include <limits>
#include <string>
#include <complex>
#include <cmath>
#include <limits>

namespace LLM {
// helper Definations for Calculating the needed Statistics
class StatisticsBase {
public:
    virtual ~StatisticsBase() = default;
    virtual std::string GetMaxStr() const = 0;
    virtual std::string GetMinStr() const = 0;
    virtual std::string GetMeanStr() const = 0;
    virtual std::string GetL2NormStr() const = 0;
};

template<typename T>
class Statistics : public StatisticsBase {
public:
    T maxValue_;
    T minValue_;
    double average_;
    double l2norm_;
    double sumValue_;
    double sumOfSquares_;
    size_t count_;

    Statistics() : maxValue_(std::numeric_limits<T>::lowest()),
                   minValue_(std::numeric_limits<T>::max()),
                   average_(0.0),
                   l2norm_(0.0),
                   sumValue_(0),
                   sumOfSquares_(0),
                   count_(0) {}

    // Combine statistics from two different instances
    void operator+=(const Statistics& other)
    {
        if (other.count_ != 0) {
            maxValue_ = std::max(maxValue_, other.maxValue_);
            minValue_ = std::min(minValue_, other.minValue_);
            sumValue_ += other.sumValue_;
            sumOfSquares_ += other.sumOfSquares_; // Assuming L2 norm is squared and will be sqrt'd at the end
            count_ += other.count_;
        }
    }

    void Compute(T value)
    {
        maxValue_ = std::max(maxValue_, value);
        minValue_ = std::min(minValue_, value);
        sumValue_ += value;
        sumOfSquares_ += value * value;
        ++count_;
    }

    void ComputeAverage()
    {
        average_ = (count_ > 0) ? (sumValue_ / count_) : 0.0;
    }

    std::string GetMaxStr() const override
    {
        return std::to_string(maxValue_);
    }

    std::string GetMinStr() const override
    {
        return std::to_string(minValue_);
    }

    std::string GetMeanStr() const override
    {
        return std::to_string(average_);
    }

    std::string GetL2NormStr() const override
    {
        return std::to_string(l2norm_);
    }
};

template<>
class Statistics<std::complex<float>> : public StatisticsBase {
public:
    std::complex<float> maxValue_;
    std::complex<float> minValue_;
    std::complex<double> average_;
    double l2norm_;
    std::complex<double> sumValue_;
    double sumOfSquares_;
    size_t count_;

    Statistics() : maxValue_(std::complex<float>(0, 0)),
                   // Since the complex number compares the length of the module when comparing the size,
                   // maxValue_ is initialized to the minimum modular length 0
                   minValue_(std::complex<float>(std::numeric_limits<float>::max(),
                                                 std::numeric_limits<float>::max())),
                   average_(std::complex<double>(0, 0)),
                   l2norm_(0),
                   sumValue_(std::complex<double>(0, 0)),
                   sumOfSquares_(0),
                   count_(0) {}

    // Combine statistics from two different instances
    void operator+=(const Statistics& other)
    {
        if (other.count_ != 0) {
            // Compare based on the magnitude (abs) of complex numbers
            double thisMag = std::norm(maxValue_);
            double otherMag = std::norm(other.maxValue_);
            maxValue_ = (thisMag > otherMag) ? maxValue_ : other.maxValue_;

            thisMag = std::norm(minValue_);
            otherMag = std::norm(other.minValue_);
            minValue_ = (thisMag < otherMag) ? minValue_ : other.minValue_;

            sumValue_ += other.sumValue_;
            sumOfSquares_ += other.sumOfSquares_;
            count_ += other.count_;
        }
    }

    void Compute(const std::complex<float>& value)
    {
        float valMag = std::norm(value);
        if (count_ == 0 || valMag > std::norm(maxValue_)) {
            maxValue_ = value;
        }
        if (count_ == 0 || valMag < std::norm(minValue_)) {
            minValue_ = value;
        }
        sumValue_ += value;
        sumOfSquares_ += std::norm(value);
        count_++;
    }

    void ComputeAverage()
    {
        bool notEmpty = count_ > 0;
        average_ = std::complex<double>(notEmpty ? sumValue_.real() / (static_cast<double>(count_)) : 0,
                                        notEmpty ? sumValue_.imag() / (static_cast<double>(count_)) : 0);
    }

    std::string GetMaxStr() const override
    {
        return "(" + std::to_string(maxValue_.real()) + "," + std::to_string(maxValue_.imag()) + ")";
    }

    std::string GetMinStr() const override
    {
        return "(" + std::to_string(minValue_.real()) + "," + std::to_string(minValue_.imag()) + ")";
    }

    std::string GetMeanStr() const override
    {
        return "(" + std::to_string(average_.real()) + "," + std::to_string(average_.imag()) + ")";
    }

    std::string GetL2NormStr() const override
    {
        return std::to_string(l2norm_);
    }
};

template<>
class Statistics<std::string> : public StatisticsBase {
public:
    std::string maxValue_;
    std::string minValue_;
    std::string average_;
    std::string l2norm_;

    Statistics() : maxValue_("N/A"),
                   minValue_("N/A"),
                   average_("N/A"),
                   l2norm_("N/A") {}

    std::string GetMaxStr() const override
    {
        return maxValue_;
    }

    std::string GetMinStr() const override
    {
        return minValue_;
    }

    std::string GetMeanStr() const override
    {
        return average_;
    }

    std::string GetL2NormStr() const override
    {
        return l2norm_;
    }
};
}
#endif // STATISTICS_H