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

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "torch_npu/csrc/npu/Stream.h"

namespace
{
namespace py = pybind11;

static std::atomic<uint64_t> serial_num{0};
static constexpr const char* kForwardStartMarker = "__msprobe_fwd_start__";
static constexpr const char* kAclRuntimeInitError =
    "ACL runtime not initialized (no current context). Ensure NPU backend is initialized before calling acl_save.";
static constexpr const char* kAclSaveLogPrefix = "[acl_save_debug]";

struct SaveTaskPayload
{
    SaveTaskPayload(at::Tensor tensor, std::string save_path)
        : tensor(std::move(tensor)), save_path(std::move(save_path))
    {
    }

    at::Tensor tensor;
    std::string save_path;
    std::atomic<bool> executed{false};
};

struct StatTaskPayload
{
    StatTaskPayload(at::Tensor stats_tensor, std::string tag, std::string dtype, std::vector<int64_t> shape)
        : stats_tensor(std::move(stats_tensor)), tag(std::move(tag)), dtype(std::move(dtype)), shape(std::move(shape))
    {
    }

    at::Tensor stats_tensor;
    std::string tag;
    std::string dtype;
    std::vector<int64_t> shape;
};

struct StatRecord
{
    std::string dtype;
    std::vector<int64_t> shape;
    double min{0.0};
    double max{0.0};
    double mean{0.0};
    double norm{0.0};
};

static std::mutex g_stats_mutex;
static std::unordered_map<std::string, uint64_t> g_tag_counter;
static std::unordered_map<std::string, uint64_t> g_current_forward_idx;
static std::unordered_map<std::string, StatRecord> g_stat_entries;
static std::vector<std::string> g_stat_entry_order;

static void check_acl(aclError err, const char* msg)
{
    if (err != ACL_ERROR_NONE)
    {
        std::ostringstream oss;
        oss << msg << " (aclError=" << static_cast<int>(err) << ")";
        throw std::runtime_error(oss.str());
    }
}

static std::string build_final_path(const std::string& path, uint64_t seq)
{
    size_t last_slash = path.find_last_of("/\\");
    std::string filename = (last_slash == std::string::npos) ? path : path.substr(last_slash + 1);
    size_t dot_pos = filename.find_last_of('.');
    std::string base = (dot_pos == std::string::npos) ? filename : filename.substr(0, dot_pos);
    std::ostringstream oss_name;
    oss_name << base << "_" << seq << ".pt";
    if (last_slash == std::string::npos)
    {
        return oss_name.str();
    }
    return path.substr(0, last_slash + 1) + oss_name.str();
}

struct IoSegment
{
    size_t pos{std::string::npos};
    std::string type;
    std::string suffix;
};

static IoSegment ParseIoSegmentFromTag(const std::string& tag)
{
    static const std::vector<std::string> ioTypes = {"input_kwargs", "input", "output"};
    IoSegment best;

    for (const auto& ioType : ioTypes)
    {
        const std::string midMarker = "." + ioType + ".";
        const size_t midPos = tag.rfind(midMarker);
        if (midPos != std::string::npos && (best.pos == std::string::npos || midPos > best.pos))
        {
            best.pos = midPos;
            best.type = ioType;
            best.suffix = tag.substr(midPos + midMarker.size());
        }

        const std::string tailMarker = "." + ioType;
        const size_t tailPos = tag.rfind(tailMarker);
        if (tailPos != std::string::npos && tailPos + tailMarker.size() == tag.size() &&
            (best.pos == std::string::npos || tailPos > best.pos))
        {
            best.pos = tailPos;
            best.type = ioType;
            best.suffix.clear();
        }
    }

    return best;
}

static std::string build_stat_key(const std::string& tag)
{
    if (tag.empty())
    {
        return "__default__";
    }

    const IoSegment io = ParseIoSegmentFromTag(tag);
    if (io.pos == std::string::npos || io.type.empty())
    {
        return tag;
    }

    const std::string module_name = tag.substr(0, io.pos);
    std::string suffix = io.suffix;
    bool is_forward_start = false;
    if (!suffix.empty())
    {
        const std::string marker(kForwardStartMarker);
        if (suffix == marker)
        {
            is_forward_start = true;
            suffix.clear();
        }
        else if (suffix.rfind(marker + ".", 0) == 0)
        {
            is_forward_start = true;
            suffix = suffix.substr(marker.size() + 1);
        }
    }

    uint64_t call_idx = 0;
    auto it = g_current_forward_idx.find(module_name);
    if (is_forward_start || it == g_current_forward_idx.end())
    {
        call_idx = g_tag_counter[module_name]++;
        g_current_forward_idx[module_name] = call_idx;
    }
    else
    {
        call_idx = it->second;
    }

    std::ostringstream oss;
    oss << module_name << "." << call_idx << ".forward." << io.type;
    if (!suffix.empty())
    {
        oss << "." << suffix;
    }
    return oss.str();
}

static void ensure_acl_runtime_initialized()
{
    aclrtContext ctx = nullptr;
    aclError err = aclrtGetCurrentContext(&ctx);
    if (err != ACL_ERROR_NONE || ctx == nullptr)
    {
        ASCEND_LOGE("%s ACL runtime unavailable, err=%d, ctx=%p", kAclSaveLogPrefix, static_cast<int>(err), ctx);
        throw std::runtime_error(kAclRuntimeInitError);
    }
}

static void write_pt_or_throw(const at::Tensor& tensor, const std::string& path)
{
    std::ofstream ofs(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!ofs.is_open())
    {
        ASCEND_LOGE("%s failed to open output file, path=%s", kAclSaveLogPrefix, path.c_str());
        std::ostringstream oss;
        oss << "Failed to open tensor save file: " << path;
        throw std::runtime_error(oss.str());
    }

    auto ivalue = torch::jit::IValue(tensor);
    auto data = torch::pickle_save(ivalue);
    ofs.write(data.data(), data.size());
    if (!ofs.good())
    {
        ASCEND_LOGE("%s failed while writing file, path=%s, bytes=%llu", kAclSaveLogPrefix, path.c_str(),
                    static_cast<unsigned long long>(data.size()));
        std::ostringstream oss;
        oss << "Failed to save tensor to: " << path;
        throw std::runtime_error(oss.str());
    }

    ofs.close();
    if (!ofs)
    {
        ASCEND_LOGE("%s failed while closing file, path=%s", kAclSaveLogPrefix, path.c_str());
        std::ostringstream oss;
        oss << "Failed to close file after write: " << path;
        throw std::runtime_error(oss.str());
    }
    ASCEND_LOGI("%s file saved successfully, path=%s", kAclSaveLogPrefix, path.c_str());
}

static std::vector<int64_t> shape_to_vector(const at::Tensor& x)
{
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(x.dim()));
    for (int64_t i = 0; i < x.dim(); ++i)
    {
        shape.push_back(x.size(i));
    }
    return shape;
}

static std::string dtype_to_string(const at::Tensor& x) { return std::string(c10::toString(x.scalar_type())); }

static void acl_save_callback(const at::Tensor& x_dev_c, const std::string& path)
{
    at::Tensor xc = x_dev_c.is_contiguous() ? x_dev_c : x_dev_c.contiguous();
    auto out = at::empty_like(
        xc,
        xc.options().device(at::kCPU),
        at::MemoryFormat::Contiguous);
    const size_t nbytes =
        static_cast<size_t>(out.numel()) * static_cast<size_t>(out.element_size());
    if (nbytes == 0)
    {
        write_pt_or_throw(out, path);
        return;
    }
    aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
    aclmdlRICaptureThreadExchangeMode(&mode);
    auto memcpy_status = aclrtMemcpy(
        out.data_ptr(),
        nbytes,
        xc.data_ptr(),
        nbytes,
        ACL_MEMCPY_DEVICE_TO_HOST);
    aclmdlRICaptureThreadExchangeMode(&mode);
    if (memcpy_status != ACL_ERROR_NONE)
    {
        ASCEND_LOGE("%s device_to_host memcpy failed, path=%s, status=%d", kAclSaveLogPrefix, path.c_str(),
                    static_cast<int>(memcpy_status));
        return;
    }
    write_pt_or_throw(out, path);
}

static at::Tensor copy_to_cpu(const at::Tensor& x)
{
    auto out = at::empty_like(
        x,
        x.options().device(at::kCPU),
        at::MemoryFormat::Contiguous);

    const size_t nbytes =
        static_cast<size_t>(x.numel()) * static_cast<size_t>(x.element_size());
    if (nbytes == 0)
    {
        return out;
    }

    const auto dev_type = x.device().type();

    if (dev_type == at::DeviceType::CPU)
    {
        at::Tensor xc = x.contiguous();
        std::memcpy(out.data_ptr(), xc.const_data_ptr(), nbytes);
        return out;
    }
    return x.to(at::kCPU, /*non_blocking=*/false).contiguous();
}

static at::Tensor compute_stats_tensor(const at::Tensor& x)
{
    at::Tensor x_stat = x;
    if (x_stat.numel() == 0)
    {
        return at::zeros({4}, x_stat.options().dtype(at::kFloat));
    }
    if (x_stat.is_complex())
    {
        x_stat = at::abs(x_stat);
    }
    x_stat = x_stat.to(at::kFloat);

    at::Tensor min_t = at::amin(x_stat);
    at::Tensor max_t = at::amax(x_stat);
    at::Tensor mean_t = at::mean(x_stat);
    at::Tensor norm_t = at::norm(x_stat);
    return at::stack({min_t, max_t, mean_t, norm_t});
}

static void update_stats_map(const std::string& tag, const std::string& dtype,
                             const std::vector<int64_t>& shape, double min_v, double max_v, double mean_v,
                             double norm_v)
{
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    const std::string key = build_stat_key(tag);
    auto it = g_stat_entries.find(key);
    if (it == g_stat_entries.end())
    {
        g_stat_entry_order.push_back(key);
    }
    g_stat_entries[key] = StatRecord{dtype, shape, min_v, max_v, mean_v, norm_v};
}

static void acl_save_host_func(void* user_data)
{
    // aclgraph replay may execute the same callback payload repeatedly, so we
    // intentionally do not reclaim the payload here.
    auto* payload = static_cast<SaveTaskPayload*>(user_data);
    if (payload == nullptr)
    {
        std::cout << kAclSaveLogPrefix << " acl_save_host_func received null payload" << std::endl;
        return;
    }
    if (payload->executed.exchange(true, std::memory_order_acq_rel))
    {
        std::cout << kAclSaveLogPrefix << " skip replayed host callback, path=" << payload->save_path << std::endl;
        return;
    }

    const uint64_t file_seq = serial_num.fetch_add(1, std::memory_order_relaxed);
    const std::string final_path = build_final_path(payload->save_path, file_seq);
    acl_save_callback(payload->tensor, final_path);
}

static void acl_stat_callback(const at::Tensor& stats_dev, const std::string& tag, const std::string& dtype,
                              const std::vector<int64_t>& shape)
{
    if (!stats_dev.defined())
    {
        return;
    }

    at::Tensor stats_c = stats_dev.is_contiguous() ? stats_dev : stats_dev.contiguous();
    auto out = at::empty_like(
        stats_c,
        stats_c.options().device(at::kCPU),
        at::MemoryFormat::Contiguous);
    const size_t nbytes =
        static_cast<size_t>(out.numel()) * static_cast<size_t>(out.element_size());
    if (nbytes == 0 || out.scalar_type() != at::kFloat || out.numel() < 4)
    {
        return;
    }

    aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
    aclmdlRICaptureThreadExchangeMode(&mode);
    auto memcpy_status = aclrtMemcpy(
        out.data_ptr(),
        nbytes,
        stats_c.data_ptr(),
        nbytes,
        ACL_MEMCPY_DEVICE_TO_HOST);
    aclmdlRICaptureThreadExchangeMode(&mode);
    if (memcpy_status != ACL_ERROR_NONE)
    {
        ASCEND_LOGE("acl_stat device_to_host memcpy failed, tag=%s, status=%d", tag.c_str(), static_cast<int>(memcpy_status));
        return;
    }

    const float* p = out.const_data_ptr<float>();
    update_stats_map(tag, dtype, shape, static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2]),
                     static_cast<double>(p[3]));
}

static void acl_stat_host_func(void* user_data)
{
    // aclgraph replay may execute the same callback payload repeatedly, so we
    // intentionally do not reclaim the payload here.
    auto* payload = static_cast<StatTaskPayload*>(user_data);
    if (payload == nullptr)
    {
        return;
    }
    acl_stat_callback(payload->stats_tensor, payload->tag, payload->dtype, payload->shape);
}

static at::Tensor acl_save_impl(const at::Tensor& x, const std::string& path)
{
    const auto dev_type = x.device().type();
    if (dev_type != at::DeviceType::PrivateUse1)
    {
        const uint64_t file_seq = serial_num.fetch_add(1, std::memory_order_relaxed);
        const std::string final_path = build_final_path(path, file_seq);
        at::Tensor out = copy_to_cpu(x);
        write_pt_or_throw(out, final_path);
        return out;
    }

    ensure_acl_runtime_initialized();
    auto stream = c10_npu::getCurrentNPUStream().stream();
    auto* payload = new SaveTaskPayload(x, path);
    auto cb_status = aclrtLaunchHostFunc(stream, acl_save_host_func, payload);
    if (cb_status != ACL_ERROR_NONE)
    {
        ASCEND_LOGE("%s failed to schedule host callback, path=%s, status=%d", kAclSaveLogPrefix, path.c_str(),
                    static_cast<int>(cb_status));
        delete payload;
        check_acl(cb_status, "aclrtLaunchHostFunc failed");
    }
    return x;
}

static at::Tensor acl_stat_impl(const at::Tensor& x, const std::string& tag)
{
    if (!x.defined())
    {
        return x;
    }

    const std::string dtype = dtype_to_string(x);
    const std::vector<int64_t> shape = shape_to_vector(x);
    const auto dev_type = x.device().type();

    if (dev_type != at::DeviceType::PrivateUse1)
    {
        at::Tensor stats = compute_stats_tensor(copy_to_cpu(x));
        at::Tensor stats_cpu = stats.to(at::kCPU, /*non_blocking=*/false).contiguous();
        if (!stats_cpu.defined() || stats_cpu.scalar_type() != at::kFloat || stats_cpu.numel() < 4)
        {
            return x;
        }
        const float* p = stats_cpu.const_data_ptr<float>();
        update_stats_map(tag, dtype, shape, static_cast<double>(p[0]), static_cast<double>(p[1]),
                         static_cast<double>(p[2]), static_cast<double>(p[3]));
        return x;
    }

    ensure_acl_runtime_initialized();
    at::Tensor stats_dev = compute_stats_tensor(x);
    auto stream = c10_npu::getCurrentNPUStream().stream();
    auto* payload = new StatTaskPayload(stats_dev, tag, dtype, shape);
    auto cb_status = aclrtLaunchHostFunc(stream, acl_stat_host_func, payload);
    if (cb_status != ACL_ERROR_NONE)
    {
        delete payload;
        check_acl(cb_status, "aclrtLaunchHostFunc failed");
    }
    return x;
}

static at::Tensor acl_save_meta(const at::Tensor& x, const std::string& /*path*/)
{
    return at::empty_like(x, x.options().device(at::kMeta));
}

static at::Tensor acl_stat_meta(const at::Tensor& x, const std::string& /*tag*/)
{
    return at::empty_like(x, x.options().device(at::kMeta));
}

static py::dict build_stat_record_dict(const StatRecord& record)
{
    py::dict item;
    item["min"] = py::none();
    item["max"] = py::none();
    item["mean"] = py::none();
    item["norm"] = py::none();
    if (std::isfinite(record.min))
    {
        item["min"] = py::float_(record.min);
    }
    if (std::isfinite(record.max))
    {
        item["max"] = py::float_(record.max);
    }
    if (std::isfinite(record.mean))
    {
        item["mean"] = py::float_(record.mean);
    }
    if (std::isfinite(record.norm))
    {
        item["norm"] = py::float_(record.norm);
    }
    item["dtype"] = record.dtype;

    py::list shape;
    for (const auto dim : record.shape)
    {
        shape.append(py::int_(dim));
    }
    item["shape"] = shape;
    return item;
}

static py::dict get_acl_stat_dict_impl(bool clear)
{
    py::dict result;
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    for (const auto& key : g_stat_entry_order)
    {
        auto it = g_stat_entries.find(key);
        if (it == g_stat_entries.end())
        {
            continue;
        }
        result[py::str(key)] = build_stat_record_dict(it->second);
    }

    if (clear)
    {
        g_stat_entries.clear();
        g_tag_counter.clear();
        g_current_forward_idx.clear();
        g_stat_entry_order.clear();
    }
    return result;
}

}  // namespace

TORCH_LIBRARY(my_ns, m)
{
    m.def("acl_save(Tensor x, str path) -> Tensor");
    m.def("acl_stat(Tensor x, str tag) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ns, Meta, m)
{
    m.impl("acl_save", acl_save_meta);
    m.impl("acl_stat", acl_stat_meta);
}

TORCH_LIBRARY_IMPL(my_ns, CPU, m)
{
    m.impl("acl_save", acl_save_impl);
    m.impl("acl_stat", acl_stat_impl);
}

TORCH_LIBRARY_IMPL(my_ns, PrivateUse1, m)
{
    m.impl("acl_save", acl_save_impl);
    m.impl("acl_stat", acl_stat_impl);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "aclgraph_dump_ext: acl_save + acl_stat + host dict access";
    m.def("get_acl_stat_dict", &get_acl_stat_dict_impl, py::arg("clear") = false);
}
