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


#ifndef DUMPTHREADPOOL_H
#define DUMPTHREADPOOL_H

#include <stdexcept>
#include <functional>
#include <future>
#include <condition_variable>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <memory>
#include <atomic>
#include "ait_logger.h"

#define EXPORT_LLM __attribute__ ((visibility("default")))

namespace ThreadPool {
class DumpThreadPool {
public:
    explicit DumpThreadPool(size_t threads);
    ~DumpThreadPool();

    // 禁止拷贝和移动
    DumpThreadPool(const DumpThreadPool&) = delete;
    DumpThreadPool& operator=(const DumpThreadPool&) = delete;
    DumpThreadPool(DumpThreadPool&&) = delete;
    DumpThreadPool& operator=(DumpThreadPool&&) = delete;

    template<class F, class... Args>
    EXPORT_LLM auto Enqueue(F &&f, Args &&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    std::vector<std::thread> thread_workers;
    std::queue<std::function<void()> > thread_tasks;

    std::mutex threadQueueMtx;
    std::condition_variable threadCondition;
    std::atomic<bool> poolStop;
};
}

ThreadPool::DumpThreadPool::DumpThreadPool(size_t threads) : poolStop(false)
{
    for (size_t i = 0; i < threads; ++i)
        thread_workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> task_lock(this->threadQueueMtx);
                    this->threadCondition.wait(task_lock, [this] {
                        return this->poolStop.load(std::memory_order_acquire) ||
                            !this->thread_tasks.empty(); // 防止虚假唤醒
                    });

                    if (this->poolStop.load(std::memory_order_acquire) && this->thread_tasks.empty()) {
                        return;
                    }
                    task = std::move(this->thread_tasks.front());
                    this->thread_tasks.pop();
                }
                try {
                    task();
                } catch (const std::exception &e) {
                    AIT_LOG_ERROR("DumpThreadPool task exception: " + std::string(e.what()));
                    throw;
                }
            }
        }
    );
}

ThreadPool::DumpThreadPool::~DumpThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(threadQueueMtx);
        poolStop.store(true, std::memory_order_release);
    }
    threadCondition.notify_all();
    for (std::thread &worker: thread_workers) {
        if (worker.joinable()) { worker.join(); }
    }
}

template<class F, class... Args>
auto ThreadPool::DumpThreadPool::Enqueue(F &&f, Args &&... args)
-> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_functype = typename std::result_of<F(Args...)>::type;

    auto nowtask = std::make_shared<std::packaged_task<return_functype()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_functype> resTask = nowtask->get_future();
    {
        std::unique_lock<std::mutex> lock(threadQueueMtx);
        if (poolStop.load(std::memory_order_acquire)) {
            throw std::runtime_error("Enqueue on stopped DumpThreadPool");
        }
        thread_tasks.emplace([nowtask]() {
            try {
                (*nowtask)();
            } catch (const std::exception &e) {
                AIT_LOG_ERROR("DumpThreadPool task exception: " + std::string(e.what()));
                throw;
            }
        });
    }
    threadCondition.notify_one();

    return resTask;
}

#endif