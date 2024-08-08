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

namespace ThreadPool {
class DumpThreadPool {
public:
    explicit DumpThreadPool(size_t threads);
    ~DumpThreadPool();
    template<class F, class... Args>
    auto Enqueue(F &&f, Args &&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    std::vector<std::thread> thread_workers;
    std::queue<std::function<void()> > thread_tasks;

    std::mutex threadQueueMtx;
    std::condition_variable threadCondition;
    bool poolStop;
};
}

inline ThreadPool::DumpThreadPool::DumpThreadPool(size_t threads) : poolStop(false)
{
    for (size_t i = 0; i < threads; ++i)
        thread_workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> task_lock(this->threadQueueMtx);
                    this->threadCondition.wait(task_lock, [this] {
                        return this->poolStop || !this->thread_tasks.empty();
                    });

                    if (this->poolStop && this->thread_tasks.empty()) {
                        return;
                    }
                    task = std::move(this->thread_tasks.front());
                    this->thread_tasks.pop();
                }

                task();
            }
        }
    );
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

        if (poolStop) {
            throw std::runtime_error("Enqueue on stopped DumpThreadPool");
        }
        thread_tasks.emplace([nowtask]() { (*nowtask)(); });
    }
    threadCondition.notify_one();
    return resTask;
}

inline ThreadPool::DumpThreadPool::~DumpThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(threadQueueMtx);
        poolStop = true;
    }
    threadCondition.notify_all();
    for (std::thread &worker: thread_workers) {
        worker.join();
    }
}

#endif