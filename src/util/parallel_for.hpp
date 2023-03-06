#ifndef PARALLEL_FOR_HPP
#define PARALLEL_FOR_HPP

#include "util/launch.hpp"
#include "util/range.hpp"
#include "device_api.hpp"
#include <omp.h>
#include <thread>
#include <condition_variable>
#include <queue>

namespace simbi 
{
	namespace pooling {
		/**
		 * Practicing ThreadPooling based on this answer:
		 * https://stackoverflow.com/a/32593825/13874039
		*/
		class ThreadPool {
			public:
				void Start() {
					const unsigned num_threads = std::thread::hardware_concurrency();
					threads.resize(num_threads);
					for (unsigned i = 0; i < num_threads; i++) {
						threads[i] = std::thread(&ThreadPool::ThreadLoop, this);
					}
				}

				void QueueJob(const std::function<void()>& job) {
					{
						std::unique_lock<std::mutex> lock(queue_mutex);
						jobs.push(job);
					}
					mutex_condition.notify_one();
				}

				void Stop() {
					{
						std::unique_lock<std::mutex> lock(queue_mutex);
						should_terminate = true;
					}
					mutex_condition.notify_all();
					for (std::thread& active_thread : threads) {
						active_thread.join();
					}
					threads.clear();
				}

				bool busy() {
					bool poolbusy;
					{
						std::unique_lock<std::mutex> lock(queue_mutex);
						poolbusy = jobs.empty();
					}
					return poolbusy;
				}

				~ThreadPool() {
					Stop();
				}

				ThreadPool() {
					Start();
				}

			private:
				void ThreadLoop() {
					while (true) {
						std::function<void()> job;
						{
							std::unique_lock<std::mutex> lock(queue_mutex);
							mutex_condition.wait(lock, [this] {
								return !jobs.empty() || should_terminate;
							});
							if (should_terminate) {
								return;
							}
							job = jobs.front();
							jobs.pop();
						}
						job();
					}
				}

				bool should_terminate = false;           // Tells threads to stop looking for jobs
				std::mutex queue_mutex;                  // Prevents data races to the job queue
				std::condition_variable mutex_condition; // Allows threads to wait on new jobs or termination 
				std::vector<std::thread> threads;
				std::queue<std::function<void()>> jobs;
		};
	}
	template <typename index_type, typename F>
	void parallel_for(index_type first, index_type last, F function)  {
		ExecutionPolicy p(last - first);
		parallel_for(p, first, last, function);
	}

	template <typename index_type, typename F, Platform P = BuildPlatform>
	void parallel_for(const ExecutionPolicy<> &p, index_type first, index_type last, F function) {
		simbi::launch(p, [=] GPU_LAMBDA () {
			#if GPU_CODE
				for (auto idx : range(first, last, globalThreadCount()))  function(idx);
			#else	
				// static auto thread_pool = pooling::ThreadPool();
				// thread_pool.QueueJob([&] { 
				// 	for(auto idx = first; idx < last; idx++) function(idx);
				// });
				#pragma omp parallel for schedule(static) 
				for(auto idx = first; idx < last; idx++) function(idx);
			#endif
			
		});
			
	}
}

#endif
