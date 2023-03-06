#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <thread>
#include <condition_variable>
#include <queue>
#include <future>
#include <iostream>

namespace simbi {
    namespace pooling {
		/**
		 * Practicing ThreadPooling based on this answer:
		 * https://stackoverflow.com/a/32593825/13874039
		*/
		class ThreadPool {
			public:
				ThreadPool(const ThreadPool &) = delete;
				ThreadPool(ThreadPool &&) = delete;
				ThreadPool &operator=(const ThreadPool &) = delete;
				ThreadPool &operator=(ThreadPool &&) = delete;

				ThreadPool(){

				}

				ThreadPool(const unsigned int nthreads) : nthreads(nthreads) {
                    std::cout << "ctor called" << "\n";
					threads.reserve(nthreads);
					for (unsigned i = 0; i < nthreads; i++) {
						threads.emplace_back(std::thread(&ThreadPool::ThreadLoop, this));
					}
				}

				~ThreadPool() {
					// Stop the thread pool and notify all threads ot finish the 
					// remaining tasks
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

				void QueueJob(const std::function<void()>& job) {
					{
						std::unique_lock<std::mutex> lock(queue_mutex);
						jobs.push(job);
					}
					mutex_condition.notify_one();
				}

				template<typename index_type, typename F>
				void parallel_for(const index_type start, const index_type stop, const F func) {
					static unsigned batch_size =  std::ceil((float)(stop - start) / (float)nthreads);
					auto block_start = start - batch_size;
					auto block_end   = start;
					
					auto step = [&] {
						block_start += batch_size;
						block_end   += batch_size;
						block_end    = (block_end > stop) ? stop : block_end;
					};

					step();
					for (auto &worker : threads)
					{
						QueueJob([=] {
							for (auto q = block_start; q < block_end; q++) {
								func(q);
							}
						});
						step();
					}
				}

				bool busy() {
					bool poolbusy;
					{
						std::unique_lock<std::mutex> lock(queue_mutex);
						poolbusy = !jobs.empty();
					}
					return poolbusy;
				}

				void set_threads(const unsigned int nthreads) {
					this->nthreads = nthreads; 
					std::cout << "ctor called" << "\n";
					threads.reserve(nthreads);
					for (unsigned i = 0; i < nthreads; i++) {
						threads.emplace_back(std::thread(&ThreadPool::ThreadLoop, this));
					}
				}

			private:
				void ThreadLoop() {
					while (true) {
						std::function<void()> job;
						// pop a job from queue and execute it
						{
							std::unique_lock<std::mutex> lock(queue_mutex);
							mutex_condition.wait(lock, [this] {
								return !jobs.empty() || should_terminate;
							});
							if (should_terminate) {
								return;
							}

							job = std::move(jobs.front());
							jobs.pop();
						}
						job();
					}
				}
				unsigned int nthreads;
				bool should_terminate = false;           // Tells threads to stop looking for jobs
				std::mutex queue_mutex;                  // Prevents data races to the job queue
				std::condition_variable mutex_condition; // Allows threads to wait on new jobs or termination 
				std::vector<std::thread> threads;
				std::queue<std::function<void()>> jobs;
		};

		inline auto get_nthreads = ([] {
			if(const char* thread_env = std::getenv("NTHREADS"))
				return static_cast<unsigned int>(std::stoul(std::string(thread_env)));

			if(const char* thread_env = std::getenv("OMP_NUM_THREADS"))
				return static_cast<unsigned int>(std::stoul(std::string(thread_env)));

			return std::thread::hardware_concurrency();
		});
	}// namespace pooling
} // namespace simbi

extern simbi::pooling::ThreadPool thread_pool;
#endif 