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
		 * Practicing ThreadPooling based on these resources:
		 * https://stackoverflow.com/a/32593825/13874039
		 * https://www.cnblogs.com/sinkinben/p/16064857.html
		 * https://gist.github.com/tonykero/9512f2fb7f47d1ee687ae8595b17666e
		 * https://stackoverflow.com/questions/23896421/efficiently-waiting-for-all-tasks-in-a-threadpool-to-finish
		*/
		class ThreadPool {
			public:
				static ThreadPool & instance(const unsigned int nthreads){
					static ThreadPool singleton(nthreads);
					return singleton;
				}
				
				void queueUp(const std::function<void()>& job) {
					{
						std::unique_lock<std::mutex> lock(queue_mutex);
						jobs.push(job);
					}
					cv_task.notify_one();
				}

				template<typename index_type, typename F>
				void parallel_for(const index_type start, const index_type stop, const F &func) {
					static unsigned batch_size =  std::ceil((float)(stop - start) / (float)nthreads);
					auto block_start = start - batch_size;
					auto block_end   = start;
					
					auto step = [&] {
						block_start += batch_size;
						block_end   += batch_size;
						block_end    = (block_end > stop) ? stop : block_end;
					};
					step();

					for (auto &worker: threads)
					{
						queueUp([block_start, block_end, func] {
							for (auto q = block_start; q < block_end; q++) {
								func(q);
							}
						});
						step();
					}

					waitUntilFinished();
				}

				bool poolBusy() {
					bool poolbusy;
					{
						std::unique_lock<std::mutex> lock(queue_mutex);
						poolbusy = !jobs.empty();
					}
					return poolbusy;
				};
			private:
				ThreadPool(const ThreadPool &) = delete;
				ThreadPool &operator=(const ThreadPool &) = delete;

				ThreadPool(const unsigned int nthreads) : nthreads(nthreads), 
				should_terminate(false), busy(0) {
					threads.reserve(nthreads);
					for (unsigned i = 0; i < nthreads; i++) {
						threads.emplace_back(std::thread(&ThreadPool::spawn_thread, this));
					}
				}

				~ThreadPool() {
					// Stop the thread pool and notify all threads ot finish the 
					// remaining tasks
					{
						std::unique_lock<std::mutex> lock(queue_mutex);
						should_terminate = true;
					}
					cv_task.notify_all();
					for (std::thread& active_thread : threads) {
						active_thread.join();
					}
					threads.clear();
				}

				void spawn_thread_proc() {
					while (true) {
						std::function<void()> job;
						std::unique_lock<std::mutex> latch(queue_mutex);
						cv_task.wait(latch, [this] {
							return !jobs.empty() || should_terminate;
						});
						// pop a job from queue and execute it
						{
							if (should_terminate) {
								return;
							}

							// got work. set busy.
							++busy;
							job = jobs.front();
							jobs.pop();
						}
						latch.unlock();
						job();
						latch.lock();
						--busy;
						cv_finished.notify_one();
					}
				}

				void spawn_thread() {
					while (true) {
						std::function<void()> job;
						// pop a job from queue and execute it
						{
							std::unique_lock<std::mutex> latch(queue_mutex);
							cv_task.wait(latch, [this] {
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

				// waits until the queue is empty.
				void waitUntilFinished() {
					while (poolBusy()) {
						/* chill out */
					}
					// std::unique_lock<std::mutex> lock(queue_mutex);
					// cv_finished.wait(lock, [this]{ return jobs.empty() && busy == 0; });
				}

				unsigned int nthreads;
				bool should_terminate;                   // Tells threads to stop looking for jobs
				std::mutex queue_mutex;                  // Prevents data races to the job queue
				std::condition_variable cv_task; // Allows threads to wait on new jobs or termination 
				std::condition_variable cv_finished;
				std::vector<std::thread> threads;
				std::queue<std::function<void()>> jobs;
				unsigned int busy;
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

// static auto &thread_pool = simbi::pooling::ThreadPool::instance(simbi::pooling::get_nthreads());
#endif 