// Header to implement custom cpu / gpu array class
#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <cstddef>            // for size_t
#include <initializer_list>   // for initializer_list
#include <iterator>           // for forward_iterator_tag
#include <memory>             // for unique_ptr
#include <vector>             // for vector
#include "build_options.hpp"  // for Platform, global::BuildPlatform
#include "device_api.hpp"     // for gpuFree, gpuMalloc, gpuMallocManaged


using size_type = std::size_t;
namespace simbi
{
	// Template class to create array of
	// different data_type
	template <typename DT, global::Platform build_mode= global::BuildPlatform>
	class ndarray {
	template <typename Deleter>
	using unique_p = std::unique_ptr<DT[], Deleter>;
	private:
		// Variable to store the size of the
		// array
		size_type sz;
		
		// Variable to store the current capacity
		// of the array
		size_type nd_capacity;

		size_type dimensions;

		std::unique_ptr<DT[]> arr;
		// Device-side array
		void* myGpuMalloc(size_type size) { 
			if constexpr(build_mode == global::Platform::GPU) {
				void* ptr; 
				gpu::api::gpuMalloc(&ptr, size);
				return ptr; 
			}
			return nullptr;
		};

		// Device-side array
		void* myGpuMallocManaged(size_type size) { 
			if constexpr(build_mode == global::Platform::GPU) {
				void* ptr; 
				gpu::api::gpuMallocManaged(&ptr, size);
				return ptr; 
			}
		};
		
		struct gpuDeleter {
			void operator()(DT* ptr) {
				if constexpr(build_mode == global::Platform::GPU) {
					gpu::api::gpuFree(ptr); 
				}
			 }
		};

		unique_p<gpuDeleter> dev_arr;
	public:
		ndarray();
		~ndarray();
		// Assignment operator
		ndarray& operator=(ndarray rhs);

		void swap(ndarray& rhs);

		// Initilizer list constructor
		ndarray(std::initializer_list<DT> list);

		// Zero-initialize the array with defined size
		ndarray(size_type size);

		// Fill-initialize the array with defined size
		ndarray(size_type size, const DT val);

		// Copy-constructor for array
		ndarray(const ndarray& rhs);
		ndarray(const std::vector<DT>& rhs);

		// Function that returns the number of
		// elements in array after pushing the data
		constexpr void push_back(const DT&);

		// function that returns the popped element
		constexpr void pop_back();

		// fucntion to resize ndarray
		constexpr void resize(size_type new_size);

		// fucntion to resize ndarray
		constexpr void resize(size_type new_size, const DT new_value);

		// Function that return the size of array
		constexpr size_type size() const;
		constexpr size_type capacity() const;
		constexpr size_type ndim() const;

		// Access operator (mutable)
		template <typename IndexType>
		constexpr DT& operator[](IndexType);

		// Const-access operator (read-only)
		template<typename IndexType>
		constexpr DT operator[](IndexType) const ;

		// Some math operator overloads
		constexpr ndarray& operator*(real);
		constexpr ndarray& operator*=(real);
		constexpr ndarray& operator/(real);
		constexpr ndarray& operator/=(real);
		constexpr ndarray& operator+=(const ndarray& rhs);



		// Check if ndarray is empty
		bool empty() const;

		// get pointers to underlying data ambigiously, on host, or on gpu
		DT* data();
		DT* host_data();
		DT* dev_data();
		// Iterator Class
		class iterator {
		private:
			// Dynamic array using pointers
			DT* ptr;
		public:
			using iterator_category = std::forward_iterator_tag;;
			using value_type        = DT;
			using difference_type   = void;
			using pointer           = void;
			using reference         = void;
			explicit iterator()
				: ptr(nullptr)
			{
			}
			explicit iterator(DT* p)
				: ptr(p)
			{
			}
			bool operator==(const iterator& rhs) const
			{
				return ptr == rhs.ptr;
			}
			bool operator!=(const iterator& rhs) const
			{
				return !(*this == rhs);
			}
			DT operator*() const
			{
				return *ptr;
			}
			iterator& operator++()
			{
				++ptr;
				return *this;
			}
			iterator operator++(int)
			{
				iterator temp(*this);
				++*this;
				return temp;
			}
		};

		// Begin iterator
		iterator begin() const;

		// End iterator
		iterator end() const;
		
		// back of container
		DT  back() const;
		DT& back();
		DT  front() const;
		DT& front();


		// GPU memeory copy helpers
		void copyToGpu();
		void copyFromGpu();
		void copyBetweenGpu(const ndarray &rhs);

	}; // end ndarray class declaration

} // namespace simbi

// Type trait 
template <typename T>
struct is_ndarray {
	static constexpr bool value = false;
};

template <typename T>
struct is_2darray {
	static constexpr bool value = false;
};

template <typename T>
struct is_3darray {
	static constexpr bool value = false;
};

template <typename T>
struct is_1darray {
	static constexpr bool value = false;
};

template<typename U>
struct is_ndarray<simbi::ndarray<U>>
{
	static constexpr bool value = true;
};

template<typename U>
struct is_1darray<simbi::ndarray<U>>
{
	static constexpr bool value = true;
};

template<typename U>
struct is_2darray<simbi::ndarray<simbi::ndarray<U>>>
{
	static constexpr bool value = true;
};

template<typename U>
struct is_3darray<simbi::ndarray<simbi::ndarray<simbi::ndarray<U>>>>
{
	static constexpr bool value = true;
};

#include "ndarray.tpp"
#endif 