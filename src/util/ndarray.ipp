#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <typeinfo>

// Zero-initialize the array with defined size
template <typename DT, int dim>
simbi::ndarray<DT, dim>::ndarray(size_type size)
    : sz(size), nd_capacity(size), dimensions(1)
{
    arr = util::make_unique<DT[]>(nd_capacity);
}

// Initialize the array with a given value
template <typename DT, int dim>
simbi::ndarray<DT, dim>::ndarray(size_type size, const DT val)
    : sz(size), nd_capacity(size), dimensions(1)
{
    arr = util::make_unique<DT[]>(size);
    std::fill(arr.get(), arr.get() + sz, val);

    // if constexpr (is_ndarray<DT, dim>::value) {
    //     dimensions += val.ndim();
    // }
}

// Copy-constructor for array
template <typename DT, int dim>
simbi::ndarray<DT, dim>::ndarray(const ndarray& rhs)
    : sz(rhs.sz), dimensions(rhs.dimensions), arr(new DT[rhs.sz])
{
    copyFromGpu();
    std::copy(rhs.arr.get(), rhs.arr.get() + sz, arr.get());
    copyToGpu();
}

// Move-constructor for array
template <typename DT, int dim>
simbi::ndarray<DT, dim>::ndarray(ndarray&& rhs) noexcept
    : sz(rhs.sz),
      nd_capacity(rhs.nd_capacity),
      dimensions(rhs.dimensions),
      arr(std::move(rhs.arr))
{
    rhs.sz          = 0;
    rhs.nd_capacity = 0;
    rhs.dimensions  = 0;
}

// Copy-constructor for vector
template <typename DT, int dim>
simbi::ndarray<DT, dim>::ndarray(const std::vector<DT>& rhs)
    : sz(rhs.size()),
      nd_capacity(rhs.capacity()),
      dimensions(1),
      arr(new DT[rhs.size()])
{
    std::copy(rhs.begin(), rhs.end(), arr.get());
}

// Move-constructor for vector
template <typename DT, int dim>
simbi::ndarray<DT, dim>::ndarray(std::vector<DT>&& rhs)
    : sz(rhs.size()),
      nd_capacity(rhs.capacity()),
      dimensions(1),
      arr(new DT[rhs.size()])
{
    std::move(rhs.begin(), rhs.end(), arr.get());
}

// Copy the arrays and deallocate the RHS
template <typename DT, int dim>
simbi::ndarray<DT, dim>& simbi::ndarray<DT, dim>::operator=(ndarray other)
{
    other.swap(*this);
    return *this;
}

// Copy the arrays and deallocate the RHS
template <typename DT, int dim>
constexpr simbi::ndarray<DT, dim>&
simbi::ndarray<DT, dim>::operator+=(const ndarray& other)
{
    simbi::ndarray<DT, dim> newArray(sz + other.sz);
    std::copy(this->arr.get(), this->arr.get() + this->sz, newArray.arr.get());
    std::copy(
        other.arr.get(),
        other.arr.get() + other.sz,
        newArray.arr.get() + this->sz
    );
    newArray.swap(*this);
    return *this;
}

template <typename DT, int dim>
void simbi::ndarray<DT, dim>::swap(ndarray& other)
{
    std::swap(arr, other.arr);
    std::swap(sz, other.sz);
    std::swap(nd_capacity, other.nd_capacity);
}

// Template class to insert the element in array
template <typename DT, int dim>
constexpr simbi::ndarray<DT, dim>&
simbi::ndarray<DT, dim>::push_back(const DT& data)
{
    if (sz == nd_capacity) {
        resize(sz == 0 ? 1 : 2 * sz);
    }
    arr[sz++] = data;
    return *this;
}

// Template class to return the popped element in array
template <typename DT, int dim>
constexpr simbi::ndarray<DT, dim>& simbi::ndarray<DT, dim>::pop_back()
{
    if (!empty()) {
        arr[sz - 1].~DT();
        --sz;
    }
    return *this;
}

template <typename DT, int dim>
constexpr simbi::ndarray<DT, dim>&
simbi::ndarray<DT, dim>::resize(size_type new_size)
{
    if (new_size > sz) {
        auto new_arr = util::make_unique<DT[]>(new_size);
        std::copy(arr.get(), arr.get() + sz, new_arr.get());
        arr.swap(new_arr);
    }
    sz          = new_size;
    nd_capacity = new_size;
    return *this;
}

template <typename DT, int dim>
constexpr simbi::ndarray<DT, dim>&
simbi::ndarray<DT, dim>::resize(size_type new_size, const DT new_value)
{
    if (new_size > sz) {
        auto new_arr = util::make_unique<DT[]>(new_size);
        std::copy(arr.get(), arr.get() + sz, new_arr.get());
        std::fill(new_arr.get() + sz, new_arr.get() + new_size, new_value);
        arr.swap(new_arr);
    }
    sz          = new_size;
    nd_capacity = new_size;
    return *this;
}

// Template class to return the size of array
template <typename DT, int dim>
constexpr size_type simbi::ndarray<DT, dim>::size() const
{
    return sz;
}

// Template class to return the capacity of array
template <typename DT, int dim>
constexpr size_type simbi::ndarray<DT, dim>::capacity() const
{
    return nd_capacity;
}

// Template class to return the number of dimensions of array
template <typename DT, int dim>
constexpr size_type simbi::ndarray<DT, dim>::ndim() const
{
    return dimensions;
}

// Template class to return the element of array at given index
template <typename DT, int dim>
template <typename IndexType>
DUAL constexpr DT& simbi::ndarray<DT, dim>::operator[](IndexType index)
{
    // if given index is greater than the size of array print Error
    if ((size_t) index >= sz) {
        printf(
            "Error: array index %" PRIu64
            " out of bounds for ndarray of size %" PRIu64 "\n",
            (luint) index,
            (luint) sz
        );
    }
    // else return value at that index
#ifdef __CUDA_ARCH__
    return dev_arr[index];
#else
    return arr[index];
#endif
}

// Template class to return the element of array at given index
template <typename DT, int dim>
template <typename IndexType>
DUAL constexpr DT simbi::ndarray<DT, dim>::operator[](IndexType index) const
{
    // if given index is greater than the size of array print Error
    if ((size_t) index >= sz) {
        printf(
            "Error: array index %" PRIu64
            " out of bounds for ndarray of size %" PRIu64 "\n",
            (luint) index,
            (luint) sz
        );
    }
    // else return value at that index
#ifdef __CUDA_ARCH__
    return dev_arr[index];
#else
    return arr[index];
#endif
}

// Template class to scale the array by a factor
template <typename DT, int dim>
constexpr simbi::ndarray<DT, dim>&
simbi::ndarray<DT, dim>::operator*(const real scale_factor)
{
    std::transform(
        arr.get(),
        arr.get() + sz,
        arr.get(),
        [scale_factor](DT& val) { return val * scale_factor; }
    );
    return *this;
}

// Template class to scale the array by a factor
template <typename DT, int dim>
constexpr simbi::ndarray<DT, dim>&
simbi::ndarray<DT, dim>::operator*=(const real scale_factor)
{
    std::transform(
        arr.get(),
        arr.get() + sz,
        arr.get(),
        [scale_factor](DT& val) { return val * scale_factor; }
    );
    return *this;
}

// Template class to divide the array by a factor
template <typename DT, int dim>
constexpr simbi::ndarray<DT, dim>&
simbi::ndarray<DT, dim>::operator/(const real scale_factor)
{
    std::transform(
        arr.get(),
        arr.get() + sz,
        arr.get(),
        [scale_factor](DT& val) { return val / scale_factor; }
    );
    return *this;
}

// Template class to divide the array by a factor
template <typename DT, int dim>
constexpr simbi::ndarray<DT, dim>&
simbi::ndarray<DT, dim>::operator/=(const real scale_factor)
{
    std::transform(
        arr.get(),
        arr.get() + sz,
        arr.get(),
        [scale_factor](DT& val) { return val / scale_factor; }
    );
    return *this;
}

// Template class to return begin iterator
template <typename DT, int dim>
typename simbi::ndarray<DT, dim>::iterator
simbi::ndarray<DT, dim>::begin() const
{
    return iterator(arr.get());
}

// Template class to return end iterator
template <typename DT, int dim>
typename simbi::ndarray<DT, dim>::iterator simbi::ndarray<DT, dim>::end() const
{
    return iterator(arr.get() + sz);
}

// Template class to return the last element
template <typename DT, int dim>
DT simbi::ndarray<DT, dim>::back() const
{
    if (empty()) {
        throw std::out_of_range("Array is empty");
    }
    return arr[sz - 1];
}

// Template class to return the last element
template <typename DT, int dim>
DT& simbi::ndarray<DT, dim>::back()
{
    if (empty()) {
        throw std::out_of_range("Array is empty");
    }
    return arr[sz - 1];
}

// Template class to return the first element
template <typename DT, int dim>
DT& simbi::ndarray<DT, dim>::front()
{
    if (empty()) {
        throw std::out_of_range("Array is empty");
    }
    return arr[0];
}

// Template class to return the first element
template <typename DT, int dim>
DT simbi::ndarray<DT, dim>::front() const
{
    if (empty()) {
        throw std::out_of_range("Array is empty");
    }
    return arr[0];
}

// Template class to check if the array is empty
template <typename DT, int dim>
bool simbi::ndarray<DT, dim>::empty() const
{
    return sz == 0;
}

// Template class to print the array
template <typename DT, int dim>
std::ostream& operator<<(std::ostream& out, const simbi::ndarray<DT, dim>& v)
{
    unsigned counter        = 1;
    const int max_cols      = 10;
    bool end_point          = false;
    int nelems              = v.size();
    static bool new_row     = false;
    static bool new_aisle   = false;
    static int cycle        = 0;
    static const auto nrows = [&]() -> auto {
        if constexpr (is_2darray<DT>::value) {
            return v[0].size();
        }
        return v.size();
    }();
    if (new_row) {
        if (!new_aisle) {
            out << "\b\b  ";
        }
        else {
            out << "\b\b[";
        }
        new_row   = false;
        new_aisle = false;
    }
    static int ii          = 0;
    static int kk          = 0;
    static int idx         = 0;
    static int aisle_count = 1;
    out << "[";
    for (auto i : v) {
        out << i << ", ";
        kk = idx / max_cols / nrows;
        ii = (idx - kk * nrows * max_cols) % max_cols;
        if (counter == max_cols) {
            if (ii == nelems - 1) {
                end_point = true;
            }
            if (!end_point) {
                if (v.ndim() == 1) {
                    out << '\n';
                    out << " ";
                    counter = 0;
                }
            }
        }
        idx++;
        counter++;
    }
    out << "\b\b]";   // use two ANSI backspace characters '\b' to overwrite
                      // final ", "
    if (idx % max_cols == 0) {
        new_row = true;
        if (idx / (nrows * max_cols) == aisle_count) {
            new_aisle = true;
            aisle_count++;
        }
        else {
            if (cycle == 0) {
                out << "\n";
                cycle++;
            }
            else {
                if (idx == cycle * nrows) {
                    if (idx != 2 * nrows) {
                        out << "\n\n";
                    }
                    cycle++;
                }
                else {
                    out << "\n";
                }
            }
        }
    }
    return out;
}

// Template class to copy data to GPU
template <typename DT, int dim>
void simbi::ndarray<DT, dim>::copyToGpu()
{
    if constexpr (global::on_gpu) {
        if (arr && !is_gpu_synced) {
            if (!dev_arr) {
                dev_arr.reset((DT*) myGpuMalloc(sz * sizeof(DT)));
            }
            if (current_stream) {
                gpu::api::asyncCopyHostToDevice(
                    dev_arr.get(),
                    arr.get(),
                    sz * sizeof(DT),
                    current_stream
                );
            }
            else {
                gpu::api::copyHostToDevice(
                    dev_arr.get(),
                    arr.get(),
                    sz * sizeof(DT)
                );
            }
            is_gpu_synced = true;
        }
    }
}

// Template class to copy data from GPU
template <typename DT, int dim>
void simbi::ndarray<DT, dim>::copyFromGpu()
{
    if (dev_arr) {
        gpu::api::copyDevToHost(arr.get(), dev_arr.get(), sz * sizeof(DT));
    }
}

// Template class to copy data between GPUs
template <typename DT, int dim>
void simbi::ndarray<DT, dim>::copyBetweenGpu(const ndarray& rhs)
{
    if (dev_arr) {
        gpu::api::copyDevToDev(
            dev_arr.get(),
            rhs.dev_arr.get(),
            rhs.sz * sizeof(DT)
        );
    }
}

// Template class to return host data pointer
template <typename DT, int dim>
DT* simbi::ndarray<DT, dim>::host_data()
{
    return arr.get();
}

// Template class to return host data pointer
template <typename DT, int dim>
DT* simbi::ndarray<DT, dim>::host_data() const
{
    return arr.get();
}

// Template class to return device data pointer
template <typename DT, int dim>
DUAL DT* simbi::ndarray<DT, dim>::dev_data()
{
    return dev_arr.get();
}

// Template class to return device data pointer
template <typename DT, int dim>
DUAL DT* simbi::ndarray<DT, dim>::dev_data() const
{
    return dev_arr.get();
}

// Template class to return data pointer
template <typename DT, int dim>
DUAL DT* simbi::ndarray<DT, dim>::data()
{
    if (sz == 0) {
        return nullptr;
    }

    if constexpr (global::on_gpu) {
        return dev_arr.get();
    }
    else {
        return arr.get();
    }
}

// Template class to return data pointer
template <typename DT, int dim>
DT* simbi::ndarray<DT, dim>::data() const
{
    if (sz == 0) {
        return nullptr;
    }

    if constexpr (global::on_gpu) {
        return dev_arr.get();
    }
    else {
        return arr.get();
    }
}

template <typename DT, int dim>
void simbi::ndarray<DT, dim>::clear()
{
    arr.reset();
    dev_arr.reset();
    sz = 0;
}

template <typename DT, int dim>
void simbi::ndarray<DT, dim>::shrink_to_fit()
{
    if (sz < nd_capacity) {
        auto new_arr = util::make_unique<DT[]>(sz);
        std::copy(arr.get(), arr.get() + sz, new_arr.get());
        arr.swap(new_arr);
        nd_capacity = sz;
    }
}

template <typename DT, int dim>
void simbi::ndarray<DT, dim>::reserve(size_type new_capacity)
{
    if (new_capacity > nd_capacity) {
        auto new_arr = util::make_unique<DT[]>(new_capacity);
        std::copy(arr.get(), arr.get() + sz, new_arr.get());
        arr.swap(new_arr);
        nd_capacity = new_capacity;
    }
}

template <typename DT, int dim>
void simbi::ndarray<DT, dim>::unpin_memory()
{
    if (dev_arr) {
        gpu::api::hostUnregister(arr.get());
    }
}

template <typename DT, int dim>
void simbi::ndarray<DT, dim>::pin_memory()
{
    if (dev_arr) {
        gpu::api::hostRegister(arr.get(), sz * sizeof(DT), 0);
    }
}

template <typename DT, int dim>
void simbi::ndarray<DT, dim>::set_stream(simbiStream_t stream)
{
    current_stream = stream;
}

template <typename DT, int dim>
void simbi::ndarray<DT, dim>::async_copy_to_gpu()
{
    if (!dev_arr) {
        dev_arr.reset((DT*) myGpuMalloc(sz * sizeof(DT)));
    }
    if (current_stream) {
        gpu::api::asyncCopyHostToDevice(
            dev_arr.get(),
            arr.get(),
            sz * sizeof(DT),
            current_stream
        );
    }
}

template <typename DT, int dim>
void simbi::ndarray<DT, dim>::ensure_gpu_synced()
{
    if constexpr (global::on_gpu) {
        if (needs_gpu_sync && !is_gpu_synced) {
            copyToGpu();
            is_gpu_synced  = true;
            needs_gpu_sync = false;
        }
    }
}

template <typename DT, int dim>
void* simbi::ndarray<DT, dim>::aligned_alloc(
    size_type size,
    size_type alignment
)
{
    void* ptr;
    if constexpr (global::on_gpu) {
        gpu::api::alignedMalloc(&ptr, size);
    }
    else {
        ptr = std::aligned_alloc(alignment, size);
    }
    return ptr;
}

//==============================================================================
//                        Functional Operations
//==============================================================================
template <typename DT, int dim>
template <typename UnaryFunction>
simbi::ndarray<DT, dim>& simbi::ndarray<DT, dim>::map(UnaryFunction f)
{
    if constexpr (global::on_gpu) {
        auto ptr = dev_arr.get();
        gpu::api::launchKernel(
            [ptr, f] DEV(size_type i) { ptr[i] = f(ptr[i]); },
            sz,
            256,
            nullptr,
            0,
            nullptr
        );
    }
    else {
        std::transform(arr.get(), arr.get() + sz, arr.get(), f);
    }
    return *this;
}

template <typename DT, int dim>
template <typename UnaryFunction>
simbi::ndarray<DT, dim> simbi::ndarray<DT, dim>::map(UnaryFunction f) const
{
    simbi::ndarray<DT, dim> result;
    std::transform(
        arr.get(),
        arr.get() + sz,
        std::back_inserter(result.arr),
        f
    );
    result.sz          = result.arr.size();
    result.nd_capacity = result.sz;
    return result;
}

template <typename DT, int dim>
template <typename UnaryPredicate>
simbi::ndarray<DT, dim>& simbi::ndarray<DT, dim>::filter(UnaryPredicate pred)
{
    auto new_end = std::remove_if(arr.get(), arr.get() + sz, pred);
    sz           = new_end - arr.get();
    return *this;
}

template <typename DT, int dim>
template <typename UnaryPredicate>
simbi::ndarray<DT, dim> simbi::ndarray<DT, dim>::filter(UnaryPredicate pred
) const
{
    simbi::ndarray<DT, dim> result;
    std::copy_if(
        arr.get(),
        arr.get() + sz,
        std::back_inserter(result.arr),
        pred
    );
    result.sz          = result.arr.size();
    result.nd_capacity = result.sz;
    return result;
}

template <typename DT, int dim>
template <typename U, typename BinaryOp>
U simbi::ndarray<DT, dim>::reduce(
    const ExecutionPolicy<>& policy,
    U init,
    BinaryOp binary_op
) const
{
    if constexpr (global::on_gpu) {
        ndarray<U> result(1, init);
        result.copyToGpu();
        auto result_ptr = result.dev_data();

        const auto in_ptr          = dev_arr.get();
        const size_type num_items  = sz;
        const size_type block_size = policy.batch_size;
        const size_type num_blocks = policy.get_num_batches(sz);

        // Two-phase reduction
        parallel_for(policy, num_blocks, [=] DEV(size_type bid) {
            // Phase 1: Block reduction
            SHARED U shared[256];   // Assuming max block size

            const int tid = threadIdx.z * blockDim.x * blockDim.y +
                            threadIdx.y * blockDim.x + threadIdx.x;
            const size_type start = bid * block_size + tid;

            // Load and reduce within thread
            U thread_sum = init;
            for (size_type i = start; i < num_items;
                 i += block_size * gridDim.x) {
                thread_sum = binary_op(thread_sum, in_ptr[i]);
            }

            // Store in shared memory
            shared[tid] = thread_sum;
            gpu::api::synchronize();

            // Reduce within block
            for (size_type s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = binary_op(shared[tid], shared[tid + s]);
                }
                gpu::api::synchronize();
            }

            // Update global result
            if (tid == 0) {
                gpu::api::atomicMin(result_ptr, shared[0]);
            }
        });

        result.copyFromGpu();
        return result[0];
    }
    else {
        // Existing CPU implementation
        std::atomic<U> result(init);
        parallel_for(policy, size(), [&](size_type i) {
            U old_val = result.load();
            U new_val = binary_op(old_val, arr[i]);
            while (!result.compare_exchange_weak(old_val, new_val)) {
                new_val = binary_op(old_val, arr[i]);
            }
        });
        return result.load();
    }
}

template <typename DT, int dim>
template <typename F, typename G>
auto simbi::ndarray<DT, dim>::compose(F f, G g) const
{
    return f(g(*this));
}

template <typename DT, int dim>
template <typename... Funcs>
auto simbi::ndarray<DT, dim>::then(Funcs... f) const
{
    return (... | f)(*this);
}

template <typename DT, int dim>
template <typename... Transforms>
simbi::ndarray<DT, dim>
simbi::ndarray<DT, dim>::transform_chain(Transforms... transforms) const
{
    return (... | transforms)(*this);
}

template <typename DT, int dim>
template <typename Func>
Maybe<simbi::ndarray<DT, dim>> simbi::ndarray<DT, dim>::safe_map(Func f) const
{
    if (empty()) {
        return Maybe<simbi::ndarray<DT, dim>>();
    }
    return Maybe<simbi::ndarray<DT, dim>>(map(f));
}

template <typename DT, int dim>
template <typename Func>
simbi::ndarray<DT, dim>
simbi::ndarray<DT, dim>::combine(const ndarray& other, Func binary_op) const
{
    simbi::ndarray<DT, dim> result(size());
    std::transform(begin(), end(), other.begin(), result.begin(), binary_op);
    return result;
}

template <typename DT, int dim>
template <typename Func>
simbi::ndarray<DT, dim> simbi::ndarray<DT, dim>::parallel_chunks(
    const ExecutionPolicy<>& policy,
    Func chunk_op
) const
{
    simbi::ndarray<DT, dim> result(size());
    size_type batch_size  = policy.batch_size;
    size_type num_batches = policy.get_num_batches(sz);
    if constexpr (global::on_gpu) {
        auto out_ptr = result.dev_data();
        auto in_ptr  = dev_arr.get();

        simbi::parallel_for(policy, num_batches, [=, this] DEV(size_t i) {
            const size_t start = i * batch_size;
            const size_t end   = my_min(start + batch_size, sz);
            for (size_t j = start; j < end; j++) {
                out_ptr[j] = chunk_op(in_ptr[j]);
            }
        });
    }
    else {
        simbi::parallel_for(policy, num_batches, [&](size_t i) {
            const size_t start = i * batch_size;
            const size_t end   = std::min(start + batch_size, sz);
            auto slice_view    = slice(start, end);
            auto result_view   = result.slice(start, end);
            for (size_t j = start; j < end; j++) {
                result_view[j - start] = chunk_op(slice_view[j - start]);
            }
        });
    }
    return result;
}

template <typename DT, int dim>
template <typename Func>
auto simbi::ndarray<DT, dim>::transform_parallel(
    const ExecutionPolicy<>& policy,
    Func transform_op
) const
    -> std::enable_if_t<
        !has_index_param<Func, const DT&>::value,
        simbi::ndarray<std::invoke_result_t<Func, const DT&>, dim>>
{
    // using result_type = fn_result_t<Func>;
    using result_type = std::invoke_result_t<Func, const DT&>;
    simbi::ndarray<result_type, dim> result(size());
    size_type batch_size  = policy.batch_size;
    size_type num_batches = policy.get_num_batches(sz);

    if constexpr (global::on_gpu) {
        result.copyToGpu();
        auto out_ptr = result.dev_data();
        auto in_ptr  = dev_arr.get();

        simbi::parallel_for(policy, num_batches, [=, this] DEV(size_type i) {
            const size_type start = i * batch_size;
            const size_type end   = my_min(start + batch_size, sz);
            for (size_type j = start; j < end; j++) {
                out_ptr[j] = transform_op(in_ptr[j]);
            }
        });
    }
    else {
        simbi::parallel_for(policy, num_batches, [&](size_type i) {
            const size_type start = i * batch_size;
            const size_type end   = std::min(start + batch_size, sz);
            const auto slice_view = slice(start, end);
            auto result_view      = result.slice(start, end);
            for (size_type j = start; j < end; j++) {
                result_view[j - start] = transform_op(slice_view[j - start]);
            }
        });
    }
    return result;
}

template <typename DT, int dim>
template <typename Func>
auto simbi::ndarray<DT, dim>::transform_parallel(
    const ExecutionPolicy<>& policy,
    Func transform_op
) const
    -> std::enable_if_t<
        has_index_param<Func, const DT&>::value,
        simbi::ndarray<std::invoke_result_t<Func, const DT&, size_type>, dim>>
{
    using result_type = std::invoke_result_t<Func, const DT&, size_type>;
    simbi::ndarray<result_type, dim> result(size());
    size_type batch_size  = policy.batch_size;
    size_type num_batches = policy.get_num_batches(sz);

    if constexpr (global::on_gpu) {
        result.copyToGpu();
        auto out_ptr = result.dev_data();
        auto in_ptr  = dev_arr.get();

        simbi::parallel_for(policy, num_batches, [=, this] DEV(size_type i) {
            const size_type start = i * batch_size;
            const size_type end   = my_min(start + batch_size, sz);
            for (size_type j = start; j < end; j++) {
                out_ptr[j] = transform_op(in_ptr[j], j);
            }
        });
    }
    else {
        simbi::parallel_for(policy, num_batches, [&](size_type i) {
            const size_type start = i * batch_size;
            const size_type end   = std::min(start + batch_size, sz);
            const auto slice_view = slice(start, end);
            auto result_view      = result.slice(start, end);
            for (size_type j = start; j < end; j++) {
                result_view[j - start] = transform_op(slice_view[j - start], j);
            }
        });
    }
    return result;
}

template <typename DT, int dim>
template <typename T, typename Func>
auto simbi::ndarray<DT, dim>::transform_parallel_with(
    const ExecutionPolicy<>& policy,
    simbi::ndarray<T>& other,
    Func transform_op
) const
{
    using result_type = std::invoke_result_t<Func, const DT&, const T&>;
    simbi::ndarray<result_type, dim> result(size());
    size_type batch_size  = policy.batch_size;
    size_type num_batches = policy.get_num_batches(sz);

    if constexpr (global::on_gpu) {
        result.copyToGpu();
        auto out_ptr   = result.dev_data();
        auto in_ptr    = dev_arr.get();
        auto other_ptr = other.dev_data();

        parallel_for(policy, num_batches, [=, this] DEV(size_type i) {
            const size_type start = i * batch_size;
            const size_type end   = my_min(start + batch_size, sz);
            for (size_type j = start; j < end; j++) {
                out_ptr[j] = transform_op(in_ptr[j], other_ptr[j]);
            }
        });
    }
    else {
        parallel_for(policy, num_batches, [&](size_type i) {
            const size_type start = i * batch_size;
            const size_type end   = std::min(start + batch_size, sz);
            const auto slice_view = slice(start, end);
            auto other_view       = other.slice(start, end);
            auto result_view      = result.slice(start, end);
            for (size_type j = start; j < end; j++) {
                result_view[j - start] =
                    transform_op(slice_view[j - start], other_view[j - start]);
            }
        });
    }
    return result;
}

template <typename DT, int dim>
template <typename T, typename Func>
auto simbi::ndarray<DT, dim>::transform_stencil_with(
    const ExecutionPolicy<>& policy,
    const simbi::ndarray<T, dim>& stencil_array,
    size_type radius,
    Func stencil_op
)
{
    // active directional zone size
    const size_type nx = policy.gridSize.x;
    const size_type ny = policy.gridSize.y;
    const size_type nz = policy.gridSize.z;

    // directional strides
    const size_type sx = policy.stride.x;
    const size_type sy = policy.stride.y;
    const size_type sz = policy.stride.z;

    // full grid zone size
    const size_type nxf = nx + 2 * radius;
    const size_type nyf = ny + 2 * radius;
    const size_type nzf = nz + 2 * radius;
    if constexpr (global::on_gpu) {
        auto data_ptr    = dev_data();
        auto stencil_ptr = stencil_array.dev_data();

        parallel_for(policy, [=] DEV(size_type idx) {
            // Get 3D indices
            const size_type kk =
                axid<dim, BlkAx::K>(idx, nx, ny) + radius * (dim > 2);
            const size_type jj =
                axid<dim, BlkAx::J>(idx, nx, ny, kk) + radius * (dim > 1);
            const size_type ii = axid<dim, BlkAx::I>(idx, nx, ny, kk) + radius;

            // Skip ghost cells
            if (ii >= nx || jj >= ny || kk >= nz) {
                return;
            }

            // Create stencil view of input array
            stencil_view<T> view(stencil_ptr, sx, sy, sz, ii, jj, kk, radius);

            // Update array in-place
            data_ptr[idx3(ii, jj, kk, nxf, nyf, nzf)] += stencil_op(view);
        });
    }
    else {
        parallel_for(policy, [&](size_type idx) {
            const size_type kk =
                axid<dim, BlkAx::K>(idx, nx, ny) + radius * (dim > 2);
            const size_type jj =
                axid<dim, BlkAx::J>(idx, nx, ny, kk) + radius * (dim > 1);
            const size_type ii = axid<dim, BlkAx::I>(idx, nx, ny, kk) + radius;

            stencil_view<T>
                view(stencil_array.data(), sx, sy, sz, ii, jj, kk, radius);

            arr[idx3(ii, jj, kk, nxf, nyf, nzf)] += stencil_op(view);
        });
    }
}

template <typename DT, int dim>
template <typename BoundaryOp>
void simbi::ndarray<DT, dim>::apply_to_boundaries(
    const ExecutionPolicy<>& policy,
    size_type radius,
    BoundaryOp&& boundary_op
)
{
    const size_type nx = policy.gridSize.x;
    const size_type ny = policy.gridSize.y;
    const size_type nz = policy.gridSize.z;
    if constexpr (global::on_gpu) {
        auto data_ptr = dev_data();

        parallel_for(policy, [=, this] DEV(size_type idx) {
            // Calculate 3D indices
            const size_type kk = axid<dim, BlkAx::K>(idx, nx, ny);
            const size_type jj = axid<dim, BlkAx::J>(idx, nx, ny, kk);
            const size_type ii = axid<dim, BlkAx::I>(idx, nx, ny, kk);

            // Check if we're on any boundary
            const bool is_boundary =
                is_boundary_point(ii, jj, kk, nx, ny, nz, radius);
            if (!is_boundary) {
                return;
            }

            // Create boundary view
            boundary_view view(data_ptr, nx, ny, nz, ii, jj, kk, radius);

            // Apply boundary operation
            boundary_op(view, data_ptr[idx]);
        });
    }
    else {
        // CPU version
        parallel_for(policy, [&](size_type idx) {
            const size_type kk = axid<dim, BlkAx::K>(idx, nx, ny);
            const size_type jj = axid<dim, BlkAx::J>(idx, nx, ny, kk);
            const size_type ii = axid<dim, BlkAx::I>(idx, nx, ny, kk);

            const bool is_boundary =
                is_boundary_point(ii, jj, kk, nx, ny, nz, radius);
            if (!is_boundary) {
                return;
            }

            // printf("Boundary point: %zu, %zu, %zu\n", ii, jj, kk);
            boundary_view view(arr.get(), nx, ny, nz, ii, jj, kk, radius);
            boundary_op(view, arr[idx]);
        });
    }
}