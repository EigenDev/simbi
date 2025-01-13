#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <typeinfo>

// Zero-initialize the array with defined size
template <typename DT>
simbi::ndarray<DT>::ndarray(size_type size)
    : sz(size), nd_capacity(size), dimensions(1)
{
    arr = util::make_unique<DT[]>(nd_capacity);
}

// Initialize the array with a given value
template <typename DT>
simbi::ndarray<DT>::ndarray(size_type size, const DT val)
    : sz(size), nd_capacity(size), dimensions(1)
{
    arr = util::make_unique<DT[]>(size);
    std::fill(arr.get(), arr.get() + sz, val);

    if constexpr (is_ndarray<DT>::value) {
        dimensions += val.ndim();
    }
}

// Copy-constructor for array
template <typename DT>
simbi::ndarray<DT>::ndarray(const ndarray& rhs)
    : sz(rhs.sz), dimensions(rhs.dimensions), arr(new DT[rhs.sz])
{
    copyFromGpu();
    std::copy(rhs.arr.get(), rhs.arr.get() + sz, arr.get());
    copyToGpu();
}

// Move-constructor for array
template <typename DT>
simbi::ndarray<DT>::ndarray(ndarray&& rhs) noexcept
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
template <typename DT>
simbi::ndarray<DT>::ndarray(const std::vector<DT>& rhs)
    : sz(rhs.size()),
      nd_capacity(rhs.capacity()),
      dimensions(1),
      arr(new DT[rhs.size()])
{
    std::copy(rhs.begin(), rhs.end(), arr.get());
}

// Move-constructor for vector
template <typename DT>
simbi::ndarray<DT>::ndarray(std::vector<DT>&& rhs)
    : sz(rhs.size()),
      nd_capacity(rhs.capacity()),
      dimensions(1),
      arr(new DT[rhs.size()])
{
    std::move(rhs.begin(), rhs.end(), arr.get());
}

// Copy the arrays and deallocate the RHS
template <typename DT>
simbi::ndarray<DT>& simbi::ndarray<DT>::operator=(ndarray other)
{
    other.swap(*this);
    return *this;
}

// Copy the arrays and deallocate the RHS
template <typename DT>
constexpr simbi::ndarray<DT>&
simbi::ndarray<DT>::operator+=(const ndarray& other)
{
    simbi::ndarray<DT> newArray(sz + other.sz);
    std::copy(this->arr.get(), this->arr.get() + this->sz, newArray.arr.get());
    std::copy(
        other.arr.get(),
        other.arr.get() + other.sz,
        newArray.arr.get() + this->sz
    );
    newArray.swap(*this);
    return *this;
}

template <typename DT>
void simbi::ndarray<DT>::swap(ndarray& other)
{
    std::swap(arr, other.arr);
    std::swap(sz, other.sz);
    std::swap(nd_capacity, other.nd_capacity);
}

// Template class to insert the element in array
template <typename DT>
constexpr simbi::ndarray<DT>& simbi::ndarray<DT>::push_back(const DT& data)
{
    if (sz == nd_capacity) {
        resize(sz == 0 ? 1 : 2 * sz);
    }
    arr[sz++] = data;
    return *this;
}

// Template class to return the popped element in array
template <typename DT>
constexpr simbi::ndarray<DT>& simbi::ndarray<DT>::pop_back()
{
    if (!empty()) {
        arr[sz - 1].~DT();
        --sz;
    }
    return *this;
}

template <typename DT>
constexpr simbi::ndarray<DT>& simbi::ndarray<DT>::resize(size_type new_size)
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

template <typename DT>
constexpr simbi::ndarray<DT>&
simbi::ndarray<DT>::resize(size_type new_size, const DT new_value)
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
template <typename DT>
constexpr size_type simbi::ndarray<DT>::size() const
{
    return sz;
}

// Template class to return the capacity of array
template <typename DT>
constexpr size_type simbi::ndarray<DT>::capacity() const
{
    return nd_capacity;
}

// Template class to return the number of dimensions of array
template <typename DT>
constexpr size_type simbi::ndarray<DT>::ndim() const
{
    return dimensions;
}

// Template class to return the element of array at given index
template <typename DT>
template <typename IndexType>
DUAL constexpr DT& simbi::ndarray<DT>::operator[](IndexType index)
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
template <typename DT>
template <typename IndexType>
DUAL constexpr DT simbi::ndarray<DT>::operator[](IndexType index) const
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
template <typename DT>
constexpr simbi::ndarray<DT>&
simbi::ndarray<DT>::operator*(const real scale_factor)
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
template <typename DT>
constexpr simbi::ndarray<DT>&
simbi::ndarray<DT>::operator*=(const real scale_factor)
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
template <typename DT>
constexpr simbi::ndarray<DT>&
simbi::ndarray<DT>::operator/(const real scale_factor)
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
template <typename DT>
constexpr simbi::ndarray<DT>&
simbi::ndarray<DT>::operator/=(const real scale_factor)
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
template <typename DT>
typename simbi::ndarray<DT>::iterator simbi::ndarray<DT>::begin() const
{
    return iterator(arr.get());
}

// Template class to return end iterator
template <typename DT>
typename simbi::ndarray<DT>::iterator simbi::ndarray<DT>::end() const
{
    return iterator(arr.get() + sz);
}

// Template class to return the last element
template <typename DT>
DT simbi::ndarray<DT>::back() const
{
    if (empty()) {
        throw std::out_of_range("Array is empty");
    }
    return arr[sz - 1];
}

// Template class to return the last element
template <typename DT>
DT& simbi::ndarray<DT>::back()
{
    if (empty()) {
        throw std::out_of_range("Array is empty");
    }
    return arr[sz - 1];
}

// Template class to return the first element
template <typename DT>
DT& simbi::ndarray<DT>::front()
{
    if (empty()) {
        throw std::out_of_range("Array is empty");
    }
    return arr[0];
}

// Template class to return the first element
template <typename DT>
DT simbi::ndarray<DT>::front() const
{
    if (empty()) {
        throw std::out_of_range("Array is empty");
    }
    return arr[0];
}

// Template class to check if the array is empty
template <typename DT>
bool simbi::ndarray<DT>::empty() const
{
    return sz == 0;
}

// Template class to print the array
template <typename DT>
std::ostream& operator<<(std::ostream& out, const simbi::ndarray<DT>& v)
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
template <typename DT>
void simbi::ndarray<DT>::copyToGpu()
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
template <typename DT>
void simbi::ndarray<DT>::copyFromGpu()
{
    if (dev_arr) {
        gpu::api::copyDevToHost(arr.get(), dev_arr.get(), sz * sizeof(DT));
    }
}

// Template class to copy data between GPUs
template <typename DT>
void simbi::ndarray<DT>::copyBetweenGpu(const ndarray& rhs)
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
template <typename DT>
DT* simbi::ndarray<DT>::host_data()
{
    return arr.get();
}

// Template class to return host data pointer
template <typename DT>
DT* simbi::ndarray<DT>::host_data() const
{
    return arr.get();
}

// Template class to return device data pointer
template <typename DT>
DUAL DT* simbi::ndarray<DT>::dev_data()
{
    return dev_arr.get();
}

// Template class to return device data pointer
template <typename DT>
DUAL DT* simbi::ndarray<DT>::dev_data() const
{
    return dev_arr.get();
}

// Template class to return data pointer
template <typename DT>
DUAL DT* simbi::ndarray<DT>::data()
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
template <typename DT>
DT* simbi::ndarray<DT>::data() const
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

template <typename DT>
void simbi::ndarray<DT>::clear()
{
    arr.reset();
    dev_arr.reset();
    sz = 0;
}

template <typename DT>
void simbi::ndarray<DT>::shrink_to_fit()
{
    if (sz < nd_capacity) {
        auto new_arr = util::make_unique<DT[]>(sz);
        std::copy(arr.get(), arr.get() + sz, new_arr.get());
        arr.swap(new_arr);
        nd_capacity = sz;
    }
}

template <typename DT>
void simbi::ndarray<DT>::reserve(size_type new_capacity)
{
    if (new_capacity > nd_capacity) {
        auto new_arr = util::make_unique<DT[]>(new_capacity);
        std::copy(arr.get(), arr.get() + sz, new_arr.get());
        arr.swap(new_arr);
        nd_capacity = new_capacity;
    }
}

template <typename DT>
void simbi::ndarray<DT>::unpin_memory()
{
    if (dev_arr) {
        gpu::api::hostUnregister(arr.get());
    }
}

template <typename DT>
void simbi::ndarray<DT>::pin_memory()
{
    if (dev_arr) {
        gpu::api::hostRegister(arr.get(), sz * sizeof(DT), 0);
    }
}

template <typename DT>
void simbi::ndarray<DT>::set_stream(simbiStream_t stream)
{
    current_stream = stream;
}

template <typename DT>
void simbi::ndarray<DT>::async_copy_to_gpu()
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

template <typename DT>
void simbi::ndarray<DT>::ensure_gpu_synced()
{
    if constexpr (global::on_gpu) {
        if (needs_gpu_sync && !is_gpu_synced) {
            copyToGpu();
            is_gpu_synced  = true;
            needs_gpu_sync = false;
        }
    }
}

template <typename DT>
void* simbi::ndarray<DT>::aligned_alloc(size_type size, size_type alignment)
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
template <typename DT>
template <typename UnaryFunction>
simbi::ndarray<DT>& simbi::ndarray<DT>::map(UnaryFunction f)
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

template <typename DT>
template <typename UnaryFunction>
simbi::ndarray<DT> simbi::ndarray<DT>::map(UnaryFunction f) const
{
    simbi::ndarray<DT> result;
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

template <typename DT>
template <typename UnaryPredicate>
simbi::ndarray<DT>& simbi::ndarray<DT>::filter(UnaryPredicate pred)
{
    auto new_end = std::remove_if(arr.get(), arr.get() + sz, pred);
    sz           = new_end - arr.get();
    return *this;
}

template <typename DT>
template <typename UnaryPredicate>
simbi::ndarray<DT> simbi::ndarray<DT>::filter(UnaryPredicate pred) const
{
    simbi::ndarray<DT> result;
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

template <typename DT>
template <typename F, typename G>
auto simbi::ndarray<DT>::compose(F f, G g) const
{
    return f(g(*this));
}

template <typename DT>
template <typename... Fs>
auto simbi::ndarray<DT>::then(Fs... f) const
{
    return (... | f)(*this);
}

template <typename DT>
template <typename... Transforms>
simbi::ndarray<DT>
simbi::ndarray<DT>::transform_chain(Transforms... transforms) const
{
    return (... | transforms)(*this);
}

template <typename DT>
template <typename F>
Maybe<simbi::ndarray<DT>> simbi::ndarray<DT>::safe_map(F f) const
{
    if (empty()) {
        return Maybe<simbi::ndarray<DT>>();
    }
    return Maybe<simbi::ndarray<DT>>(map(f));
}

template <typename DT>
template <typename F>
simbi::ndarray<DT>
simbi::ndarray<DT>::combine(const ndarray& other, F binary_op) const
{
    simbi::ndarray<DT> result(size());
    std::transform(begin(), end(), other.begin(), result.begin(), binary_op);
    return result;
}

template <typename DT>
template <typename F>
simbi::ndarray<DT> simbi::ndarray<DT>::parallel_chunks(
    const ExecutionPolicy<>& policy,
    F chunk_op
) const
{
    simbi::ndarray<DT> result(size());
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

template <typename DT>
template <typename NewType, typename F>
simbi::ndarray<NewType> simbi::ndarray<DT>::transform_parallel(
    const ExecutionPolicy<>& policy,
    F transform_op
) const
{
    simbi::ndarray<NewType> result(size());
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