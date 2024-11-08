#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <typeinfo>

// Zero-initialize the array with defined size
template <typename DT, global::Platform build_mode>
simbi::ndarray<DT, build_mode>::ndarray(size_type size)
    : sz(size), nd_capacity(size * sizeof(DT)), dimensions(1)
{
    arr = util::make_unique<DT[]>(nd_capacity);
};

// Initialize the array with a given value
template <typename DT, global::Platform build_mode>
simbi::ndarray<DT, build_mode>::ndarray(size_type size, const DT val)
    : sz(size), nd_capacity(size * sizeof(DT)), dimensions(1)
{
    arr = util::make_unique<DT[]>(size);
    std::fill(arr.get(), arr.get() + sz, val);

    if constexpr (is_ndarray<DT>::value) {
        dimensions += val.ndim();
    }
};

// Copy-constructor for array
template <typename DT, global::Platform build_mode>
simbi::ndarray<DT, build_mode>::ndarray(const ndarray& rhs)
    : sz(rhs.sz), dimensions(rhs.dimensions), arr(new DT[rhs.sz])
{
    copyBetweenGpu(rhs);
    std::copy(rhs.arr.get(), rhs.arr.get() + sz, arr.get());
    copyToGpu();
};

// Copy-constructor for vector
template <typename DT, global::Platform build_mode>
simbi::ndarray<DT, build_mode>::ndarray(const std::vector<DT>& rhs)
    : sz(rhs.size()),
      nd_capacity(rhs.capacity() * sizeof(DT)),
      dimensions(1),
      arr(new DT[rhs.size()])
{
    std::copy(rhs.begin(), rhs.end(), arr.get());
};

// Move-constructor for vector
template <typename DT, global::Platform build_mode>
DUAL simbi::ndarray<DT, build_mode>::ndarray(std::vector<DT>&& rhs)
    : sz(rhs.size()),
      nd_capacity(rhs.capacity() * sizeof(DT)),
      dimensions(1),
      arr(new DT[rhs.size()])
{
    std::move(rhs.begin(), rhs.end(), arr.get());
};

// Copy the arrays and deallocate the RHS
template <typename DT, global::Platform build_mode>
simbi::ndarray<DT, build_mode>&
simbi::ndarray<DT, build_mode>::ndarray::operator=(ndarray other)
{
    other.swap(*this);
    return *this;
};

// Copy the arrays and deallocate the RHS
template <typename DT, global::Platform build_mode>
constexpr simbi::ndarray<DT, build_mode>&
simbi::ndarray<DT, build_mode>::ndarray::operator+=(const ndarray& other)
{
    simbi::ndarray<DT, build_mode> newArray(sz + other.sz);
    std::copy(this->arr.get(), this->arr.get() + this->sz, newArray.arr.get());
    std::copy(
        other.arr.get(),
        other.arr.get() + other.sz,
        newArray.arr.get() + this->sz
    );
    newArray.swap(*this);
    return *this;
};

template <typename DT, global::Platform build_mode>
void simbi::ndarray<DT, build_mode>::swap(ndarray& other)
{
    std::swap(arr, other.arr);
    std::swap(sz, other.sz);
    std::swap(nd_capacity, other.nd_capacity);
}

// Template class to insert the element
// in array
template <typename DT, global::Platform build_mode>
constexpr void simbi::ndarray<DT, build_mode>::push_back(const DT& data)
{
    if (sz == nd_capacity) {
        resize(sz == 0 ? 1 : 2 * sz);
    }
    arr[sz++] = data;
}

// Template class to return the popped element
// in array
template <typename DT, global::Platform build_mode>
constexpr void simbi::ndarray<DT, build_mode>::pop_back()
{
    if (!empty()) {
        arr[sz - 1].~DT();
        --sz;
    }
}

template <typename DT, global::Platform build_mode>
constexpr void simbi::ndarray<DT, build_mode>::resize(size_type new_size)
{
    if (new_size > sz) {
        auto new_arr = util::make_unique<DT[]>(new_size * sizeof(DT));
        std::copy(arr.get(), arr.get() + sz, new_arr.get());
        arr.swap(new_arr);
    }
    sz          = new_size;
    nd_capacity = new_size * sizeof(DT);
}

template <typename DT, global::Platform build_mode>
constexpr void
simbi::ndarray<DT, build_mode>::resize(size_type new_size, const DT new_value)
{
    if (new_size > sz) {
        auto new_arr = util::make_unique<DT[]>(new_size * sizeof(DT));
        std::copy(arr.get(), arr.get() + sz, new_arr.get());
        std::fill(new_arr.get() + sz, new_arr.get() + new_size, new_value);
        arr.swap(new_arr);
    }
    sz          = new_size;
    nd_capacity = new_size * sizeof(DT);
}

// Template class to return the size of
// array
template <typename DT, global::Platform build_mode>
constexpr size_type simbi::ndarray<DT, build_mode>::size() const
{
    return sz;
}

// Template class to return the size of
// array
template <typename DT, global::Platform build_mode>
constexpr size_type simbi::ndarray<DT, build_mode>::capacity() const
{
    return nd_capacity;
}

// Template class to return the size of
// array
template <typename DT, global::Platform build_mode>
constexpr size_type simbi::ndarray<DT, build_mode>::ndim() const
{
    return dimensions;
}

// Template class to return the element of
// array at given index
template <typename DT, global::Platform build_mode>
template <typename IndexType>
DUAL constexpr DT& simbi::ndarray<DT, build_mode>::operator[](IndexType index)
{
    // if given index is greater than the
    // size of array print Error
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

// Template class to return the element of
// array at given index
template <typename DT, global::Platform build_mode>
template <typename IndexType>
DUAL constexpr DT simbi::ndarray<DT, build_mode>::operator[](IndexType index
) const
{
    // if given index is greater than the
    // size of array print Error
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

template <typename DT, global::Platform build_mode>
constexpr simbi::ndarray<DT, build_mode>&
simbi::ndarray<DT, build_mode>::operator*(const real scale_factor)
{
    std::transform(
        arr.get(),
        arr.get() + sz,
        arr.get(),
        [scale_factor](DT& val) { return val * scale_factor; }
    );
    return *this;
};

template <typename DT, global::Platform build_mode>
constexpr simbi::ndarray<DT, build_mode>&
simbi::ndarray<DT, build_mode>::operator*=(const real scale_factor)
{
    std::transform(
        arr.get(),
        arr.get() + sz,
        arr.get(),
        [scale_factor](DT& val) { return val * scale_factor; }
    );
    return *this;
};

template <typename DT, global::Platform build_mode>
constexpr simbi::ndarray<DT, build_mode>&
simbi::ndarray<DT, build_mode>::operator/(const real scale_factor)
{
    std::transform(
        arr.get(),
        arr.get() + sz,
        arr.get(),
        [scale_factor](DT& val) { return val / scale_factor; }
    );
    return *this;
    ;
};

template <typename DT, global::Platform build_mode>
constexpr simbi::ndarray<DT, build_mode>&
simbi::ndarray<DT, build_mode>::operator/=(const real scale_factor)
{
    std::transform(
        arr.get(),
        arr.get() + sz,
        arr.get(),
        [scale_factor](DT& val) { return val / scale_factor; }
    );
    return *this;
};

// Template class to return begin iterator
template <typename DT, global::Platform build_mode>
typename simbi::ndarray<DT, build_mode>::iterator
simbi::ndarray<DT, build_mode>::begin() const
{
    return iterator(arr.get());
}

// Template class to return end iterator
template <typename DT, global::Platform build_mode>
typename simbi::ndarray<DT, build_mode>::iterator
simbi::ndarray<DT, build_mode>::end() const
{
    return iterator(arr.get() + sz);
}

template <typename DT, global::Platform build_mode>
DT simbi::ndarray<DT, build_mode>::back() const
{
    if (empty()) {
        throw std::out_of_range("Array is empty");
    }
    return arr[sz - 1];
}

template <typename DT, global::Platform build_mode>
DT& simbi::ndarray<DT, build_mode>::back()
{
    if (empty()) {
        throw std::out_of_range("Array is empty");
    }
    return arr[sz - 1];
}

template <typename DT, global::Platform build_mode>
DT& simbi::ndarray<DT, build_mode>::front()
{
    if (empty()) {
        throw std::out_of_range("Array is empty");
    }
    return arr[0];
}

template <typename DT, global::Platform build_mode>
DT simbi::ndarray<DT, build_mode>::front() const
{
    if (empty()) {
        throw std::out_of_range("Array is empty");
    }
    return arr[0];
}

template <typename DT, global::Platform build_mode>
bool simbi::ndarray<DT, build_mode>::empty() const
{
    return sz == 0;
}

template <typename DT, global::Platform build_mode>
std::ostream&
operator<<(std::ostream& out, const simbi::ndarray<DT, build_mode>& v)
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

template <typename DT, global::Platform build_mode>
void simbi::ndarray<DT, build_mode>::copyToGpu()
{
    if (arr) {
        if (!dev_arr) {
            dev_arr.reset((DT*) myGpuMalloc(nd_capacity));
        }
        gpu::api::copyHostToDevice(dev_arr.get(), arr.get(), nd_capacity);
    }
}

template <typename DT, global::Platform build_mode>
void simbi::ndarray<DT, build_mode>::copyFromGpu()
{
    if (dev_arr) {
        gpu::api::copyDevToHost(arr.get(), dev_arr.get(), nd_capacity);
    }
}

template <typename DT, global::Platform build_mode>
void simbi::ndarray<DT, build_mode>::copyBetweenGpu(const ndarray& rhs)
{
    if (dev_arr) {
        gpu::api::copyDevToDev(
            dev_arr.get(),
            rhs.dev_arr.get(),
            rhs.nd_capacity
        );
    }
}

template <typename DT, global::Platform build_mode>
DT* simbi::ndarray<DT, build_mode>::host_data()
{
    return arr.get();
};

template <typename DT, global::Platform build_mode>
DT* simbi::ndarray<DT, build_mode>::host_data() const
{
    return arr.get();
};

template <typename DT, global::Platform build_mode>
DUAL DT* simbi::ndarray<DT, build_mode>::dev_data()
{
    return dev_arr.get();
};

template <typename DT, global::Platform build_mode>
DUAL DT* simbi::ndarray<DT, build_mode>::dev_data() const
{
    return dev_arr.get();
};

template <typename DT, global::Platform build_mode>
DUAL DT* simbi::ndarray<DT, build_mode>::data()
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
};

template <typename DT, global::Platform build_mode>
DT* simbi::ndarray<DT, build_mode>::data() const
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
};