#if defined(PARAS_PJRT_BACKEND_SYCL)
#include <sycl/sycl.hpp>
#endif

#include "paras_pjrt_sycl.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <vector>

namespace paras_pjrt {
namespace {

struct DeviceSlot {
    ParasDeviceInfo info;
#if defined(PARAS_PJRT_BACKEND_SYCL)
    std::unique_ptr<sycl::queue> queue;
#endif
};

std::mutex& runtime_mutex() {
    static std::mutex m;
    return m;
}

std::vector<DeviceSlot>& device_slots() {
    static std::vector<DeviceSlot> slots;
    return slots;
}

bool& runtime_initialized() {
    static bool init = false;
    return init;
}

DeviceSlot* get_slot(int device_id, std::string* err_msg) {
    auto& slots = device_slots();
    if (device_id < 0 || static_cast<std::size_t>(device_id) >= slots.size()) {
        if (err_msg) {
            std::ostringstream os;
            os << "Invalid device_id " << device_id << " (count=" << slots.size() << ")";
            *err_msg = os.str();
        }
        return nullptr;
    }
    return &slots[static_cast<std::size_t>(device_id)];
}

} // namespace

int init_runtime(std::string* err_msg) {
    std::lock_guard<std::mutex> lock(runtime_mutex());
    if (runtime_initialized()) {
        return 0;
    }

    auto& slots = device_slots();
    slots.clear();

    bool any_gpu = false;
#if defined(PARAS_PJRT_BACKEND_SYCL)
    try {
        auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
        for (std::size_t i = 0; i < gpus.size(); ++i) {
            DeviceSlot slot;
            slot.info.id = static_cast<int>(i);
            std::string name = gpus[i].get_info<sycl::info::device::name>();
            std::snprintf(slot.info.name, sizeof(slot.info.name), "%s", name.c_str());
            slot.info.compute_units =
                static_cast<int>(gpus[i].get_info<sycl::info::device::max_compute_units>());
            slot.info.global_mem_bytes =
                static_cast<std::size_t>(gpus[i].get_info<sycl::info::device::global_mem_size>());
            slot.info.is_gpu = true;
            slot.queue = std::make_unique<sycl::queue>(
                gpus[i], sycl::property::queue::in_order{});
            slots.push_back(std::move(slot));
            any_gpu = true;
        }
    } catch (const sycl::exception& e) {
        if (err_msg) {
            *err_msg = std::string("SYCL device enumeration failed: ") + e.what();
        }
        return -1;
    }
#endif

    if (slots.empty()) {
        DeviceSlot cpu;
        cpu.info.id = 0;
        std::snprintf(cpu.info.name, sizeof(cpu.info.name), "ParaS CPU Fallback");
        cpu.info.compute_units = 1;
        cpu.info.global_mem_bytes = 1ULL << 30;
        cpu.info.is_gpu = false;
        slots.push_back(std::move(cpu));
    }

    if (!clap_init_backend(any_gpu, err_msg)) {
        return -1;
    }

    runtime_initialized() = true;
    return 0;
}

int device_count() {
    return static_cast<int>(device_slots().size());
}

bool get_device_info(int device_id, ParasDeviceInfo* out) {
    if (!out) return false;
    DeviceSlot* slot = get_slot(device_id, nullptr);
    if (!slot) return false;
    *out = slot->info;
    return true;
}

void* alloc_device(int device_id, std::size_t nbytes, std::string* err_msg) {
    if (nbytes == 0) nbytes = 1;
    DeviceSlot* slot = get_slot(device_id, err_msg);
    if (!slot) return nullptr;

#if defined(PARAS_PJRT_BACKEND_SYCL)
    if (slot->info.is_gpu && slot->queue) {
        try {
            void* p = static_cast<void*>(
                sycl::malloc_shared<std::byte>(nbytes, *slot->queue));
            if (!p && err_msg) {
                std::ostringstream os;
                os << "sycl::malloc_shared(" << nbytes << ") returned nullptr";
                *err_msg = os.str();
            }
            return p;
        } catch (const sycl::exception& e) {
            if (err_msg) *err_msg = std::string("sycl::malloc_shared failed: ") + e.what();
            return nullptr;
        }
    }
#endif

    void* p = nullptr;
    if (posix_memalign(&p, 64, nbytes) != 0 || !p) {
        if (err_msg) *err_msg = "posix_memalign failed";
        return nullptr;
    }
    return p;
}

void free_device(int device_id, void* ptr) {
    if (!ptr) return;
    DeviceSlot* slot = get_slot(device_id, nullptr);
    if (!slot) return;

#if defined(PARAS_PJRT_BACKEND_SYCL)
    if (slot->info.is_gpu && slot->queue) {
        sycl::free(ptr, *slot->queue);
        return;
    }
#endif
    std::free(ptr);
}

bool copy_h2d(int device_id, void* dst_dev, const void* src_host,
              std::size_t nbytes, bool blocking, std::string* err_msg) {
    if (nbytes == 0) return true;
    DeviceSlot* slot = get_slot(device_id, err_msg);
    if (!slot) return false;

#if defined(PARAS_PJRT_BACKEND_SYCL)
    if (slot->info.is_gpu && slot->queue) {
        try {
            slot->queue->memcpy(dst_dev, src_host, nbytes);
            if (blocking) slot->queue->wait();
            return true;
        } catch (const sycl::exception& e) {
            if (err_msg) *err_msg = std::string("copy_h2d failed: ") + e.what();
            return false;
        }
    }
#endif
    (void)blocking;
    std::memcpy(dst_dev, src_host, nbytes);
    return true;
}

bool copy_d2h(int device_id, void* dst_host, const void* src_dev,
              std::size_t nbytes, bool blocking, std::string* err_msg) {
    if (nbytes == 0) return true;
    DeviceSlot* slot = get_slot(device_id, err_msg);
    if (!slot) return false;

#if defined(PARAS_PJRT_BACKEND_SYCL)
    if (slot->info.is_gpu && slot->queue) {
        try {
            slot->queue->wait();
            slot->queue->memcpy(dst_host, src_dev, nbytes);
            if (blocking) slot->queue->wait();
            return true;
        } catch (const sycl::exception& e) {
            if (err_msg) *err_msg = std::string("copy_d2h failed: ") + e.what();
            return false;
        }
    }
#endif
    (void)blocking;
    std::memcpy(dst_host, src_dev, nbytes);
    return true;
}

bool copy_d2d(int device_id, void* dst_dev, const void* src_dev,
              std::size_t nbytes, std::string* err_msg) {
    if (nbytes == 0) return true;
    DeviceSlot* slot = get_slot(device_id, err_msg);
    if (!slot) return false;

#if defined(PARAS_PJRT_BACKEND_SYCL)
    if (slot->info.is_gpu && slot->queue) {
        try {
            slot->queue->memcpy(dst_dev, src_dev, nbytes);
            slot->queue->wait();
            return true;
        } catch (const sycl::exception& e) {
            if (err_msg) *err_msg = std::string("copy_d2d failed: ") + e.what();
            return false;
        }
    }
#endif
    std::memcpy(dst_dev, src_dev, nbytes);
    return true;
}

bool memset_device(int device_id, void* dst_dev, int value,
                   std::size_t nbytes, std::string* err_msg) {
    if (nbytes == 0) return true;
    DeviceSlot* slot = get_slot(device_id, err_msg);
    if (!slot) return false;

#if defined(PARAS_PJRT_BACKEND_SYCL)
    if (slot->info.is_gpu && slot->queue) {
        try {
            slot->queue->memset(dst_dev, value, nbytes);
            slot->queue->wait();
            return true;
        } catch (const sycl::exception& e) {
            if (err_msg) *err_msg = std::string("memset_device failed: ") + e.what();
            return false;
        }
    }
#endif
    std::memset(dst_dev, value, nbytes);
    return true;
}

bool synchronize_device(int device_id, std::string* err_msg) {
    DeviceSlot* slot = get_slot(device_id, err_msg);
    if (!slot) return false;

#if defined(PARAS_PJRT_BACKEND_SYCL)
    if (slot->info.is_gpu && slot->queue) {
        try {
            slot->queue->wait();
        } catch (const sycl::exception& e) {
            if (err_msg) *err_msg = std::string("sycl::queue::wait failed: ") + e.what();
            return false;
        }
    }
#endif
    return true;
}

bool gemm_f32(int device_id,
              const void* a, const void* b, void* c,
              int m, int n, int k,
              bool trans_a, bool trans_b,
              float alpha, float beta,
              std::string* err_msg) {
    // Ensure all pending SYCL H2D copies complete before raw-pointer CLAP GEMM.
    if (!synchronize_device(device_id, err_msg)) {
        return false;
    }
    if (!clap_gemm_f32_raw(a, b, c, m, n, k, trans_a, trans_b, alpha, beta, err_msg)) {
        return false;
    }
    if (!synchronize_device(device_id, err_msg)) {
        return false;
    }
    return true;
}

} // namespace paras_pjrt
