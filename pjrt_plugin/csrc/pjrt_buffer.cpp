#include "pjrt_types.hpp"
#include "paras_pjrt_sycl.hpp"

#include <cstring>
#include <vector>

namespace paras_pjrt_api {
namespace {

std::size_t element_size_bytes(PJRT_Buffer_Type type) {
    switch (type) {
        case PJRT_Buffer_Type_PRED:
        case PJRT_Buffer_Type_S8:
        case PJRT_Buffer_Type_U8:
            return 1;
        case PJRT_Buffer_Type_S16:
        case PJRT_Buffer_Type_U16:
        case PJRT_Buffer_Type_F16:
        case PJRT_Buffer_Type_BF16:
            return 2;
        case PJRT_Buffer_Type_S32:
        case PJRT_Buffer_Type_U32:
        case PJRT_Buffer_Type_F32:
            return 4;
        case PJRT_Buffer_Type_S64:
        case PJRT_Buffer_Type_U64:
        case PJRT_Buffer_Type_F64:
        case PJRT_Buffer_Type_C64:
            return 8;
        case PJRT_Buffer_Type_C128:
            return 16;
        default:
            return 4;
    }
}

void build_default_minor_to_major(std::size_t ndim, std::vector<int64_t>& out) {
    out.resize(ndim);
    for (std::size_t i = 0; i < ndim; ++i) {
        out[i] = static_cast<int64_t>(ndim - 1 - i);
    }
}

bool is_row_major_contiguous(const int64_t* dims, std::size_t ndim,
                             const int64_t* byte_strides, std::size_t num_strides,
                             std::size_t elem_size) {
    if (!byte_strides || num_strides == 0 || ndim == 0) return true;
    int64_t expected = static_cast<int64_t>(elem_size);
    for (std::size_t i = ndim; i-- > 0;) {
        if (dims[i] > 1 && byte_strides[i] != expected) return false;
        expected *= (dims[i] > 0 ? dims[i] : 1);
    }
    return true;
}

void pack_strided_to_contiguous(const char* src, char* dst,
                                const int64_t* dims, const int64_t* strides,
                                std::size_t ndim, std::size_t elem_size) {
    std::vector<int64_t> idx(ndim, 0);
    int64_t total = 1;
    for (std::size_t i = 0; i < ndim; ++i) total *= dims[i];
    for (int64_t linear = 0; linear < total; ++linear) {
        int64_t offset = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            offset += idx[d] * strides[d];
        }
        std::memcpy(dst + linear * elem_size, src + offset, elem_size);
        for (std::size_t d = ndim; d-- > 0;) {
            if (++idx[d] < dims[d]) break;
            idx[d] = 0;
        }
    }
}

} // namespace

PJRT_Error* client_buffer_from_host_buffer(PJRT_Client_BufferFromHostBuffer_Args* args) {
    if (!args || !args->client) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "invalid BufferFromHostBuffer args");
    }

    ParasClient* client = as_impl(args->client);
    ParasDevice* device = args->device ? as_impl(args->device) : nullptr;
    if (!device && args->memory) {
        device = as_impl(args->memory)->device;
    }
    if (!device && !client->device_ptrs.empty()) {
        device = as_impl(client->device_ptrs[0]);
    }
    if (!device) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "no target device");
    }

    const std::size_t elem_size = element_size_bytes(args->type);
    std::size_t num_elems = 1;
    std::vector<int64_t> dims(args->dims, args->dims + args->num_dims);
    for (int64_t d : dims) {
        num_elems *= static_cast<std::size_t>(d);
    }
    const std::size_t nbytes = num_elems * elem_size;

    std::string err_msg;
    void* dev_ptr = paras_pjrt::alloc_device(device->local_hardware_id, nbytes, &err_msg);
    if (!dev_ptr) {
        return make_pjrt_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, err_msg);
    }

    if (nbytes > 0 && args->data) {
        if (is_row_major_contiguous(args->dims, args->num_dims,
                                    args->byte_strides, args->num_byte_strides, elem_size)) {
            if (!paras_pjrt::copy_h2d(device->local_hardware_id, dev_ptr, args->data,
                                      nbytes, true, &err_msg)) {
                paras_pjrt::free_device(device->local_hardware_id, dev_ptr);
                return make_pjrt_error(PJRT_Error_Code_INTERNAL, err_msg);
            }
        } else {
            std::vector<char> packed(nbytes);
            pack_strided_to_contiguous(
                static_cast<const char*>(args->data), packed.data(),
                args->dims, args->byte_strides, args->num_dims, elem_size);
            if (!paras_pjrt::copy_h2d(device->local_hardware_id, dev_ptr, packed.data(),
                                      nbytes, true, &err_msg)) {
                paras_pjrt::free_device(device->local_hardware_id, dev_ptr);
                return make_pjrt_error(PJRT_Error_Code_INTERNAL, err_msg);
            }
        }
    }

    auto* buf = new ParasBuffer();
    buf->client = client;
    buf->device = device;
    buf->memory = device->default_memory;
    buf->element_type = args->type;
    buf->dims = std::move(dims);
    build_default_minor_to_major(buf->dims.size(), buf->minor_to_major);
    buf->data = dev_ptr;
    buf->byte_size = nbytes;

    args->buffer = to_pjrt(buf);
    args->done_with_host_buffer = make_ready_event();
    return nullptr;
}

PJRT_Error* buffer_destroy(PJRT_Buffer_Destroy_Args* args) {
    if (!args || !args->buffer) return nullptr;
    ParasBuffer* buf = as_impl(args->buffer);
    if (!buf->deleted.exchange(true) && buf->data) {
        paras_pjrt::free_device(buf->device->local_hardware_id, buf->data);
        buf->data = nullptr;
    }
    delete buf;
    return nullptr;
}

PJRT_Error* buffer_element_type(PJRT_Buffer_ElementType_Args* args) {
    if (!args || !args->buffer) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null buffer");
    }
    args->type = as_impl(args->buffer)->element_type;
    return nullptr;
}

PJRT_Error* buffer_dimensions(PJRT_Buffer_Dimensions_Args* args) {
    if (!args || !args->buffer) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null buffer");
    }
    ParasBuffer* buf = as_impl(args->buffer);
    args->dims = buf->dims.data();
    args->num_dims = buf->dims.size();
    return nullptr;
}

PJRT_Error* buffer_unpadded_dimensions(PJRT_Buffer_UnpaddedDimensions_Args* args) {
    if (!args || !args->buffer) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null buffer");
    }
    ParasBuffer* buf = as_impl(args->buffer);
    args->unpadded_dims = buf->dims.data();
    args->num_dims = buf->dims.size();
    return nullptr;
}

PJRT_Error* buffer_dynamic_dimension_indices(PJRT_Buffer_DynamicDimensionIndices_Args* args) {
    if (!args || !args->buffer) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null buffer");
    }
    args->dynamic_dim_indices = nullptr;
    args->num_dynamic_dims = 0;
    return nullptr;
}

PJRT_Error* buffer_get_memory_layout(PJRT_Buffer_GetMemoryLayout_Args* args) {
    if (!args || !args->buffer) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null buffer");
    }
    ParasBuffer* buf = as_impl(args->buffer);
    args->layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
    args->layout.tiled.minor_to_major = buf->minor_to_major.data();
    args->layout.tiled.minor_to_major_size = buf->minor_to_major.size();
    args->layout.tiled.tile_dims = nullptr;
    args->layout.tiled.tile_dim_sizes = nullptr;
    args->layout.tiled.num_tiles = 0;
    return nullptr;
}

PJRT_Error* buffer_on_device_size_in_bytes(PJRT_Buffer_OnDeviceSizeInBytes_Args* args) {
    if (!args || !args->buffer) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null buffer");
    }
    args->on_device_size_in_bytes = as_impl(args->buffer)->byte_size;
    return nullptr;
}

PJRT_Error* buffer_device(PJRT_Buffer_Device_Args* args) {
    if (!args || !args->buffer) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null buffer");
    }
    args->device = to_pjrt(as_impl(args->buffer)->device);
    return nullptr;
}

PJRT_Error* buffer_memory(PJRT_Buffer_Memory_Args* args) {
    if (!args || !args->buffer) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null buffer");
    }
    args->memory = to_pjrt(as_impl(args->buffer)->memory);
    return nullptr;
}

PJRT_Error* buffer_delete(PJRT_Buffer_Delete_Args* args) {
    if (!args || !args->buffer) return nullptr;
    ParasBuffer* buf = as_impl(args->buffer);
    if (!buf->deleted.exchange(true) && buf->data) {
        paras_pjrt::free_device(buf->device->local_hardware_id, buf->data);
        buf->data = nullptr;
    }
    return nullptr;
}

PJRT_Error* buffer_is_deleted(PJRT_Buffer_IsDeleted_Args* args) {
    if (!args || !args->buffer) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null buffer");
    }
    args->is_deleted = as_impl(args->buffer)->deleted.load();
    return nullptr;
}

PJRT_Error* buffer_copy_to_device(PJRT_Buffer_CopyToDevice_Args* args) {
    if (!args || !args->buffer || !args->dst_device) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "invalid CopyToDevice args");
    }
    ParasBuffer* src = as_impl(args->buffer);
    ParasDevice* dst_dev = as_impl(args->dst_device);
    std::string err_msg;
    void* dst_ptr = paras_pjrt::alloc_device(
        dst_dev->local_hardware_id, src->byte_size, &err_msg);
    if (!dst_ptr) {
        return make_pjrt_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, err_msg);
    }
    if (!paras_pjrt::copy_d2d(dst_dev->local_hardware_id, dst_ptr, src->data,
                              src->byte_size, &err_msg)) {
        paras_pjrt::free_device(dst_dev->local_hardware_id, dst_ptr);
        return make_pjrt_error(PJRT_Error_Code_INTERNAL, err_msg);
    }
    auto* dst = new ParasBuffer();
    dst->client = src->client;
    dst->device = dst_dev;
    dst->memory = dst_dev->default_memory;
    dst->element_type = src->element_type;
    dst->dims = src->dims;
    dst->minor_to_major = src->minor_to_major;
    dst->data = dst_ptr;
    dst->byte_size = src->byte_size;
    args->dst_buffer = to_pjrt(dst);
    return nullptr;
}

PJRT_Error* buffer_to_host_buffer(PJRT_Buffer_ToHostBuffer_Args* args) {
    if (!args || !args->src) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null src buffer");
    }
    ParasBuffer* buf = as_impl(args->src);
    if (args->dst == nullptr) {
        args->dst_size = buf->byte_size;
        return nullptr;
    }
    if (args->dst_size < buf->byte_size) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT,
                               "dst_size smaller than buffer byte_size");
    }
    if (buf->deleted.load() || !buf->data) {
        return make_pjrt_error(PJRT_Error_Code_FAILED_PRECONDITION, "buffer is deleted");
    }

    std::string err_msg;
    if (!paras_pjrt::copy_d2h(buf->device->local_hardware_id, args->dst, buf->data,
                              buf->byte_size, true, &err_msg)) {
        return make_pjrt_error(PJRT_Error_Code_INTERNAL, err_msg);
    }
    args->event = make_ready_event();
    return nullptr;
}

PJRT_Error* buffer_is_on_cpu(PJRT_Buffer_IsOnCpu_Args* args) {
    if (!args || !args->buffer) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null buffer");
    }
    args->is_on_cpu = false;
    return nullptr;
}

PJRT_Error* buffer_ready_event(PJRT_Buffer_ReadyEvent_Args* args) {
    if (!args || !args->buffer) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null buffer");
    }
    args->event = make_ready_event();
    return nullptr;
}

PJRT_Error* buffer_unsafe_pointer(PJRT_Buffer_UnsafePointer_Args* args) {
    if (!args || !args->buffer) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null buffer");
    }
    args->unsafe_pointer = reinterpret_cast<uintptr_t>(as_impl(args->buffer)->data);
    return nullptr;
}

} // namespace paras_pjrt_api
