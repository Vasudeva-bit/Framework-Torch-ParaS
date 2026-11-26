#include "pjrt_types.hpp"
#include "paras_pjrt_sycl.hpp"

#include <limits>
#include <string>

namespace paras_pjrt_api {

PJRT_Error* executable_destroy(PJRT_Executable_Destroy_Args* args) {
    if (!args || !args->executable) return nullptr;
    delete args->executable;
    return nullptr;
}

PJRT_Error* executable_name(PJRT_Executable_Name_Args* args) {
    if (!args || !args->executable) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null executable");
    }
    args->executable_name = args->executable->name.c_str();
    args->executable_name_size = args->executable->name.size();
    return nullptr;
}

PJRT_Error* executable_num_outputs(PJRT_Executable_NumOutputs_Args* args) {
    if (!args || !args->executable) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null executable");
    }
    args->num_outputs = 1;
    return nullptr;
}

PJRT_Error* executable_size_of_generated_code(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args) {
    if (!args || !args->executable) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null executable");
    }
    args->size_in_bytes = 0;
    return nullptr;
}

PJRT_Error* executable_output_element_types(PJRT_Executable_OutputElementTypes_Args* args) {
    if (!args || !args->executable) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null executable");
    }
    args->output_types = args->executable->output_types.data();
    args->num_output_types = args->executable->output_types.size();
    return nullptr;
}

PJRT_Error* executable_output_dimensions(PJRT_Executable_OutputDimensions_Args* args) {
    if (!args || !args->executable) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null executable");
    }
    args->num_outputs = 1;
    args->dims = args->executable->output_dims.data();
    args->dim_sizes = args->executable->output_dim_sizes.data();
    return nullptr;
}

PJRT_Error* executable_output_memory_kinds(PJRT_Executable_OutputMemoryKinds_Args* args) {
    if (!args || !args->executable) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null executable");
    }
    static const char* kDeviceMemKind = "device";
    static const std::size_t kDeviceMemKindSize = 6;
    args->num_outputs = 1;
    args->memory_kinds = &kDeviceMemKind;
    args->memory_kind_sizes = &kDeviceMemKindSize;
    return nullptr;
}

PJRT_Error* executable_fingerprint(PJRT_Executable_Fingerprint_Args* args) {
    if (!args || !args->executable) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null executable");
    }
    args->executable_fingerprint = args->executable->fingerprint.c_str();
    args->executable_fingerprint_size = args->executable->fingerprint.size();
    return nullptr;
}

PJRT_Error* executable_get_cost_analysis(PJRT_Executable_GetCostAnalysis_Args* args) {
    if (!args || !args->executable) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null executable");
    }
    args->properties = nullptr;
    args->num_properties = 0;
    return nullptr;
}

PJRT_Error* executable_get_compiled_memory_stats(PJRT_Executable_GetCompiledMemoryStats_Args* args) {
    if (!args || !args->executable) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null executable");
    }
    args->generated_code_size_in_bytes = 0;
    args->argument_size_in_bytes = 0;
    args->output_size_in_bytes = 0;
    args->alias_size_in_bytes = 0;
    args->temp_size_in_bytes = 0;
    return nullptr;
}

PJRT_Error* loaded_executable_destroy(PJRT_LoadedExecutable_Destroy_Args* args) {
    if (!args || !args->executable) return nullptr;
    delete args->executable;
    return nullptr;
}

PJRT_Error* loaded_executable_get_executable(PJRT_LoadedExecutable_GetExecutable_Args* args) {
    if (!args || !args->loaded_executable) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null loaded_executable");
    }
    // Caller owns the returned PJRT_Executable* and will destroy it via PJRT_Executable_Destroy.
    auto* copy = new PJRT_Executable(*args->loaded_executable->executable);
    args->executable = copy;
    return nullptr;
}

PJRT_Error* loaded_executable_addressable_devices(PJRT_LoadedExecutable_AddressableDevices_Args* args) {
    if (!args || !args->executable) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null loaded_executable");
    }
    args->addressable_devices = args->executable->addressable_devices.data();
    args->num_addressable_devices = args->executable->addressable_devices.size();
    return nullptr;
}

PJRT_Error* loaded_executable_delete(PJRT_LoadedExecutable_Delete_Args* args) {
    if (!args || !args->executable) return nullptr;
    args->executable->deleted = true;
    return nullptr;
}

PJRT_Error* loaded_executable_is_deleted(PJRT_LoadedExecutable_IsDeleted_Args* args) {
    if (!args || !args->executable) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null loaded_executable");
    }
    args->is_deleted = args->executable->deleted;
    return nullptr;
}

PJRT_Error* loaded_executable_execute(PJRT_LoadedExecutable_Execute_Args* args) {
    if (!args || !args->executable) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null loaded_executable");
    }
    PJRT_LoadedExecutable* loaded = args->executable;
    if (args->num_devices != 1) {
        return make_pjrt_error(PJRT_Error_Code_UNIMPLEMENTED,
                               "v0.1 supports single-device execution only");
    }
    if (args->num_args != 2) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT,
                               "MatMul executable expects exactly 2 arguments");
    }

    PJRT_Buffer* buf_a = args->argument_lists[0][0];
    PJRT_Buffer* buf_b = args->argument_lists[0][1];
    if (!buf_a || !buf_b || buf_a->deleted.load() || buf_b->deleted.load()) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "invalid or deleted input buffer");
    }
    if (buf_a->element_type != PJRT_Buffer_Type_F32 ||
        buf_b->element_type != PJRT_Buffer_Type_F32) {
        return make_pjrt_error(PJRT_Error_Code_UNIMPLEMENTED, "v0.1 supports float32 only");
    }

    const MatMulPlan& plan = loaded->executable->plan;
    if (plan.m > std::numeric_limits<int>::max() ||
        plan.n > std::numeric_limits<int>::max() ||
        plan.k > std::numeric_limits<int>::max()) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "GEMM dimension exceeds int32");
    }

    PJRT_Device* device = loaded->addressable_devices[0];
    const std::size_t out_elems = static_cast<std::size_t>(plan.m) * static_cast<std::size_t>(plan.n);
    const std::size_t out_bytes = out_elems * sizeof(float);

    std::string err_msg;
    void* out_ptr = paras_pjrt::alloc_device(device->local_hardware_id, out_bytes, &err_msg);
    if (!out_ptr) {
        return make_pjrt_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, err_msg);
    }

    if (plan.k == 0) {
        if (!paras_pjrt::memset_device(device->local_hardware_id, out_ptr, 0, out_bytes, &err_msg)) {
            paras_pjrt::free_device(device->local_hardware_id, out_ptr);
            return make_pjrt_error(PJRT_Error_Code_INTERNAL, err_msg);
        }
    } else {
        if (!paras_pjrt::gemm_f32(
                device->local_hardware_id,
                buf_a->data, buf_b->data, out_ptr,
                static_cast<int>(plan.m),
                static_cast<int>(plan.n),
                static_cast<int>(plan.k),
                plan.trans_a, plan.trans_b,
                1.0f, 0.0f, &err_msg)) {
            paras_pjrt::free_device(device->local_hardware_id, out_ptr);
            return make_pjrt_error(PJRT_Error_Code_INTERNAL, err_msg);
        }
    }

    auto* out_buf = new PJRT_Buffer();
    out_buf->client = loaded->client;
    out_buf->device = device;
    out_buf->memory = device->default_memory;
    out_buf->element_type = PJRT_Buffer_Type_F32;
    out_buf->dims = plan.out_dims;
    out_buf->minor_to_major = {1, 0};
    out_buf->data = out_ptr;
    out_buf->byte_size = out_bytes;

    args->output_lists[0][0] = out_buf;
    if (args->device_complete_events != nullptr) {
        args->device_complete_events[0] = make_ready_event();
    }
    return nullptr;
}

} // namespace paras_pjrt_api
