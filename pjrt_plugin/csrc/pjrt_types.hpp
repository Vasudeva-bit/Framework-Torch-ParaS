#pragma once

#include "pjrt_c_api.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

struct PJRT_Error {
    PJRT_Error_Code code = PJRT_Error_Code_OK;
    std::string message;
};

inline PJRT_Error* make_pjrt_error(PJRT_Error_Code code, const std::string& msg) {
    auto* err = new PJRT_Error();
    err->code = code;
    err->message = msg;
    return err;
}

struct PJRT_Event {
    std::mutex mu;
    std::condition_variable cv;
    bool ready = true;
    PJRT_Error* error = nullptr;
    std::vector<std::pair<PJRT_Event_OnReadyCallback, void*>> callbacks;
};

PJRT_Event* make_ready_event(PJRT_Error* err = nullptr);

struct PJRT_DeviceDescription {
    int id = 0;
    int process_index = 0;
    std::string kind = "paras_gpu";
    std::string debug_string;
    std::string to_string;
};

struct PJRT_Memory;

struct PJRT_Device {
    PJRT_DeviceDescription description;
    PJRT_Client* client = nullptr;
    PJRT_Memory* default_memory = nullptr;
    int local_hardware_id = 0;
};

struct PJRT_Memory {
    int id = 0;
    std::string kind = "device";
    int kind_id = 0;
    std::string debug_string;
    std::string to_string;
    PJRT_Device* device = nullptr;
};

struct PJRT_Buffer {
    PJRT_Client* client = nullptr;
    PJRT_Device* device = nullptr;
    PJRT_Memory* memory = nullptr;
    PJRT_Buffer_Type element_type = PJRT_Buffer_Type_F32;
    std::vector<int64_t> dims;
    std::vector<int64_t> minor_to_major;
    void* data = nullptr;
    std::size_t byte_size = 0;
    std::atomic<bool> deleted{false};
};

struct MatMulPlan {
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    bool trans_a = false;
    bool trans_b = false;
    std::vector<int64_t> out_dims;
};

struct PJRT_Executable {
    std::string name = "paras_matmul";
    MatMulPlan plan;
    std::vector<PJRT_Buffer_Type> output_types = {PJRT_Buffer_Type_F32};
    std::vector<int64_t> output_dims;
    std::vector<std::size_t> output_dim_sizes;
    std::string output_memory_kind = "device";
    std::string fingerprint = "paras_matmul_v01";
};

struct PJRT_LoadedExecutable {
    PJRT_Client* client = nullptr;
    std::unique_ptr<PJRT_Executable> executable;
    std::vector<PJRT_Device*> addressable_devices;
    bool deleted = false;
};

struct PJRT_Client {
    std::string platform_name = "paras";
    std::string platform_version = "0.1.0";
    int process_index = 0;
    std::vector<std::unique_ptr<PJRT_Device>> devices;
    std::vector<std::unique_ptr<PJRT_Memory>> memories;
    std::vector<PJRT_Device*> device_ptrs;
    std::vector<PJRT_Memory*> memory_ptrs;
};

namespace paras_pjrt_api {

// Event functions
void        event_destroy(PJRT_Event_Destroy_Args* args);
PJRT_Error* event_is_ready(PJRT_Event_IsReady_Args* args);
PJRT_Error* event_error(PJRT_Event_Error_Args* args);
PJRT_Error* event_await(PJRT_Event_Await_Args* args);
PJRT_Error* event_on_ready(PJRT_Event_OnReady_Args* args);

// Buffer functions
PJRT_Error* client_buffer_from_host_buffer(PJRT_Client_BufferFromHostBuffer_Args* args);
PJRT_Error* buffer_destroy(PJRT_Buffer_Destroy_Args* args);
PJRT_Error* buffer_element_type(PJRT_Buffer_ElementType_Args* args);
PJRT_Error* buffer_dimensions(PJRT_Buffer_Dimensions_Args* args);
PJRT_Error* buffer_unpadded_dimensions(PJRT_Buffer_UnpaddedDimensions_Args* args);
PJRT_Error* buffer_dynamic_dimension_indices(PJRT_Buffer_DynamicDimensionIndices_Args* args);
PJRT_Error* buffer_get_memory_layout(PJRT_Buffer_GetMemoryLayout_Args* args);
PJRT_Error* buffer_on_device_size_in_bytes(PJRT_Buffer_OnDeviceSizeInBytes_Args* args);
PJRT_Error* buffer_device(PJRT_Buffer_Device_Args* args);
PJRT_Error* buffer_memory(PJRT_Buffer_Memory_Args* args);
PJRT_Error* buffer_delete(PJRT_Buffer_Delete_Args* args);
PJRT_Error* buffer_is_deleted(PJRT_Buffer_IsDeleted_Args* args);
PJRT_Error* buffer_copy_to_device(PJRT_Buffer_CopyToDevice_Args* args);
PJRT_Error* buffer_to_host_buffer(PJRT_Buffer_ToHostBuffer_Args* args);
PJRT_Error* buffer_is_on_cpu(PJRT_Buffer_IsOnCpu_Args* args);
PJRT_Error* buffer_ready_event(PJRT_Buffer_ReadyEvent_Args* args);
PJRT_Error* buffer_unsafe_pointer(PJRT_Buffer_UnsafePointer_Args* args);

// Compile & Executable functions
PJRT_Error* client_compile(PJRT_Client_Compile_Args* args);
PJRT_Error* executable_destroy(PJRT_Executable_Destroy_Args* args);
PJRT_Error* executable_name(PJRT_Executable_Name_Args* args);
PJRT_Error* executable_num_outputs(PJRT_Executable_NumOutputs_Args* args);
PJRT_Error* executable_size_of_generated_code(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args);
PJRT_Error* executable_output_element_types(PJRT_Executable_OutputElementTypes_Args* args);
PJRT_Error* executable_output_dimensions(PJRT_Executable_OutputDimensions_Args* args);
PJRT_Error* executable_output_memory_kinds(PJRT_Executable_OutputMemoryKinds_Args* args);
PJRT_Error* executable_fingerprint(PJRT_Executable_Fingerprint_Args* args);
PJRT_Error* executable_get_cost_analysis(PJRT_Executable_GetCostAnalysis_Args* args);
PJRT_Error* executable_get_compiled_memory_stats(PJRT_Executable_GetCompiledMemoryStats_Args* args);
PJRT_Error* loaded_executable_destroy(PJRT_LoadedExecutable_Destroy_Args* args);
PJRT_Error* loaded_executable_get_executable(PJRT_LoadedExecutable_GetExecutable_Args* args);
PJRT_Error* loaded_executable_addressable_devices(PJRT_LoadedExecutable_AddressableDevices_Args* args);
PJRT_Error* loaded_executable_delete(PJRT_LoadedExecutable_Delete_Args* args);
PJRT_Error* loaded_executable_is_deleted(PJRT_LoadedExecutable_IsDeleted_Args* args);
PJRT_Error* loaded_executable_execute(PJRT_LoadedExecutable_Execute_Args* args);

} // namespace paras_pjrt_api
