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

struct ParasError {
    PJRT_Error_Code code = PJRT_Error_Code_OK;
    std::string message;
};

struct ParasEvent {
    std::mutex mu;
    std::condition_variable cv;
    bool ready = true;
    ParasError* error = nullptr;
    std::vector<std::pair<PJRT_Event_OnReadyCallback, void*>> callbacks;
};

struct ParasDeviceDescription {
    int id = 0;
    int process_index = 0;
    std::string kind = "paras_gpu";
    std::string debug_string;
    std::string to_string;
};

struct ParasClient;
struct ParasMemory;

struct ParasDevice {
    ParasDeviceDescription description;
    ParasClient* client = nullptr;
    ParasMemory* default_memory = nullptr;
    int local_hardware_id = 0;
};

struct ParasMemory {
    int id = 0;
    std::string kind = "device";
    int kind_id = 0;
    std::string debug_string;
    std::string to_string;
    ParasDevice* device = nullptr;
};

struct ParasBuffer {
    ParasClient* client = nullptr;
    ParasDevice* device = nullptr;
    ParasMemory* memory = nullptr;
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

struct ParasExecutable {
    std::string name = "paras_matmul";
    MatMulPlan plan;
    std::vector<PJRT_Buffer_Type> output_types = {PJRT_Buffer_Type_F32};
    std::vector<int64_t> output_dims;
    std::vector<std::size_t> output_dim_sizes;
    std::string output_memory_kind = "device";
    std::string fingerprint = "paras_matmul_v01";
};

struct ParasLoadedExecutable {
    ParasClient* client = nullptr;
    std::unique_ptr<ParasExecutable> executable;
    std::vector<PJRT_Device*> addressable_devices;
    bool deleted = false;
};

struct ParasClient {
    std::string platform_name = "paras";
    std::string platform_version = "0.1.0";
    int process_index = 0;
    std::vector<std::unique_ptr<ParasDevice>> devices;
    std::vector<std::unique_ptr<ParasMemory>> memories;
    std::vector<PJRT_Device*> device_ptrs;
    std::vector<PJRT_Memory*> memory_ptrs;
};

// Opaque handle cast helpers (avoids struct redefinition against pjrt_c_api.h)
inline PJRT_Error* make_pjrt_error(PJRT_Error_Code code, const std::string& msg) {
    auto* err = new ParasError();
    err->code = code;
    err->message = msg;
    return reinterpret_cast<PJRT_Error*>(err);
}

inline ParasError* as_impl(PJRT_Error* p) { return reinterpret_cast<ParasError*>(p); }
inline const ParasError* as_impl(const PJRT_Error* p) { return reinterpret_cast<const ParasError*>(p); }
inline ParasEvent* as_impl(PJRT_Event* p) { return reinterpret_cast<ParasEvent*>(p); }
inline ParasDeviceDescription* as_impl(PJRT_DeviceDescription* p) { return reinterpret_cast<ParasDeviceDescription*>(p); }
inline ParasDevice* as_impl(PJRT_Device* p) { return reinterpret_cast<ParasDevice*>(p); }
inline ParasMemory* as_impl(PJRT_Memory* p) { return reinterpret_cast<ParasMemory*>(p); }
inline ParasBuffer* as_impl(PJRT_Buffer* p) { return reinterpret_cast<ParasBuffer*>(p); }
inline ParasExecutable* as_impl(PJRT_Executable* p) { return reinterpret_cast<ParasExecutable*>(p); }
inline ParasLoadedExecutable* as_impl(PJRT_LoadedExecutable* p) { return reinterpret_cast<ParasLoadedExecutable*>(p); }
inline ParasClient* as_impl(PJRT_Client* p) { return reinterpret_cast<ParasClient*>(p); }

inline PJRT_Event* to_pjrt(ParasEvent* p) { return reinterpret_cast<PJRT_Event*>(p); }
inline PJRT_DeviceDescription* to_pjrt(ParasDeviceDescription* p) { return reinterpret_cast<PJRT_DeviceDescription*>(p); }
inline PJRT_Device* to_pjrt(ParasDevice* p) { return reinterpret_cast<PJRT_Device*>(p); }
inline PJRT_Memory* to_pjrt(ParasMemory* p) { return reinterpret_cast<PJRT_Memory*>(p); }
inline PJRT_Buffer* to_pjrt(ParasBuffer* p) { return reinterpret_cast<PJRT_Buffer*>(p); }
inline PJRT_Executable* to_pjrt(ParasExecutable* p) { return reinterpret_cast<PJRT_Executable*>(p); }
inline PJRT_LoadedExecutable* to_pjrt(ParasLoadedExecutable* p) { return reinterpret_cast<PJRT_LoadedExecutable*>(p); }
inline PJRT_Client* to_pjrt(ParasClient* p) { return reinterpret_cast<PJRT_Client*>(p); }

PJRT_Event* make_ready_event(ParasError* err = nullptr);

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
