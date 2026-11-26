#include "pjrt_types.hpp"
#include "paras_pjrt_sycl.hpp"

#include <cstring>
#include <memory>
#include <sstream>
#include <string>

namespace {

// -----------------------------------------------------------------------------
// Error API
// -----------------------------------------------------------------------------
void error_destroy(PJRT_Error_Destroy_Args* args) {
    if (!args || !args->error) return;
    delete args->error;
}

void error_message(PJRT_Error_Message_Args* args) {
    if (!args || !args->error) return;
    args->message = args->error->message.c_str();
    args->message_size = args->error->message.size();
}

PJRT_Error* error_get_code(PJRT_Error_GetCode_Args* args) {
    if (!args || !args->error) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null error");
    }
    args->code = args->error->code;
    return nullptr;
}

// -----------------------------------------------------------------------------
// Plugin Initialize / Attributes
// -----------------------------------------------------------------------------
PJRT_Error* plugin_initialize(PJRT_Plugin_Initialize_Args* args) {
    (void)args;
    std::string err_msg;
    if (paras_pjrt::init_runtime(&err_msg) != 0) {
        return make_pjrt_error(PJRT_Error_Code_INTERNAL, err_msg);
    }
    return nullptr;
}

PJRT_Error* plugin_attributes(PJRT_Plugin_Attributes_Args* args) {
    if (!args) return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null args");
    args->attributes = nullptr;
    args->num_attributes = 0;
    return nullptr;
}

// -----------------------------------------------------------------------------
// Client API
// -----------------------------------------------------------------------------
PJRT_Error* client_create(PJRT_Client_Create_Args* args) {
    if (!args) return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null args");

    std::string err_msg;
    if (paras_pjrt::init_runtime(&err_msg) != 0) {
        return make_pjrt_error(PJRT_Error_Code_INTERNAL, err_msg);
    }

    auto client = std::make_unique<PJRT_Client>();
    const int count = paras_pjrt::device_count();
    for (int i = 0; i < count; ++i) {
        paras_pjrt::ParasDeviceInfo info;
        paras_pjrt::get_device_info(i, &info);

        auto dev = std::make_unique<PJRT_Device>();
        dev->client = client.get();
        dev->local_hardware_id = i;
        dev->description.id = i;
        dev->description.process_index = 0;
        dev->description.kind = info.is_gpu ? "paras_gpu" : "paras_cpu";
        {
            std::ostringstream os;
            os << "ParaSDevice(id=" << i << ", name=" << info.name << ")";
            dev->description.debug_string = os.str();
            dev->description.to_string = os.str();
        }

        auto mem = std::make_unique<PJRT_Memory>();
        mem->id = i;
        mem->kind = "device";
        mem->kind_id = 0;
        mem->device = dev.get();
        {
            std::ostringstream os;
            os << "ParaSMemory(id=" << i << ", kind=device)";
            mem->debug_string = os.str();
            mem->to_string = os.str();
        }

        dev->default_memory = mem.get();
        client->device_ptrs.push_back(dev.get());
        client->memory_ptrs.push_back(mem.get());
        client->devices.push_back(std::move(dev));
        client->memories.push_back(std::move(mem));
    }

    args->client = client.release();
    return nullptr;
}

PJRT_Error* client_destroy(PJRT_Client_Destroy_Args* args) {
    if (!args || !args->client) return nullptr;
    delete args->client;
    return nullptr;
}

PJRT_Error* client_platform_name(PJRT_Client_PlatformName_Args* args) {
    if (!args || !args->client) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
    }
    args->platform_name = args->client->platform_name.c_str();
    args->platform_name_size = args->client->platform_name.size();
    return nullptr;
}

PJRT_Error* client_process_index(PJRT_Client_ProcessIndex_Args* args) {
    if (!args || !args->client) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
    }
    args->process_index = args->client->process_index;
    return nullptr;
}

PJRT_Error* client_platform_version(PJRT_Client_PlatformVersion_Args* args) {
    if (!args || !args->client) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
    }
    args->platform_version = args->client->platform_version.c_str();
    args->platform_version_size = args->client->platform_version.size();
    return nullptr;
}

PJRT_Error* client_devices(PJRT_Client_Devices_Args* args) {
    if (!args || !args->client) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
    }
    args->devices = args->client->device_ptrs.data();
    args->num_devices = args->client->device_ptrs.size();
    return nullptr;
}

PJRT_Error* client_addressable_devices(PJRT_Client_AddressableDevices_Args* args) {
    if (!args || !args->client) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
    }
    args->addressable_devices = args->client->device_ptrs.data();
    args->num_addressable_devices = args->client->device_ptrs.size();
    return nullptr;
}

PJRT_Error* client_lookup_device(PJRT_Client_LookupDevice_Args* args) {
    if (!args || !args->client) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
    }
    if (args->id < 0 || static_cast<std::size_t>(args->id) >= args->client->device_ptrs.size()) {
        return make_pjrt_error(PJRT_Error_Code_NOT_FOUND, "device id not found");
    }
    args->device = args->client->device_ptrs[static_cast<std::size_t>(args->id)];
    return nullptr;
}

PJRT_Error* client_lookup_addressable_device(PJRT_Client_LookupAddressableDevice_Args* args) {
    if (!args || !args->client) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
    }
    if (args->local_hardware_id < 0 ||
        static_cast<std::size_t>(args->local_hardware_id) >= args->client->device_ptrs.size()) {
        return make_pjrt_error(PJRT_Error_Code_NOT_FOUND, "local_hardware_id not found");
    }
    args->addressable_device =
        args->client->device_ptrs[static_cast<std::size_t>(args->local_hardware_id)];
    return nullptr;
}

PJRT_Error* client_addressable_memories(PJRT_Client_AddressableMemories_Args* args) {
    if (!args || !args->client) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null client");
    }
    args->addressable_memories = args->client->memory_ptrs.data();
    args->num_addressable_memories = args->client->memory_ptrs.size();
    return nullptr;
}

PJRT_Error* client_default_device_assignment(PJRT_Client_DefaultDeviceAssignment_Args* args) {
    if (!args || !args->client || !args->default_assignment) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "invalid DefaultDeviceAssignment args");
    }
    for (std::size_t i = 0; i < args->default_assignment_size; ++i) {
        args->default_assignment[i] = 0;
    }
    return nullptr;
}

// -----------------------------------------------------------------------------
// DeviceDescription & Device API
// -----------------------------------------------------------------------------
PJRT_Error* device_desc_id(PJRT_DeviceDescription_Id_Args* args) {
    if (!args || !args->device_description) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null device_description");
    }
    args->id = args->device_description->id;
    return nullptr;
}

PJRT_Error* device_desc_process_index(PJRT_DeviceDescription_ProcessIndex_Args* args) {
    if (!args || !args->device_description) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null device_description");
    }
    args->process_index = args->device_description->process_index;
    return nullptr;
}

PJRT_Error* device_desc_attributes(PJRT_DeviceDescription_Attributes_Args* args) {
    if (!args || !args->device_description) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null device_description");
    }
    args->attributes = nullptr;
    args->num_attributes = 0;
    return nullptr;
}

PJRT_Error* device_desc_kind(PJRT_DeviceDescription_Kind_Args* args) {
    if (!args || !args->device_description) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null device_description");
    }
    args->device_kind = args->device_description->kind.c_str();
    args->device_kind_size = args->device_description->kind.size();
    return nullptr;
}

PJRT_Error* device_desc_debug_string(PJRT_DeviceDescription_DebugString_Args* args) {
    if (!args || !args->device_description) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null device_description");
    }
    args->debug_string = args->device_description->debug_string.c_str();
    args->debug_string_size = args->device_description->debug_string.size();
    return nullptr;
}

PJRT_Error* device_desc_to_string(PJRT_DeviceDescription_ToString_Args* args) {
    if (!args || !args->device_description) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null device_description");
    }
    args->to_string = args->device_description->to_string.c_str();
    args->to_string_size = args->device_description->to_string.size();
    return nullptr;
}

PJRT_Error* device_get_description(PJRT_Device_GetDescription_Args* args) {
    if (!args || !args->device) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null device");
    }
    args->device_description = &args->device->description;
    return nullptr;
}

PJRT_Error* device_is_addressable(PJRT_Device_IsAddressable_Args* args) {
    if (!args || !args->device) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null device");
    }
    args->is_addressable = true;
    return nullptr;
}

PJRT_Error* device_local_hardware_id(PJRT_Device_LocalHardwareId_Args* args) {
    if (!args || !args->device) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null device");
    }
    args->local_hardware_id = args->device->local_hardware_id;
    return nullptr;
}

PJRT_Error* device_addressable_memories(PJRT_Device_AddressableMemories_Args* args) {
    if (!args || !args->device) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null device");
    }
    args->memories = &args->device->default_memory;
    args->num_memories = 1;
    return nullptr;
}

PJRT_Error* device_default_memory(PJRT_Device_DefaultMemory_Args* args) {
    if (!args || !args->device) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null device");
    }
    args->memory = args->device->default_memory;
    return nullptr;
}

// -----------------------------------------------------------------------------
// Memory API
// -----------------------------------------------------------------------------
PJRT_Error* memory_id(PJRT_Memory_Id_Args* args) {
    if (!args || !args->memory) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null memory");
    }
    args->id = args->memory->id;
    return nullptr;
}

PJRT_Error* memory_kind(PJRT_Memory_Kind_Args* args) {
    if (!args || !args->memory) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null memory");
    }
    args->kind = args->memory->kind.c_str();
    args->kind_size = args->memory->kind.size();
    return nullptr;
}

PJRT_Error* memory_kind_id(PJRT_Memory_Kind_Id_Args* args) {
    if (!args || !args->memory) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null memory");
    }
    args->kind_id = args->memory->kind_id;
    return nullptr;
}

PJRT_Error* memory_debug_string(PJRT_Memory_DebugString_Args* args) {
    if (!args || !args->memory) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null memory");
    }
    args->debug_string = args->memory->debug_string.c_str();
    args->debug_string_size = args->memory->debug_string.size();
    return nullptr;
}

PJRT_Error* memory_to_string(PJRT_Memory_ToString_Args* args) {
    if (!args || !args->memory) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null memory");
    }
    args->to_string = args->memory->to_string.c_str();
    args->to_string_size = args->memory->to_string.size();
    return nullptr;
}

PJRT_Error* memory_addressable_by_devices(PJRT_Memory_AddressableByDevices_Args* args) {
    if (!args || !args->memory) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null memory");
    }
    args->devices = &args->memory->device;
    args->num_devices = 1;
    return nullptr;
}

PJRT_Api build_pjrt_api() {
    PJRT_Api api;
    std::memset(&api, 0, sizeof(api));

    api.struct_size = PJRT_Api_STRUCT_SIZE;
    api.extension_start = nullptr;
    api.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
    api.pjrt_api_version.extension_start = nullptr;
    api.pjrt_api_version.major_version = PJRT_API_MAJOR;
    api.pjrt_api_version.minor_version = PJRT_API_MINOR;

    api.PJRT_Error_Destroy = error_destroy;
    api.PJRT_Error_Message = error_message;
    api.PJRT_Error_GetCode = error_get_code;

    api.PJRT_Plugin_Initialize = plugin_initialize;
    api.PJRT_Plugin_Attributes = plugin_attributes;

    api.PJRT_Event_Destroy = paras_pjrt_api::event_destroy;
    api.PJRT_Event_IsReady = paras_pjrt_api::event_is_ready;
    api.PJRT_Event_Error = paras_pjrt_api::event_error;
    api.PJRT_Event_Await = paras_pjrt_api::event_await;
    api.PJRT_Event_OnReady = paras_pjrt_api::event_on_ready;

    api.PJRT_Client_Create = client_create;
    api.PJRT_Client_Destroy = client_destroy;
    api.PJRT_Client_PlatformName = client_platform_name;
    api.PJRT_Client_ProcessIndex = client_process_index;
    api.PJRT_Client_PlatformVersion = client_platform_version;
    api.PJRT_Client_Devices = client_devices;
    api.PJRT_Client_AddressableDevices = client_addressable_devices;
    api.PJRT_Client_LookupDevice = client_lookup_device;
    api.PJRT_Client_LookupAddressableDevice = client_lookup_addressable_device;
    api.PJRT_Client_AddressableMemories = client_addressable_memories;
    api.PJRT_Client_Compile = paras_pjrt_api::client_compile;
    api.PJRT_Client_DefaultDeviceAssignment = client_default_device_assignment;
    api.PJRT_Client_BufferFromHostBuffer = paras_pjrt_api::client_buffer_from_host_buffer;

    api.PJRT_DeviceDescription_Id = device_desc_id;
    api.PJRT_DeviceDescription_ProcessIndex = device_desc_process_index;
    api.PJRT_DeviceDescription_Attributes = device_desc_attributes;
    api.PJRT_DeviceDescription_Kind = device_desc_kind;
    api.PJRT_DeviceDescription_DebugString = device_desc_debug_string;
    api.PJRT_DeviceDescription_ToString = device_desc_to_string;

    api.PJRT_Device_GetDescription = device_get_description;
    api.PJRT_Device_IsAddressable = device_is_addressable;
    api.PJRT_Device_LocalHardwareId = device_local_hardware_id;
    api.PJRT_Device_AddressableMemories = device_addressable_memories;
    api.PJRT_Device_DefaultMemory = device_default_memory;

    api.PJRT_Memory_Id = memory_id;
    api.PJRT_Memory_Kind = memory_kind;
    api.PJRT_Memory_Kind_Id = memory_kind_id;
    api.PJRT_Memory_DebugString = memory_debug_string;
    api.PJRT_Memory_ToString = memory_to_string;
    api.PJRT_Memory_AddressableByDevices = memory_addressable_by_devices;

    api.PJRT_Executable_Destroy = paras_pjrt_api::executable_destroy;
    api.PJRT_Executable_Name = paras_pjrt_api::executable_name;
    api.PJRT_Executable_NumOutputs = paras_pjrt_api::executable_num_outputs;
    api.PJRT_Executable_SizeOfGeneratedCodeInBytes =
        paras_pjrt_api::executable_size_of_generated_code;
    api.PJRT_Executable_OutputElementTypes = paras_pjrt_api::executable_output_element_types;
    api.PJRT_Executable_OutputDimensions = paras_pjrt_api::executable_output_dimensions;
    api.PJRT_Executable_OutputMemoryKinds = paras_pjrt_api::executable_output_memory_kinds;
    api.PJRT_Executable_Fingerprint = paras_pjrt_api::executable_fingerprint;
    api.PJRT_Executable_GetCostAnalysis = paras_pjrt_api::executable_get_cost_analysis;
    api.PJRT_Executable_GetCompiledMemoryStats =
        paras_pjrt_api::executable_get_compiled_memory_stats;

    api.PJRT_LoadedExecutable_Destroy = paras_pjrt_api::loaded_executable_destroy;
    api.PJRT_LoadedExecutable_GetExecutable = paras_pjrt_api::loaded_executable_get_executable;
    api.PJRT_LoadedExecutable_AddressableDevices =
        paras_pjrt_api::loaded_executable_addressable_devices;
    api.PJRT_LoadedExecutable_Delete = paras_pjrt_api::loaded_executable_delete;
    api.PJRT_LoadedExecutable_IsDeleted = paras_pjrt_api::loaded_executable_is_deleted;
    api.PJRT_LoadedExecutable_Execute = paras_pjrt_api::loaded_executable_execute;

    api.PJRT_Buffer_Destroy = paras_pjrt_api::buffer_destroy;
    api.PJRT_Buffer_ElementType = paras_pjrt_api::buffer_element_type;
    api.PJRT_Buffer_Dimensions = paras_pjrt_api::buffer_dimensions;
    api.PJRT_Buffer_UnpaddedDimensions = paras_pjrt_api::buffer_unpadded_dimensions;
    api.PJRT_Buffer_DynamicDimensionIndices = paras_pjrt_api::buffer_dynamic_dimension_indices;
    api.PJRT_Buffer_GetMemoryLayout = paras_pjrt_api::buffer_get_memory_layout;
    api.PJRT_Buffer_OnDeviceSizeInBytes = paras_pjrt_api::buffer_on_device_size_in_bytes;
    api.PJRT_Buffer_Device = paras_pjrt_api::buffer_device;
    api.PJRT_Buffer_Memory = paras_pjrt_api::buffer_memory;
    api.PJRT_Buffer_Delete = paras_pjrt_api::buffer_delete;
    api.PJRT_Buffer_IsDeleted = paras_pjrt_api::buffer_is_deleted;
    api.PJRT_Buffer_CopyToDevice = paras_pjrt_api::buffer_copy_to_device;
    api.PJRT_Buffer_ToHostBuffer = paras_pjrt_api::buffer_to_host_buffer;
    api.PJRT_Buffer_IsOnCpu = paras_pjrt_api::buffer_is_on_cpu;
    api.PJRT_Buffer_ReadyEvent = paras_pjrt_api::buffer_ready_event;
    api.PJRT_Buffer_UnsafePointer = paras_pjrt_api::buffer_unsafe_pointer;

    return api;
}

} // namespace

extern "C" __attribute__((visibility("default"))) const PJRT_Api* GetPjrtApi() {
    static const PJRT_Api kApi = build_pjrt_api();
    return &kApi;
}
