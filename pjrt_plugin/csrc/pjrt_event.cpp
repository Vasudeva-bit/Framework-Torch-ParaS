#include "pjrt_types.hpp"

PJRT_Event* make_ready_event(PJRT_Error* err) {
    auto* ev = new PJRT_Event();
    ev->ready = true;
    ev->error = err;
    return ev;
}

namespace paras_pjrt_api {

void event_destroy(PJRT_Event_Destroy_Args* args) {
    if (!args || !args->event) return;
    delete args->event;
}

PJRT_Error* event_is_ready(PJRT_Event_IsReady_Args* args) {
    if (!args || !args->event) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null event");
    }
    std::lock_guard<std::mutex> lock(args->event->mu);
    args->is_ready = args->event->ready;
    return nullptr;
}

PJRT_Error* event_error(PJRT_Event_Error_Args* args) {
    if (!args || !args->event) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null event");
    }
    std::lock_guard<std::mutex> lock(args->event->mu);
    if (!args->event->error) return nullptr;
    return make_pjrt_error(args->event->error->code, args->event->error->message);
}

PJRT_Error* event_await(PJRT_Event_Await_Args* args) {
    if (!args || !args->event) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null event");
    }
    std::unique_lock<std::mutex> lock(args->event->mu);
    args->event->cv.wait(lock, [&] { return args->event->ready; });
    if (!args->event->error) return nullptr;
    return make_pjrt_error(args->event->error->code, args->event->error->message);
}

PJRT_Error* event_on_ready(PJRT_Event_OnReady_Args* args) {
    if (!args || !args->event || !args->callback) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "invalid OnReady args");
    }
    bool already_ready = false;
    PJRT_Error* err_copy = nullptr;
    {
        std::lock_guard<std::mutex> lock(args->event->mu);
        if (args->event->ready) {
            already_ready = true;
            if (args->event->error) {
                err_copy = make_pjrt_error(args->event->error->code, args->event->error->message);
            }
        } else {
            args->event->callbacks.emplace_back(args->callback, args->user_arg);
        }
    }
    if (already_ready) {
        args->callback(err_copy, args->user_arg);
    }
    return nullptr;
}

} // namespace paras_pjrt_api
