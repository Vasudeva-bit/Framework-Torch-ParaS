#include "pjrt_types.hpp"

PJRT_Event* make_ready_event(ParasError* err) {
    auto* ev = new ParasEvent();
    ev->ready = true;
    ev->error = err;
    return to_pjrt(ev);
}

namespace paras_pjrt_api {

PJRT_Error* event_destroy(PJRT_Event_Destroy_Args* args) {
    if (!args || !args->event) return nullptr;
    delete as_impl(args->event);
    return nullptr;
}

PJRT_Error* event_is_ready(PJRT_Event_IsReady_Args* args) {
    if (!args || !args->event) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null event");
    }
    ParasEvent* ev = as_impl(args->event);
    std::lock_guard<std::mutex> lock(ev->mu);
    args->is_ready = ev->ready;
    return nullptr;
}

PJRT_Error* event_error(PJRT_Event_Error_Args* args) {
    if (!args || !args->event) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null event");
    }
    ParasEvent* ev = as_impl(args->event);
    std::lock_guard<std::mutex> lock(ev->mu);
    if (!ev->error) return nullptr;
    return make_pjrt_error(ev->error->code, ev->error->message);
}

PJRT_Error* event_await(PJRT_Event_Await_Args* args) {
    if (!args || !args->event) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "null event");
    }
    ParasEvent* ev = as_impl(args->event);
    std::unique_lock<std::mutex> lock(ev->mu);
    ev->cv.wait(lock, [&] { return ev->ready; });
    if (!ev->error) return nullptr;
    return make_pjrt_error(ev->error->code, ev->error->message);
}

PJRT_Error* event_on_ready(PJRT_Event_OnReady_Args* args) {
    if (!args || !args->event || !args->callback) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "invalid OnReady args");
    }
    ParasEvent* ev = as_impl(args->event);
    bool already_ready = false;
    PJRT_Error* err_copy = nullptr;
    {
        std::lock_guard<std::mutex> lock(ev->mu);
        if (ev->ready) {
            already_ready = true;
            if (ev->error) {
                err_copy = make_pjrt_error(ev->error->code, ev->error->message);
            }
        } else {
            ev->callbacks.emplace_back(args->callback, args->user_arg);
        }
    }
    if (already_ready) {
        args->callback(err_copy, args->user_arg);
    }
    return nullptr;
}

} // namespace paras_pjrt_api
