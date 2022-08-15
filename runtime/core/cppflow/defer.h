#pragma once
#include <functional>

namespace cppflow {

class defer {
public:
    typedef std::function<void ()> Func;

    explicit defer(const Func& func) : _func(func) {}
    ~defer() {
        _func();
    }

    defer(const defer&) = delete;
    defer(defer&&) = delete;
    defer& operator=(const defer&) = delete;
    void* operator new (size_t) = delete;
    void operator delete (void*) = delete;

private:
    Func _func;
};

} // namespace cppflow
