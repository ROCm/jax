#pragma once

#include <dlfcn.h>
#include <string>
#include <stdexcept>

class SharedLib {
 public:
  // flags default: RTLD_NOW | RTLD_LOCAL
  explicit SharedLib(const std::string& path, int flags=RTLD_NOW | RTLD_LOCAL){
    clear_dl_error();
    handle_ = dlopen(path.c_str(), flags);
    if (!handle_) {
      // dlopen sets an error string retrievable via dlerror()
      throw std::runtime_error("dlopen failed for '" + path + "': " + last_dl_error());
    }
  }

  ~SharedLib(){
    // created with RAII ( Resource Acquisition Is Initialization) design pattern, due to GPU async call execution, this doesnt fit here.
  //if (handle_) {
  //  dlclose(handle_);
  //  handle_ = nullptr;
  //}
  }

  //make copy construction and copy assignment operation unavable due to resource leak
  SharedLib(const SharedLib&) = delete;
  SharedLib& operator=(const SharedLib&) = delete;

  // move operators, zero copy from one to another.
  SharedLib(SharedLib&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  SharedLib& operator=(SharedLib&& other) noexcept {
    if (this != &other) {
      if (handle_) dlclose(handle_);
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  // Load a symbol and cast it to the requested function-pointer type.
  // Example:
  //   using FooFn = int (*)(int, float);
  //   FooFn foo = lib.load<FooFn>("FOO");
  template <class Fn>
  Fn load(const char* symbol) const;

  void* native_handle() const noexcept { return handle_; }
 

 private:
  void* handle_ = nullptr;

  static std::string last_dl_error(){
    const char* e = dlerror();
    return e ? std::string(e) : std::string();
  }
  static void clear_dl_error(){
    (void)dlerror();  // clears any existing error
  }
};


template <class Fn>
Fn SharedLib::load(const char* symbol) const {
  if (!handle_) {
    throw std::runtime_error("SharedLib::load called on null handle");
  }

  clear_dl_error();
  void* sym = dlsym(handle_, symbol);

  // Must call dlerror() to determine if dlsym failed.
  if (const char* err = dlerror(); err != nullptr) {
    throw std::runtime_error(std::string("dlsym failed for '") + symbol + "': " + err);
  }

  return reinterpret_cast<Fn>(sym);
}