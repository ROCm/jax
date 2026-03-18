#pragma once
#include <dlfcn.h>
#include <string>

class SharedLib {
 public:
  // flags default: RTLD_NOW | RTLD_LOCAL
  explicit SharedLib(const std::string& path, int flags=RTLD_NOW | RTLD_LOCAL);
  ~SharedLib();

  //make copy construction and copy assignment operation unavable due to resource leak
  SharedLib(const SharedLib&) = delete;
  SharedLib& operator=(const SharedLib&) = delete;

  // move operators, zero copy from one to another.
  SharedLib(SharedLib&& other) noexcept;
  SharedLib& operator=(SharedLib&& other) noexcept;

  // Load a symbol and cast it to the requested function-pointer type.
  // Example:
  //   using FooFn = int (*)(int, float);
  //   FooFn foo = lib.load<FooFn>("FOO");
  template <class Fn>
  Fn load(const char* symbol) const;

  void* native_handle() const noexcept { return handle_; }
 

 private:
  void* handle_ = nullptr;

  static std::string last_dl_error();
  static void clear_dl_error();
};

#include "shared_libs_template.h"  // template implementation