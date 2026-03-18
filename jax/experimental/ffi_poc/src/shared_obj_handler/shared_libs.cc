#include "shared_libs.h"

#include <dlfcn.h>
#include <stdexcept>
#include <utility>

void SharedLib::clear_dl_error() {
  (void)dlerror();  // clears any existing error
}

std::string SharedLib::last_dl_error() {
  const char* e = dlerror();
  return e ? std::string(e) : std::string();
}

SharedLib::SharedLib(const std::string& path, int flags) {
  clear_dl_error();
  handle_ = dlopen(path.c_str(), flags);
  if (!handle_) {
    // dlopen sets an error string retrievable via dlerror()
    throw std::runtime_error("dlopen failed for '" + path + "': " + last_dl_error());
  }
}

SharedLib::~SharedLib() {
  // created with RAII ( Resource Acquisition Is Initialization) design pattern, due to GPU async call execution, this doesnt fit here.
  //if (handle_) {
  //  dlclose(handle_);
  //  handle_ = nullptr;
  //}
}

SharedLib::SharedLib(SharedLib&& other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

SharedLib& SharedLib::operator=(SharedLib&& other) noexcept {
  if (this != &other) {
    if (handle_) dlclose(handle_);
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}