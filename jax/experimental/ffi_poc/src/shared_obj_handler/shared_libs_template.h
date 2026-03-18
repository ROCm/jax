
#pragma once
#include "shared_libs.h"
#include <dlfcn.h>
#include <stdexcept>

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