/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PY_BUFFER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PY_BUFFER_H_

#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"
#include "xla/python/py_client.h"
#include "xla/python/traceback.h"
#include "xla/statusor.h"
#include "xla/types.h"

namespace xla {

namespace nb = nanobind;

// Python wrapper around PjRtBuffer. We use a wrapper class:
// a) to keep the PjRtClient alive via a std::shared_ptr<>
// b) to add Python-specific functionality.
//
// A `PyBuffer` can be used from Python without being wrapped in a Python
// `DeviceArray` object.
class PyBuffer {
 public:
  // nb::object typed subclass for PyBuffer objects.
  class pyobject : public nb::object {
   public:
    NB_OBJECT(pyobject,  // NOLINT
              nb::object, "PyBuffer", PyBuffer::IsPyBuffer);
    pyobject() = default;
    PyBuffer* buf() const { return PyBuffer::AsPyBufferUnchecked(*this); }
  };
  using object = pyobject;

  static object Make(std::shared_ptr<PyClient> client,
                     std::shared_ptr<PjRtBuffer> buffer,
                     std::shared_ptr<Traceback> traceback);

  // Returns true if `h` is a PyBuffer.
  static bool IsPyBuffer(nb::handle handle);
  // Converts `handle` to a PyBuffer*. Does not do any checking.
  static PyBuffer* AsPyBufferUnchecked(nb::handle handle);
  // Converts `handle` to a PyBuffer*. Returns an error status if
  // !IsPyBuffer(handle)
  static StatusOr<PyBuffer*> AsPyBuffer(nb::handle handle);

  // Gets a Python handle to an existing PyBuffer. Assumes the PyObject was
  // allocated on the Python heap, which is the case if Make() was used.
  nb::handle AsHandle();

  ~PyBuffer();

  std::shared_ptr<PyClient> client() const { return client_; }
  PjRtBuffer* buffer() const { return buffer_.get(); }
  std::shared_ptr<PjRtBuffer> shared_ptr_buffer() const { return buffer_; }

  nb_class_ptr<PjRtDevice> device() const;
  absl::string_view platform_name() const {
    return buffer_->client()->platform_name();
  }
  bool is_deleted() const { return buffer_->IsDeleted(); }

  StatusOr<nb::object> CopyToDevice(
      const nb_class_ptr<PjRtDevice>& dst_device) const;
  std::pair<Status, bool> CopyToRemoteDevice(
      absl::string_view serialized_descriptor) const;

  StatusOr<size_t> OnDeviceSizeInBytes() {
    return buffer_->GetOnDeviceSizeInBytes();
  }

  void Delete() {
    buffer_->Delete();
    host_value_ = nullptr;
  }

  // Makes a copy of this PyBuffer object that shares the underlying PjRtBuffer.
  // This is useful because we may wish to change JAX metadata (e.g., the sticky
  // device) without copying the buffer.
  object Clone() const;

  // Returns xla::InvalidArgument if the buffer has been deleted.
  // See `PjRtFuture` for the semantics of `IsReady` and `IsKnownReady`.
  StatusOr<bool> IsReady() {
    if (buffer_->IsDeleted()) {
      return InvalidArgument("DeviceArray has been deleted.");
    }
    return buffer_->GetReadyFuture().IsReady();
  }
  StatusOr<bool> IsKnownReady() {
    if (buffer_->IsDeleted()) {
      return InvalidArgument("DeviceArray has been deleted.");
    }
    return buffer_->GetReadyFuture().IsKnownReady();
  }

  // Returns xla::InvalidArgument if the buffer has been deleted.
  Status BlockHostUntilReady();
  Status CopyToHostAsync();

  const Shape& shape() { return buffer_->on_device_shape(); }

  StatusOr<std::uintptr_t> UnsafeBufferPointer() const;

  // Implementation of the CUDA array interface for sharing GPU buffers with
  // other Python libraries.
  StatusOr<nb::dict> CudaArrayInterface();

  const std::shared_ptr<Traceback>& traceback() const { return traceback_; }

  // Returns the size (i.e. number of elements) of the (host) numpy array.
  StatusOr<int64_t> size();

  // Returns the number of dimensions of the (host) numpy array.
  int ndim() const { return buffer()->on_device_shape().dimensions_size(); }

  nb::tuple python_shape() const;
  xla::nb_dtype python_dtype() const;

  // Representing the logical view of the underlying dynamic shapes.
  StatusOr<const Shape*> xla_dynamic_shape();

  Status set_sticky_device(PjRtDevice* sticky_device) {
    TF_RET_CHECK(sticky_device == nullptr ||
                 sticky_device == buffer_->device());
    sticky_device_ = sticky_device;
    return OkStatus();
  }
  PjRtDevice* sticky_device() const { return sticky_device_; }

  void set_weak_type(std::optional<bool> weak_type) { weak_type_ = weak_type; }
  std::optional<bool> weak_type() const { return weak_type_; }

  StatusOr<nb::object> AsNumPyArray(nb::handle this_obj);

  void SetAval(nb::object aval) { aval_ = aval; }
  nb::object GetAval() const { return aval_; }

  static Status RegisterTypes(nb::module_& m);
  static PyObject* base_type() { return base_type_; }
  static PyObject* type() { return type_; }

 private:
  // PyBuffer objects must not be allocated directly since they must always live
  // on the Python heap. Use Make() instead.
  PyBuffer(std::shared_ptr<PyClient> client, std::shared_ptr<PjRtBuffer> buffer,
           std::shared_ptr<Traceback> traceback);

  static PyObject* base_type_;
  static PyObject* type_;

  friend class PyClient;

  struct HostValue {
    absl::Notification ready;
    Status status;
    std::shared_ptr<xla::Literal> value;
  };
  std::shared_ptr<PyClient> client_;
  std::shared_ptr<PjRtBuffer> buffer_;
  std::shared_ptr<Traceback> traceback_;
  std::shared_ptr<HostValue> host_value_;  // Protected by the GIL.

  // JAX uses this field to record whether a buffer is committed to a particular
  // device by the user (https://github.com/google/jax/pull/1916).
  PjRtDevice* sticky_device_ = nullptr;

  // TODO(phawkins): consider not keeping an explicit aval on C++ buffer
  // objects.
  nb::object aval_ = nb::none();

  // An optional weak type. If absent, the JAX jit code computes the weak_type
  // from the aval_.weak_type attribute. This is a backwards compatibility
  // measure for older Python code that does not set weak_type explicitly.
  // TODO(phawkins): drop support for older jax Python versions and make
  // weak_type mandatory.
  std::optional<bool> weak_type_ = std::nullopt;

  std::optional<Shape> dynamic_shape_ = std::nullopt;
  // Doubly-linked list of all PyBuffers known to the client. Protected by the
  // GIL. Since multiple PyBuffers may share the same PjRtBuffer, there may be
  // duplicate PjRtBuffers in this list.
  PyBuffer* next_;
  PyBuffer* prev_;
};

// A batched version of python wrapper around a list of PjRtBuffers.
class PyShardedBuffer {
 public:
  static PyShardedBuffer CreateFromPyBuffers(
      absl::Span<const PyBuffer::object> py_buffers);

  PyShardedBuffer(std::shared_ptr<PyClient> client,
                  std::vector<std::shared_ptr<PjRtBuffer>> buffers,
                  std::shared_ptr<Traceback> traceback, bool sticky = false)
      : client_(std::move(client)),
        buffers_(std::move(buffers)),
        traceback_(std::move(traceback)),
        sticky_(sticky) {}

  std::vector<PyBuffer::object> GetPyBuffers() const {
    std::vector<PyBuffer::object> results;
    results.reserve(buffers_.size());
    for (const auto& pjrt_buffer : buffers_) {
      auto py_buffer = PyBuffer::Make(client_, pjrt_buffer, traceback_);
      if (sticky_) {
        TF_CHECK_OK(py_buffer.buf()->set_sticky_device(pjrt_buffer->device()));
      }
      results.push_back(std::move(py_buffer));
    }
    return results;
  }

  PyBuffer::object GetPyBuffer(int device_id) const {
    const auto& pjrt_buffer = buffers_.at(device_id);
    auto py_buffer = PyBuffer::Make(client_, pjrt_buffer, traceback_);
    if (sticky_) {
      TF_CHECK_OK(py_buffer.buf()->set_sticky_device(pjrt_buffer->device()));
    }
    return py_buffer;
  }

  PrimitiveType dtype() const {
    return buffers_.at(0)->on_device_shape().element_type();
  }

  PjRtBuffer* GetPjRtBuffer(int device_id) const {
    return buffers_.at(device_id).get();
  }

  int num_devices() const { return buffers_.size(); }

  Status BlockHostUntilReady();

 private:
  std::shared_ptr<PyClient> client_;
  std::vector<std::shared_ptr<PjRtBuffer>> buffers_;
  std::shared_ptr<Traceback> traceback_;
  bool sticky_ = false;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_BUFFER_H_
