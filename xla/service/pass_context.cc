#include "xla/service/pass_context.h"

#include "absl/container/flat_hash_map.h"
#include "absl/types/any.h"
#include "xla/util.h"

namespace xla {
namespace pass_context {

namespace nb = nanobind;

absl::flat_hash_map<std::string, absl::any> current_context;

void SetPassContext(nb::dict dict) {
  for (auto item : dict) {
    std::string name = nb::cast<std::string>(nb::str(item.first));
    absl::any obj;

    if (nb::isinstance<nb::bool_>(item.second)) {
      obj = absl::any(nb::cast<bool>(item.second));
    } else if (nb::isinstance<nb::int_>(item.second)) {
      obj = absl::any(nb::cast<int64_t>(item.second));
    } else if (nb::isinstance<nb::float_>(item.second)) {
      obj = absl::any(nb::cast<double>(item.second));
    } else if (nb::isinstance<nb::str>(item.second)) {
      obj = absl::any(nb::cast<std::string>(item.second));
    } else if (nb::isinstance<nb::list>(item.second) ||
               nb::isinstance<nb::tuple>(item.second)) {
      auto tuple_val = nb::cast<nb::tuple>(item.second);
      // Infer the type according to the first element of the tuple.
      if (tuple_val.size() > 0 && nb::isinstance<nb::int_>(tuple_val[0])) {
        std::vector<int64_t> int_vector;
        int_vector.reserve(tuple_val.size());
        for (size_t i = 0; i < tuple_val.size(); ++i) {
          int_vector.push_back(nb::cast<int64_t>(tuple_val[i]));
        }
        obj = absl::any(std::move(int_vector));
      } else if (tuple_val.size() > 0 &&
                 nb::isinstance<nb::float_>(tuple_val[0])) {
        std::vector<double> double_vector;
        double_vector.reserve(tuple_val.size());
        for (size_t i = 0; i < tuple_val.size(); ++i) {
          double_vector.push_back(nb::cast<double>(tuple_val[i]));
        }
        obj = absl::any(std::move(double_vector));
      } else if (tuple_val.size() > 0 && nb::isinstance<nb::str>(tuple_val[0])) {
        std::vector<std::string> str_vector;
        str_vector.reserve(tuple_val.size());
        for (size_t i = 0; i < tuple_val.size(); ++i) {
          str_vector.push_back(nb::cast<std::string>(tuple_val[i]));
        }
        obj = absl::any(std::move(str_vector));
      } else {
        obj = absl::any(nb::cast<nb::object>(item.second));
      }
    } else {
      obj = absl::any(nb::cast<nb::object>(item.second));
    }

    current_context[name] = std::move(obj);
  }
}

void ClearPassContext() { current_context.clear(); }

template <typename T>
T GetWithDefaultValue(const std::string& name, const T& default_value) {
  auto iter = current_context.find(name);
  if (iter == current_context.end()) {
    return default_value;
  } else {
    try {
      return absl::any_cast<T>(iter->second);
    } catch (const absl::bad_any_cast& e) {
      LOG(FATAL) << "Bad cast of '" << name;
    }
  }
}

template <typename T>
T GetWithoutDefaultValue(const std::string& name) {
  auto iter = current_context.find(name);
  if (iter == current_context.end()) {
    LOG(FATAL) << "Cannot find " << name << " in the pass context";
  } else {
    try {
      return absl::any_cast<T>(iter->second);
    } catch (const absl::bad_any_cast& e) {
      LOG(FATAL) << "Bad cast of '" << name;
    }
  }
}

int64_t GetInt(const std::string& name, int64_t default_value) {
  return GetWithDefaultValue<int64_t>(name, default_value);
}

bool GetBool(const std::string& name, bool default_value) {
  return GetWithDefaultValue<bool>(name, default_value);
}

double GetDouble(const std::string& name) {
  return GetWithoutDefaultValue<double>(name);
}

std::string GetString(const std::string& name,
                      const std::string& default_value) {
  return GetWithDefaultValue<std::string>(name, default_value);
}

std::vector<int64_t> GetIntVector(const std::string& name) {
  return GetWithoutDefaultValue<std::vector<int64_t>>(name);
}

std::vector<double> GetDoubleVector(const std::string& name) {
  return GetWithoutDefaultValue<std::vector<double>>(name);
}

std::vector<std::string> GetStringVector(const std::string& name) {
  return GetWithoutDefaultValue<std::vector<std::string>>(name);
}

nb::object GetPyObject(const std::string& name) {
  return GetWithoutDefaultValue<nb::object>(name);
}

}  // namespace pass_context
}  // namespace xla