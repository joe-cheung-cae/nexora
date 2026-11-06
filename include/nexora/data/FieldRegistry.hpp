#pragma once
#include <nexora/data/FieldDescriptor.hpp>
#include <unordered_map>
#include <utility>
namespace nexora {
class FieldRegistry {
  public:
    Status add(FieldDescriptor field) {
        auto valid = field.validate();
        if (!valid.ok())
            return valid;
        if (fields_.count(field.id))
            return {StatusCode::already_exists, "duplicate field id"};
        fields_.emplace(field.id, std::move(field));
        return {};
    }
    const FieldDescriptor* find(FieldId id) const noexcept {
        auto it = fields_.find(id);
        return it == fields_.end() ? nullptr : &it->second;
    }
    std::size_t size() const noexcept {
        return fields_.size();
    }

  private:
    std::unordered_map<FieldId, FieldDescriptor> fields_;
};
} // namespace nexora
