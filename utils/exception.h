#pragma once

#include <exception>
#include <string>

namespace OptiSik {

/// Exception class for computation errors
class computationError : public std::exception {
private:
    std::string message;

public:
    computationError(const std::string& msg) : message(msg) {}
    
    const char* what() const noexcept override {
        return message.c_str();
    }
};

using invalidArgument = std::invalid_argument;

} // namespace OptiSik