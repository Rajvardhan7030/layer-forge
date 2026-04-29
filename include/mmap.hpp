#pragma once

#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>

namespace layer_forge {

class MemoryMappedFile {
public:
    MemoryMappedFile(const std::string& path) {
        fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Failed to open file for mmap: " + path);
        }

        struct stat st;
        if (fstat(fd, &st) == -1) {
            close(fd);
            throw std::runtime_error("Failed to get file size: " + path);
        }
        size = st.st_size;

        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to mmap file: " + path);
        }
    }

    ~MemoryMappedFile() {
        if (data != MAP_FAILED) {
            munmap(data, size);
        }
        if (fd != -1) {
            close(fd);
        }
    }

    void* get_ptr(size_t offset) const {
        if (offset >= size) return nullptr;
        return static_cast<char*>(data) + offset;
    }

    size_t get_size() const { return size; }

private:
    int fd = -1;
    size_t size = 0;
    void* data = MAP_FAILED;
};

} // namespace layer_forge
