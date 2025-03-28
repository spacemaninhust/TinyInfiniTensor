#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        size_t offset = 0;
        if (freeBlocks.empty())
        {
            offset = this->used;
            this->used += size;
            if (this->used > this->peak)
            {
                this->peak = this->used;
            }
        } else {
            for (auto it = freeBlocks.begin(); it != freeBlocks.end(); it++)
            {
                offset = it->first;
                freeBlocks.erase(it);
                break;
            }
        }
        return offset;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        freeBlocks.insert(std::make_pair(addr, size));
        auto it = freeBlocks.find(addr);
        if(it != freeBlocks.end()) {
            auto prev = it;
            if(prev != freeBlocks.begin()) {
                --prev;
                if (prev->first + prev->second == it->first)
                {
                    size_t newSize = prev->second + it->second;
                    freeBlocks.erase(prev);
                    freeBlocks.erase(it);
                    freeBlocks.insert(std::make_pair(addr - prev->second, newSize));
                    return;
                }
            }
            auto next = std::next(it);
            if (next != freeBlocks.end() && it->first + it->second == next->first)
            {
                size_t newSize = it->second + next->second;
                freeBlocks.erase(it);
                freeBlocks.erase(next);
                freeBlocks.insert(std::make_pair(addr, newSize));
            }
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
