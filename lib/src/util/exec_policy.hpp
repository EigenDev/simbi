#ifndef EXEC_POLICY_HPP
#define EXEC_POLICY_HPP

#include "build_options.hpp"
#include <vector>
#include <iostream>
struct ExecutionException : public std::exception {
   const char * what () const throw () {
      return "Invalid constructor args";
   }
};

namespace simbi {

    enum class AutoConfig {
        GRID_SIZE  = 1 << 20,
        BLOCK_SIZE = 32
    };


    template<typename T = luint, typename U = luint>
    struct ExecutionPolicy
    {
        T      nzones;
        dim3   gridSize;
        dim3   blockSize;
        size_t sharedMemBytes;
        simbiStream_t stream;

        ExecutionPolicy(const T nzones) 
        :  
        nzones(nzones),  
        sharedMemBytes(0),
        stream(0) {
            const T nBlocks = compute_blocks(nzones, BLOCK_SIZE);
            this->gridSize  = dim3(nBlocks);
            this->blockSize = dim3(BLOCK_SIZE);
        }

        ExecutionPolicy(const T nzones, const U blockSize)
        : stream(0) ,
        blockSize(dim3(blockSize)),
        sharedMemBytes(sharedMemBytes)  
        {
            const T nBlocks = compute_blocks(nzones, blockSize);
            this->gridSize  = dim3(nBlocks);
        }
        
        ExecutionPolicy(const T nzones, const U blockSize, const size_t sharedMemBytes)
        : stream(0) ,
        blockSize(dim3(blockSize)),
        sharedMemBytes(sharedMemBytes)  
        {
            const T nBlocks = compute_blocks(nzones, blockSize);
            this->gridSize  = dim3(nBlocks);
        }

        ExecutionPolicy(const T nzones, const U blockSize, const size_t sharedMemBytes, const simbiStream_t stream)
        :
        blockSize(dim3(blockSize)),
        sharedMemBytes(sharedMemBytes),
        stream(stream)
        {
            const T nBlocks = compute_blocks(nzones, blockSize);
            this->gridSize  = dim3(nBlocks);
        }

        ExecutionPolicy(const std::vector<T> glist, const std::vector<U> blist)
        : stream(0) ,
        sharedMemBytes(0)
        {
            try {
                if (glist.size() != blist.size())
                {
                    throw ExecutionException();
                }
            } catch (ExecutionException& e)
            {
                std::cout << "Bad construction of execution policy" << std::endl;
                std::cout << e.what() << std::endl;
            }
            if (glist.size() == 1)
            {
                this->gridSize  = dim3((glist[0] + blist[0] - 1) / blist[0]);
                this->blockSize = dim3(blist[0]); 
            } else if (glist.size() == 2)
            {
                luint nxBlocks    = (glist[0] + blist[0] - 1) / blist[0];
                luint nyBlocks    = (glist[1] + blist[1] - 1) / blist[1];
                this->gridSize  = dim3(nxBlocks, nyBlocks);
                this->blockSize = dim3(blist[0], blist[1]); 
            } else if (glist.size() == 3)
            {
                luint nxBlocks    = (glist[0] + blist[0] - 1) / blist[0];
                luint nyBlocks    = (glist[1] + blist[1] - 1) / blist[1];
                luint nzBlocks    = (glist[2] + blist[2] - 1) / blist[2];
                this->gridSize  = dim3(nxBlocks, nyBlocks, nzBlocks);
                this->blockSize = dim3(blist[0], blist[1], blist[2]); 
            }
        }

        ExecutionPolicy(const std::vector<T> glist, const std::vector<U> blist, const size_t sharedMemBytes)
        : stream(0) ,
        sharedMemBytes(sharedMemBytes)
        {
            try {
                if (glist.size() != blist.size())
                {
                    throw ExecutionException();
                }
            } catch (ExecutionException& e)
            {
                std::cout << "Bad construction of execution policy" << std::endl;
                std::cout << e.what() << std::endl;
            }
            
            if (glist.size() == 1)
            {
                this->gridSize  = dim3((glist[0] + blist[0] - 1) / blist[0]);
                this->blockSize = dim3(blist[0]); 
            } else if (glist.size() == 2)
            {
                luint nxBlocks    = (glist[0] + blist[0] - 1) / blist[0];
                luint nyBlocks    = (glist[1] + blist[1] - 1) / blist[1];
                this->gridSize  = dim3(nxBlocks, nyBlocks);
                this->blockSize = dim3(blist[0], blist[1]); 
            } else if (glist.size() == 3)
            {
                luint nxBlocks    = (glist[0] + blist[0] - 1) / blist[0];
                luint nyBlocks    = (glist[1] + blist[1] - 1) / blist[1];
                luint nzBlocks    = (glist[2] + blist[2] - 1) / blist[2];
                this->gridSize  = dim3(nxBlocks, nyBlocks, nzBlocks);
                this->blockSize = dim3(blist[0], blist[1], blist[2]); 
            }
        }

        ExecutionPolicy(const std::vector<T> glist, const std::vector<U> blist, const size_t sharedMemBytes, const simbiStream_t stream)
        : stream(stream),
        sharedMemBytes(sharedMemBytes)
        {
            try {
                if (glist.size() != blist.size())
                {
                    throw ExecutionException();
                }
            } catch (ExecutionException& e)
            {
                std::cout << "Bad construction of execution policy" << std::endl;
                std::cout << e.what() << std::endl;
            }
            if (glist.size() == 1)
            {
                this->gridSize  = dim3((glist[0] + blist[0] - 1) / blist[0]);
                this->blockSize = dim3(blist[0]); 
            } else if (glist.size() == 2)
            {
                luint nxBlocks    = (glist[0] + blist[0] - 1) / blist[0];
                luint nyBlocks    = (glist[1] + blist[1] - 1) / blist[1];
                this->gridSize  = dim3(nxBlocks, nyBlocks);
                this->blockSize = dim3(blist[0], blist[1]); 
            } else if (glist.size() == 3)
            {
                luint nxBlocks    = (glist[0] + blist[0] - 1) / blist[0];
                luint nyBlocks    = (glist[1] + blist[1] - 1) / blist[1];
                luint nzBlocks    = (glist[2] + blist[2] - 1) / blist[2];
                this->gridSize  = dim3(nxBlocks, nyBlocks, nzBlocks);
                this->blockSize = dim3(blist[0], blist[1], blist[2]); 
            }
        }

            
        ~ExecutionPolicy() {}

        T compute_blocks(const T nzones, const luint nThreads) const
        {
            return (nzones + nThreads - 1) / nThreads;
        }
    };

} //namespace simbi

#endif