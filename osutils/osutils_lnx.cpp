/**
 * @file osutils_lnx.cpp
 * @author Chenglin.Bi <Chenglin.Bi@iluvatar.ai>
 * @brief
 * @version 0.1
 * @date 2018-11-17
 *
 * @copyright Copyright (c) 2018
 *
 */
#include <dlfcn.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include "osutils.h"

namespace OS
{
/**
 * @brief To RESERVE memory in Linux, use mmap with a private, anonymous, non-accessible mapping.
 *
 * @param size
 * @return void*
 */
void* ReserveVa(size_t size)
{
    return mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
}

void* ReserveSharedRwVa(size_t size, bool mallocPA)
{
    int flags = MAP_ANONYMOUS | MAP_SHARED;
    if (mallocPA)
        flags |= MAP_POPULATE;
    else
        flags |= MAP_NORESERVE;
    return mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
}

void* ReserveFdSharedRwVa(size_t size, int fd)
{
    int flags = MAP_SHARED | MAP_NORESERVE;
    return mmap(NULL, size, PROT_READ | PROT_WRITE, flags, fd, 0);
}

int MapVa(void* p, size_t size, bool read, bool write)
{
    return mprotect(p, size, (read ? PROT_READ : 0) | (write ? PROT_WRITE : 0));
}

extern int PinVa(void* p, size_t size)
{
    return mlock(p, size);
}
extern int UnPinVa(void* p, size_t size)
{
    return munlock(p, size);
}
extern int FreeVa(void* p, size_t size)
{
    return munmap(p, size);
}

void* LoadDynamicLib(const char* libname)
{
    return dlopen(libname, RTLD_NOW);
}

void* GetSymbol(void* h, const char* symbolName)
{
    return dlsym(h, symbolName);
}

void UnloadDynamicLib(void* h)
{
    dlclose(h);
}

const unsigned int GetTlsKeysMax()
{
    return sysconf(_SC_THREAD_KEYS_MAX);
}

void TlsInit(unsigned int* pKey, void (*destroyFunc)(void*))
{
    pthread_key_create(pKey, destroyFunc);
}

void TlsDestroy(unsigned int key)
{
    pthread_key_delete(key);
}

void* GetTlsData(unsigned int key)
{
    return pthread_getspecific(key);
}

void SetTlsData(unsigned int key, void* p)
{
    pthread_setspecific(key, p);
}

unsigned int GetThreadId()
{
    return pthread_self();
}

int GetNumCores()
{
    return sysconf(_SC_NPROCESSORS_ONLN);
}

void GetCurrentProcessExecPath(char** path, unsigned int size)
{
    ssize_t bytes = readlink("/proc/self/exe", *path, size);
    if (bytes == -1)
        (*path)[0] = '\x00';
}

bool IsFileExists(const char* filename)
{
    return (access(filename, F_OK) != -1);
}

}  // namespace OS