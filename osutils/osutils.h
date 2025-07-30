/**
 * @file osutils.h
 * @author Chenglin.Bi <Chenglin.Bi@iluvatar.ai>
 * @brief
 * @version 0.1
 * @date 2018-11-17
 *
 * @copyright Copyright (c) 2018
 *
 */
#ifndef __OSUTILS_H__
#define __OSUTILS_H__

#include <cstddef>

namespace OS
{
extern void* ReserveVa(size_t size);
extern void* ReserveSharedRwVa(size_t size, bool mallocPA);
extern void* ReserveFdSharedRwVa(size_t size, int fd);
extern int   MapVa(void* p, size_t size, bool read, bool write);
extern int   PinVa(void* p, size_t size);
extern int   UnPinVa(void* p, size_t size);
extern int   FreeVa(void* p, size_t size);

extern void* LoadDynamicLib(const char* libname);
extern void* GetSymbol(void* h, const char* symbolName);
extern void  UnloadDynamicLib(void* h);

extern unsigned int GetThreadId();

extern const unsigned int GetTlsKeysMax();
extern void*              GetTlsData(unsigned int key);
extern void               SetTlsData(unsigned int key, void* p);
extern void               TlsInit(unsigned int* pKey, void (*destroyFunc)(void*));
extern void               TlsDestroy(unsigned int key);

extern int  GetNumCores();
extern void GetCurrentProcessExecPath(char** path, unsigned int size);

extern bool IsFileExists(const char* filename);
};  // namespace OS

#endif  // __OSUTILS_H__