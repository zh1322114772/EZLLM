#pragma once
#include <string>

#ifdef _WIN32
    #define NOMINMAX
    #include <windows.h>
#endif


inline void InitializeUtf8Console()
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

