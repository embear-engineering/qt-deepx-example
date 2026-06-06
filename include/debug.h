#pragma once

#include <iostream>

// Global verbose flag — set to true when -v is passed on the command line.
// Defined in main.cpp; declared extern here for use in all translation units.
extern bool g_verbose;

// Stream-style debug logging macro. Usage: DLOG("value=" << x)
#define DLOG(msg) \
    do { if (g_verbose) { std::cerr << "[DBG] " << msg << "\n"; } } while (0)
