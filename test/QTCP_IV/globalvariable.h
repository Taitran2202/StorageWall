#ifndef GLOBALVARIABLE_H
#define GLOBALVARIABLE_H

#include "logger/qlogger.h"
#include <unistd.h>

extern QLogger::LogLevel g_log_debug_level;
extern bool g_is_write_image_debug;
extern QString g_app_name;
extern std::atomic_bool g_force_exit_app;


#endif // GLOBALVARIABLE_H
