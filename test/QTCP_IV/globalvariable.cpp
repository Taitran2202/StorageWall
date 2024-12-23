#include "globalvariable.h"

QLogger::LogLevel g_log_debug_level = QLogger::InfoLevel;
bool g_is_write_image_debug = false;
QString g_app_name = "QTCP";
std::atomic_bool g_force_exit_app(false);

