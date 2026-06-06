#include "systemstats.h"
#include "debug.h"
#include <QFile>
#include <QTextStream>

SystemStats::SystemStats(QObject* parent)
    : QObject(parent)
{
    connect(&m_timer, &QTimer::timeout, this, &SystemStats::update);
    m_timer.start(1000);
    update(); // initial sample
}

void SystemStats::update()
{
    // --- CPU load from /proc/stat ---
    {
        QFile f("/proc/stat");
        if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QTextStream in(&f);
            QString line = in.readLine(); // first line: "cpu  user nice system idle iowait irq softirq ..."
            QStringList parts = line.split(' ', Qt::SkipEmptyParts);
            // parts[0] == "cpu", parts[1..] are the fields
            if (parts.size() >= 5) {
                long long user    = parts[1].toLongLong();
                long long nice    = parts[2].toLongLong();
                long long system  = parts[3].toLongLong();
                long long idle    = parts[4].toLongLong();
                long long iowait  = parts.size() > 5 ? parts[5].toLongLong() : 0;
                long long irq     = parts.size() > 6 ? parts[6].toLongLong() : 0;
                long long softirq = parts.size() > 7 ? parts[7].toLongLong() : 0;

                long long idleTime  = idle + iowait;
                long long totalTime = user + nice + system + idle + iowait + irq + softirq;

                long long deltaTotal = totalTime - m_prevTotal;
                long long deltaIdle  = idleTime  - m_prevIdle;

                if (deltaTotal > 0) {
                    double load = (1.0 - static_cast<double>(deltaIdle) / deltaTotal) * 100.0;
                    m_cpuLoad = qBound(0.0, load, 100.0);
                    emit cpuLoadChanged();
                }

                m_prevTotal = totalTime;
                m_prevIdle  = idleTime;
            }
        } else {
            DLOG("SystemStats: cannot open /proc/stat");
        }
    }

    // --- Memory from /proc/meminfo ---
    {
        QFile f("/proc/meminfo");
        if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QTextStream in(&f);
            long long memTotal = 0, memAvailable = 0;
            while (!in.atEnd()) {
                QString line = in.readLine();
                if (line.startsWith("MemTotal:"))
                    memTotal = line.split(' ', Qt::SkipEmptyParts)[1].toLongLong();
                else if (line.startsWith("MemAvailable:"))
                    memAvailable = line.split(' ', Qt::SkipEmptyParts)[1].toLongLong();
                if (memTotal && memAvailable) break;
            }
            if (memTotal > 0) {
                int newUsed  = static_cast<int>((memTotal - memAvailable) / 1024);
                int newTotal = static_cast<int>(memTotal / 1024);
                if (newUsed != m_memUsedMb || newTotal != m_memTotalMb) {
                    m_memUsedMb  = newUsed;
                    m_memTotalMb = newTotal;
                    emit memChanged();
                }
            }
        } else {
            DLOG("SystemStats: cannot open /proc/meminfo");
        }
    }
}
