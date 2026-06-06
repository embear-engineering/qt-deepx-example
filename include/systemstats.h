#pragma once

#include <QObject>
#include <QTimer>

class SystemStats : public QObject {
    Q_OBJECT
    Q_PROPERTY(double cpuLoad    READ cpuLoad    NOTIFY cpuLoadChanged)
    Q_PROPERTY(int    memUsedMb  READ memUsedMb  NOTIFY memChanged)
    Q_PROPERTY(int    memTotalMb READ memTotalMb NOTIFY memChanged)
public:
    explicit SystemStats(QObject* parent = nullptr);

    double cpuLoad()    const { return m_cpuLoad; }
    int    memUsedMb()  const { return m_memUsedMb; }
    int    memTotalMb() const { return m_memTotalMb; }

signals:
    void cpuLoadChanged();
    void memChanged();

private slots:
    void update();

private:
    QTimer    m_timer;
    double    m_cpuLoad    = 0.0;
    int       m_memUsedMb  = 0;
    int       m_memTotalMb = 0;

    long long m_prevIdle  = 0;
    long long m_prevTotal = 0;
};
