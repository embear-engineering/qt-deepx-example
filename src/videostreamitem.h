#ifndef VIDEOSTREAMITEM_H
#define VIDEOSTREAMITEM_H

#include <QQuickPaintedItem>
#include <QImage>
#include <QMutex>

class VideoStreamItem : public QQuickPaintedItem
{
    Q_OBJECT
public:
    explicit VideoStreamItem(QQuickItem *parent = nullptr);
    void paint(QPainter *painter) override;

public slots:
    void updateImage(QImage image);

private:
    QImage m_image;
    QMutex m_mutex;
};

#endif // VIDEOSTREAMITEM_H
