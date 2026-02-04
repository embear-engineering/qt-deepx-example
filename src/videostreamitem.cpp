#include "videostreamitem.h"
#include <QPainter>

VideoStreamItem::VideoStreamItem(QQuickItem *parent)
    : QQuickPaintedItem(parent)
{
    // Optimizations?
    // setRenderTarget(QQuickPaintedItem::FramebufferObject); // Maybe faster?
}

void VideoStreamItem::paint(QPainter *painter)
{
    QMutexLocker locker(&m_mutex);
    if (!m_image.isNull()) {
        QRectF targetRect(0, 0, width(), height());
        painter->drawImage(targetRect, m_image);
    } else {
        painter->fillRect(0, 0, width(), height(), Qt::black);
        painter->setPen(Qt::white);
        painter->drawText(boundingRect(), Qt::AlignCenter, "No Signal");
    }
}

void VideoStreamItem::updateImage(QImage image)
{
    {
        QMutexLocker locker(&m_mutex);
        m_image = image;
    }
    update(); // Request repaint
}
