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
        QSize scaledSize = m_image.size();
        scaledSize.scale(size().toSize(), Qt::KeepAspectRatio);

        QRectF targetRect(
            (width() - scaledSize.width()) / 2.0,
            (height() - scaledSize.height()) / 2.0,
            scaledSize.width(),
            scaledSize.height()
        );

        // No fillRect here to keep background transparent/inherited
        painter->drawImage(targetRect, m_image);
    } else {
        painter->setPen(Qt::black);
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
