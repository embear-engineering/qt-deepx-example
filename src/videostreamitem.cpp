#include "videostreamitem.h"
#include "debug.h"
#include <QPainter>

VideoStreamItem::VideoStreamItem(QQuickItem *parent)
    : QQuickPaintedItem(parent)
{
    setRenderTarget(QQuickPaintedItem::FramebufferObject);
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
    DLOG("VideoStreamItem::updateImage " << image.width() << "x" << image.height()
         << " format=" << image.format());
    {
        QMutexLocker locker(&m_mutex);
        m_image = image;
    }
    update(); // Request repaint
}
