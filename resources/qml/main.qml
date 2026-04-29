import QtQuick 2.12
import QtQuick.Window 2.12
import QtQuick.Layouts 1.12
import QtQuick.Controls 2.12
import com.deepx.app 1.0

Window {
    visible: true
    width: 1280
    height: 720
    title: qsTr("DeepX Object Detection (QML)")
    color: "#222222"

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // Header with Logos
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 100
            color: "#000000"

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 20
                anchors.rightMargin: 20
                spacing: 0

                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Image {
                        anchors.left: parent.left
                        anchors.verticalCenter: parent.verticalCenter
                        source: "qrc:/img/avocado-os.jpg"
                        height: 80
                        fillMode: Image.PreserveAspectFit
                    }
                }

                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Image {
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter
                        source: "qrc:/img/deepx-logo.jpg"
                        height: 80
                        fillMode: Image.PreserveAspectFit
                    }
                }

                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Image {
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        source: "qrc:/img/grinn.jpg"
                        height: 80
                        fillMode: Image.PreserveAspectFit
                    }
                }
            }

            // Bottom border for header
            Rectangle {
                anchors.bottom: parent.bottom
                width: parent.width
                height: 1
                color: "#e0e0e0"
            }
        }

        // Video Streams
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 10
            Layout.margins: 10

            VideoStreamItem {
                id: stream0
                objectName: "stream0"
                Layout.fillWidth: true
                Layout.fillHeight: true
                visible: true
            }

            VideoStreamItem {
                id: stream1
                objectName: "stream1"
                Layout.fillWidth: true
                Layout.fillHeight: true
                visible: false
            }
        }
    }
}
