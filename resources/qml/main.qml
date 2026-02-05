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
    color: "#f5f5f5"

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // Header with Logos
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 100
            color: "white"

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 20
                anchors.rightMargin: 20

                Image {
                    source: "qrc:/img/tis-logo.jpg"
                    Layout.preferredHeight: 80
                    fillMode: Image.PreserveAspectFit
                    Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                }

                Item { Layout.fillWidth: true } // Spacer

                Image {
                    source: "qrc:/img/deepx-logo.jpg"
                    Layout.preferredHeight: 80
                    fillMode: Image.PreserveAspectFit
                    Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
                }

                Item { Layout.fillWidth: true } // Spacer

                Image {
                    source: "qrc:/img/embear-logo.jpg"
                    Layout.preferredHeight: 80
                    fillMode: Image.PreserveAspectFit
                    Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
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
