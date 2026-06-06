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

        // Video Streams + Status Panel
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

            // Status Panel
            Rectangle {
                Layout.preferredWidth: 200
                Layout.fillHeight: true
                color: "#1a1a2e"
                radius: 8

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 20

                    Text {
                        text: "System"
                        color: "#aaaaaa"
                        font.pixelSize: 13
                        font.bold: true
                        Layout.alignment: Qt.AlignHCenter
                    }

                    // CPU Load
                    ColumnLayout {
                        spacing: 4
                        Layout.fillWidth: true
                        Text { text: "CPU Load"; color: "#888888"; font.pixelSize: 11 }
                        Text {
                            text: systemStats.cpuLoad.toFixed(1) + " %"
                            color: "#00e5ff"
                            font.pixelSize: 22
                            font.bold: true
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            height: 6
                            radius: 3
                            color: "#333355"
                            Rectangle {
                                width: parent.width * Math.min(systemStats.cpuLoad / 100.0, 1.0)
                                height: parent.height
                                radius: parent.radius
                                color: systemStats.cpuLoad > 80 ? "#ff4444" : "#00e5ff"
                                Behavior on width { SmoothedAnimation { duration: 400 } }
                            }
                        }
                    }

                    // Memory Usage
                    ColumnLayout {
                        spacing: 4
                        Layout.fillWidth: true
                        Text { text: "Memory"; color: "#888888"; font.pixelSize: 11 }
                        Text {
                            text: systemStats.memUsedMb + " / " + systemStats.memTotalMb + " MB"
                            color: "#69ff47"
                            font.pixelSize: 16
                            font.bold: true
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            height: 6
                            radius: 3
                            color: "#333355"
                            Rectangle {
                                width: systemStats.memTotalMb > 0
                                       ? parent.width * (systemStats.memUsedMb / systemStats.memTotalMb)
                                       : 0
                                height: parent.height
                                radius: parent.radius
                                color: "#69ff47"
                                Behavior on width { SmoothedAnimation { duration: 400 } }
                            }
                        }
                    }

                    // People seen counter
                    ColumnLayout {
                        spacing: 4
                        Layout.fillWidth: true
                        Text { text: "People Seen"; color: "#888888"; font.pixelSize: 11 }
                        Text {
                            text: peopleCounter.peopleSeen
                            color: "#ffd740"
                            font.pixelSize: 40
                            font.bold: true
                            Layout.alignment: Qt.AlignHCenter
                        }
                    }

                    Item { Layout.fillHeight: true }
                }
            }
        }
    }
}
