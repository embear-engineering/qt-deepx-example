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

    RowLayout {
        anchors.fill: parent
        spacing: 10
        
        // Dynamic creation of video items based on available streams would be nice,
        // but for now we can define two slots and hide if not used.
        
        VideoStreamItem {
            id: stream0
            objectName: "stream0"
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredWidth: parent.width / 2
            visible: true // Controlled by C++
        }

        VideoStreamItem {
            id: stream1
            objectName: "stream1"
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredWidth: parent.width / 2
            visible: false // Controlled by C++
        }
    }
}
