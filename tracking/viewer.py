from os import listdir
from os.path import *
from application import Track_app
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog, QWidget, QBoxLayout, QVBoxLayout, QGraphicsView, QGraphicsScene


img_data_path = "/Users/hyungjungu/Documents/Project/CV/project/img_data"

def comp(img_file):
    return int(img_file[:-4])

class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self._pixmapHandle = None
        self.setWindowTitle("Image Viewer")
        self.resize(800,600)
        self.img_files = [f for f in listdir(img_data_path) if isfile(join(img_data_path, f))]
        self.img_files.remove('.DS_Store')
        self.img_files = sorted(self.img_files, key=comp)
        print(f"imgae files: {self.img_files}")
        self.img_files = [join(img_data_path, f) for f in self.img_files]
        self.img_files = [QPixmap.fromImage(QImage(f)) for f in self.img_files]
        self.img_files_len = len(self.img_files)
        self.setImage(self.img_files[0])

        self.track_app = Track_app(self.img_files_len)
        self.track_app.face_track_thread()
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.updateImage)
        self.timer.start()

    def hasImage(self):
        """ Returns whether or not the scene contains image pixmap"""
        return self._pixmapHandle is not None

    def setImage(self, image):
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        if self.hasImage():
            self._pixmapHandle.setPixmap(pixmap)
        else:
            self._pixmapHandle = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))


    def updateImage(self):
        image_number = self.track_app.get_face_section()
        print(f"image_number from update_image : {image_number}")
        # image = QImage(self.img_files[image_number])
        # print(f"imagefile is : {image}")
        # self.imageLabel.setPixmap(QPixmap.fromImage(image))
        # self.imageLabel.setPixmap(QPixmap(QImage(self.img_files[image_number])))
        self.setImage(self.img_files[image_number])
        
    

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    image_viewer = ImageViewer()
    image_viewer.show()
    sys.exit(app.exec_())