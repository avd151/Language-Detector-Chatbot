import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from ocr import ocr
from gui_vars import *
from cb_chat import *

'''
|-------------------------------------GUI Structure-------------------------------------|
|                                                                                       |
|                                          win                                          |
|                                        QWidget()                                      |
|                                           |                                           |
|          _________________________________|________________________________           |
|          |           |           |               |            |            |          |
|      chat_area  display_img  select_img     export_chat   input_txt     send_txt      |
|    QScrollArea()  QLabel()  QPushButton()  QPushButton() QTextEdit()  QPushButton()   |
|          |                                                                            |
|   chat_area_inside                                                                    |
|       QWidget()                                                                       |
|          |                                                                            |
|    ______|______                                                                      |
|    |           |                                                                      |
|   left       right                                                                    |
|  QLabel()   QLabel()                                                                  |
|                                                                                       |
|---------------------------------------------------------------------------------------|
'''

#main window
class Window(object):
    def __init__(self) -> None:
        super().__init__()

        #build app
        self.initApp()
        self.initWindow()
        self.initShortcuts()

        #create gui
        self.createChatArea()
        self.createDisplayImage()
        self.createSelectImage()
        self.createExportChat()
        self.createInputText()
        self.createSendButton()
        self.createSpacer()

        #execute app
        self.execApp()
        
    #initialize app
    def initApp(self) -> None:
        '''
        Initialize application

        Instantiation of QApplication, QWidget, QGridLayout classes of the 
        application. Initialization of CHAT string, which stores the 
        conversation to be exported.
        '''
        self.app = QApplication(sys.argv)
        self.win = QWidget()
        self.grid = QGridLayout()
        self.CHAT = ''

    #initialize window
    def initWindow(self) -> None:
        '''
        Initialize window.

        Set grid layout, location on screen, size, title, name for qss, 
        icon and stylesheet to the window.
        '''
        self.win.setLayout(self.grid)
        self.win.setGeometry(WINDOW_POSX,   WINDOW_POSY, 
                             WINDOW_WIDTH,  WINDOW_HEIGHT)
        self.win.setWindowTitle(WINDOW_TITLE)
        self.win.setObjectName(WINDOW_NAME)
        self.win.setWindowIcon(QIcon(WINDOW_ICON))
        self.app.setStyleSheet(open(STYLE_SHEET, 'r').read())

    #initialize shortcuts
    def initShortcuts(self) -> None:
        '''
        Initialize keyboard shortcuts.

        Instantiation QShortcut and bind them to the window.
        ⦿ Keyboard shortcuts are as follows:
        1. Ctrl + I     =   Select Image
        2. Ctrl + E     =   Export Chat
        3. Ctrl + Enter =   Send Message
        4. Ctrl + Q     =   Quit the application.
        '''
        QShortcut(QKeySequence("Ctrl+I"), self.win).activated.connect(
            self.selectImage
        )
        QShortcut(QKeySequence("Ctrl+E"), self.win).activated.connect(
            self.exportChat
        )
        QShortcut(QKeySequence("Ctrl+Return"), self.win).activated.connect(
            self.sendText
        )
        QShortcut(QKeySequence("Ctrl+Q"), self.win).activated.connect(
            self.win.close
        )

    #create chat area
    def createChatArea(self) -> None:
        '''
        Create scrollable chat area.

        Instantiation of QScrollArea for scrollable chat area and QWidget 
        inside it. Assign Vertical Box layout to the QWidget to align chat 
        vertically.
        '''
        self.chat_area = QScrollArea()
        self.chat_area.setObjectName(CHAT_AREA_NAME)
        self.chat_area.setWidgetResizable(True)
        self.chat_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.chat_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.grid.addWidget(self.chat_area, 0, 0, 5, 6)
        self.chat_area_inside = QWidget()
        self.chat_area_inside.setObjectName(CHAT_AREA_INSIDE_NAME)
        self.chat_area.setWidget(self.chat_area_inside)
        self.chat_box = QVBoxLayout()
        self.chat_area_inside.setLayout(self.chat_box)

    #create display image label
    def createDisplayImage(self) -> None:
        '''
        Create label to display image selected.

        Image selected with select_img button will be visible in QLabel for 
        users reference.
        '''
        self.display_img = QLabel()
        self.display_img.setObjectName(DISPLAY_IMAGE_NAME)
        self.display_img.setMinimumHeight(150)
        self.display_img.setMinimumWidth(150)
        self.display_img.setAlignment(Qt.AlignCenter)
        self.grid.addWidget(self.display_img, 0, 6, 2, 2)

    #create select image button
    def createSelectImage(self) -> None:
        '''
        Create button for selecting image.

        To perform OCR, image should be selected with help of this button.
        '''
        self.select_img = QPushButton()
        self.select_img.setObjectName(SELECT_IMAGE_NAME)
        self.select_img.setFixedSize(60, 60)
        self.select_img.setCursor(QCursor(Qt.PointingHandCursor))
        self.select_img.setIcon(QIcon(SELECT_IMAGE_ICON))
        self.select_img.setIconSize(QSize(30, 30))
        self.select_img.clicked.connect(self.selectImage)
        self.grid.addWidget(self.select_img, 0, 8, 1, 1)

    #select image
    def selectImage(self) -> None:
        '''
        Select image for OCR.

        Bound to select_img button. This will open a dialogbox to choose a 
        jpeg or png image and set it to display_img label with aspect ratio 
        preserved.

        Shortcut : Ctrl + I.
        '''
        filename = QFileDialog.getOpenFileName(self.win, 'Open an image', '', 
                                               'Image Files (*.jpeg *jpg *.png)')
        if(filename[0] != ''):
            pixmap = QPixmap(filename[0])
            pixmap = pixmap.scaled(self.display_img.width(), 
                                   self.display_img.height(), 
                                   Qt.KeepAspectRatio)
            self.display_img.setPixmap(pixmap)
            self.input_txt.setText('detect: ' + ocr(filename[0]))

    #create export chat button
    def createExportChat(self) -> None:
        '''
        Create button for exporting the chat.

        If user wants to save the chat for future reference, he can by clicking 
        this button.
        '''
        self.export_chat = QPushButton()
        self.export_chat.setObjectName(EXPORT_CHAT_NAME)
        self.export_chat.setFixedSize(60, 60)
        self.export_chat.setCursor(QCursor(Qt.PointingHandCursor))
        self.export_chat.setIcon(QIcon(EXPORT_CHAT_ICON))
        self.export_chat.setIconSize(QSize(30, 30))
        self.export_chat.clicked.connect(self.exportChat)
        self.grid.addWidget(self.export_chat, 1, 8, 1, 1)

    #export chat to a file
    def exportChat(self) -> None:
        '''
        Export chat to a file.

        Bound to export_chat button. This will open a dialogbox to save the 
        chat as a file and write CHAT string to it.

        Shortcut : Ctrl + E.
        '''
        filename = QFileDialog.getSaveFileName(self.win, 'Export Chat')
        if(filename[0] != ''):
            with open(filename[0], 'w') as file:
                file.write(self.CHAT)

    #create input text
    def createInputText(self) -> None:
        '''
        Create area to type the message.

        User can type a multilined message in this scrollable field to send.
        '''
        self.input_txt = QTextEdit()
        self.input_txt.setObjectName(INPUT_TEXT_NAME)
        self.grid.addWidget(self.input_txt, 2, 6, 2, 3)

    #create send button
    def createSendButton(self) -> None:
        '''
        Create button to send the message.

        On click, the text is sent to the chatbot.
        '''
        self.send_txt = QPushButton()
        self.send_txt.setObjectName(SEND_TEXT_NAME)
        self.send_txt.setCursor(QCursor(Qt.PointingHandCursor))
        self.send_txt.setIcon(QIcon(SEND_TEXT_ICON))
        self.send_txt.setIconSize(QSize(30, 30))
        self.send_txt.clicked.connect(self.sendText)
        self.grid.addWidget(self.send_txt, 4, 6, 1, 3)

    #send the text from input text
    def sendText(self) -> None:
        '''
        Send message to chatbot.

        Bound to send_text button. This will fetch text from input_text field 
        and send to the chatbot if it is not empty. The chat bubble will be 
        shown in the chat area. The response of the chatbot is also displayed 
        as a bubble. This is where the main chat function lies.

        Shortcut : Ctrl + Enter.
        '''
        msg = self.input_txt.toPlainText()
        if(msg.replace('\n', '') != ''):
            self.input_txt.clear()
            self.createRightBubble(msg)
            res = generate_response(msg)
            self.CHAT += f'You : {msg}\n'
            self.createLeftBubble(res)
            self.CHAT += f'{CHATBOT} : {res}\n'
            self.display_img.clear()

    #create left chat bubble
    def createLeftBubble(self, text:str) -> None:
        '''
        Create left chat bubble.

        For displaying response from the chatbot in chat area.

        Parameter : str:text
        '''
        self.chat_box.removeItem(self.spacer)
        self.left = QLabel(text)
        self.left.setObjectName(LEFT_BUBBLE_NAME)
        self.left.setWordWrap(True)
        self.left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.chat_box.addWidget(self.left)
        self.chat_box.addItem(self.spacer)
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )  

    #create right chat bubble
    def createRightBubble(self, text:str) -> None:
        '''
        Create right chat bubble.

        For displaying users message in chat area.
        
        Parameter : str:text
        '''
        self.chat_box.removeItem(self.spacer)
        self.right = QLabel(text)
        self.right.setObjectName(RIGHT_BUBBLE_NAME)
        self.right.setAlignment(Qt.AlignRight)
        self.right.setWordWrap(True)
        self.chat_box.addWidget(self.right)
        self.chat_box.addItem(self.spacer)
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )    

    #create vertical spacer for chat area
    def createSpacer(self) -> None:
        '''
        Create vertical spacer for chat area.
        
        To make the bubble stick to the top of the available area(like other 
        popular chatting applications), we add a vertical spacer which acts 
        like a pushing spring between the last(bottom-most) message and the 
        bottom edge of chat area.
        '''
        self.spacer = QSpacerItem(0, 0, QSizePolicy.Minimum, 
                                  QSizePolicy.Expanding)
        self.chat_box.addItem(self.spacer)    

    #execute app
    def execApp(self) -> None:
        '''
        Execution of application.

        To show the window and start the execution loop until user closes the 
        application.
        '''
        self.win.show()
        sys.exit(self.app.exec_())