import ctypes

#chatbot name
CHATBOT = 'Duolingo'

#screen
user32 = ctypes.windll.user32
SCREEN_WIDTH, SCREEN_HEIGHT = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

#window
WINDOW_NAME = 'win'
WINDOW_WIDTH, WINDOW_HEIGHT = 900, 500
WINDOW_POSX, WINDOW_POSY = int((SCREEN_WIDTH - WINDOW_WIDTH) / 2), int((SCREEN_HEIGHT - WINDOW_HEIGHT) / 2)
WINDOW_TITLE = "Language Detector Chatbot"
STYLE_SHEET = 'style.qss'
WINDOW_ICON = './img/icon.ico'

#chat area
CHAT_AREA_NAME = 'chatarea'

#chat area inside
CHAT_AREA_INSIDE_NAME = 'chat_area_inside'

#display image
DISPLAY_IMAGE_NAME = 'display_image'

#select image
SELECT_IMAGE_NAME = 'select_image'
SELECT_IMAGE_ICON = './img/selectImage.png'

#export chat
EXPORT_CHAT_NAME = 'export_chat'
EXPORT_CHAT_ICON = './img/exportChat.png'

#input text
INPUT_TEXT_NAME = 'input_text'

#send text
SEND_TEXT_NAME = 'send_text'
SEND_TEXT_ICON = './img/sendText.png'

#left bubble
LEFT_BUBBLE_NAME = 'left_bubble'

#right bubble
RIGHT_BUBBLE_NAME = 'right_bubble'