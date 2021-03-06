# Language-Detector-Chatbot

It is a Chatbot which provides Language detection features by identifying language of given text and images  

## Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install following python libraries.

```bash
pip install numpy
pip install pandas
pip install seaborn
pip install matplotlib
pip install scikit-learn
pip install nltk
nltk.download('punkt')
pip install PyQt5
pip install Pillow
pip install Pytesseract
```
1. Download pytorch from https://pytorch.org/ using customizations as required, or using the below command.
```bash
pip3 install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```
2. Download and install tesseract 32 bit or 64 bit from https://github.com/UB-Mannheim/tesseract/wiki and add path of tesseract.exe to the 4<sup>th</sup> line of file ocr.py

## Usage
Run chat.py file by opening terminal or command prompt, and typing the following command
```bash
python chat.py
```
Run main.py file in terminal to chat on GUI
```bash
python main.py
```

## Output
![chatbot demo](https://github.com/avd151/Language-Detector-Chatbot/blob/main/ss/ss1.png?raw=true)
