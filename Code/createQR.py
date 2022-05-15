from pyqrcode import QRCode

url = 'https://tulimid1.github.io/what-is-covering-me/'

myQR = QRCode(url)

myQR.png('QRCode.png', scale=8)
