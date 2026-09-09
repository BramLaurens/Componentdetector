import serial

ser = serial.Serial(
    port="/dev/ttyUSB0",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1
)

def rotate(angle):
    ser.write(b"ROTATE {angle}\r\n".encode())

def led(state):
    if state:
        ser.write(b"LED ON\r\n")
    else:
        ser.write(b"LED OFF\r\n")

led(False)