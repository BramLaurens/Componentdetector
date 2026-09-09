import serial
import customtkinter
import time
import cv2
from PIL import Image, ImageTk
import threading

stream = None
stream_running = False

ser = serial.Serial(
    port="/dev/ttyUSB0",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1
)

def rotate(angle):
    ser.write(f"ROTATE {angle}\r\n".encode())

def led(state):
    if state:
        ser.write(b"LED ON\r\n")
    else:
        ser.write(b"LED OFF\r\n")

def open_stream():
    global stream

    if stream is not None:
        return
    
    #create connecting popup window
    popup = customtkinter.CTkToplevel(app)
    popup.title("Connecting")
    popup.geometry("300x100")
    popup.transient(app)

    text = customtkinter.CTkLabel(popup, text="Connecting to the camera...")
    text.pack(expand=True, padx=20, pady=20)

    # Start a new thread to open the stream
    threading.Thread(
        target=open_stream_thread,
        args=(popup,),
        daemon=True
    ).start()

def show_error_popup(message):
    error_popup = customtkinter.CTkToplevel(app)
    error_popup.title("Error")
    error_popup.geometry("400x150")
    error_popup.transient(app)

    text = customtkinter.CTkLabel(error_popup, text=message)
    text.pack(expand=True, padx=20, pady=20)

    ok_button = customtkinter.CTkButton(error_popup, text="OK", command=lambda: error_popup.destroy())
    ok_button.pack(pady=10)

def open_stream_thread(popup):
    global stream

    url = "rtsp://192.168.4.1/mjpeg/1"

    try:
        new_stream = cv2.VideoCapture(
            url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000
            ]
        )

        if new_stream.isOpened():
            stream = new_stream

            app.after(0, update_frame)
            app.after(0, popup.destroy)

        else:
            new_stream.release()

            app.after(0, popup.destroy)
            app.after(
                0,
                lambda: show_error_popup(
                    "Failed to open the stream.\n"
                    "Please check the camera connection."
                )
            )

    except Exception as e:
        app.after(0, popup.destroy)
        app.after(
            0,
            lambda: show_error_popup(f"Camera error:\n{e}")
        )

def update_frame():
    if stream is not None:

        ret, frame = stream.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(frame)

            imgtk = customtkinter.CTkImage(
                light_image=img,
                dark_image=img,
                size=(640, 480)
            )

            video.configure(image=imgtk)
            video.imgtk = imgtk

        app.after(10, update_frame)

app = customtkinter.CTk()
app.title ("Beeldherkenning Capture Tool")
app.geometry("2000x300")

button = customtkinter.CTkButton(app, text="Rotate 90°", width=200, height=100, command=lambda: rotate(90))
button.grid(row=0, column=0, padx=10, pady=10)

button2 = customtkinter.CTkButton(app, text="LED ON", width=200, height=100, command=lambda: led(True))
button2.grid(row=0, column=1, padx=10, pady=10)

button3 = customtkinter.CTkButton(app, text="LED OFF", width=200, height=100, command=lambda: led(False))
button3.grid(row=0, column=2, padx=10, pady=10)

stream_button = customtkinter.CTkButton( app, text="Open stream", width=200, height=100, command=open_stream )
stream_button.grid(row=0, column=3, padx=10, pady=10)

video = customtkinter.CTkLabel(app, width=640, height=480)
video.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=1)
app.grid_columnconfigure(2, weight=1)
app.grid_columnconfigure(3, weight=1)

app.mainloop()