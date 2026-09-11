import serial
import serial.tools.list_ports
import customtkinter
import time
import cv2
from PIL import Image, ImageTk
import threading
import os

#surpress ffmpeg log messages
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;error"

app = customtkinter.CTk()
app.title ("Beeldherkenning Capture Tool")
app.geometry("1920x1080")

stream = None
stream_running = False
latest_frame = None
frame_lock = threading.Lock()
stream_lock = threading.Lock()

current_componenttype = customtkinter.StringVar(value="Resistor")
selected_port = None

ser = None

def port_selection_callback(choice):
    global selected_port
    selected_port = choice
    print(f"Selected port: {choice}")

ports = [port.device for port in serial.tools.list_ports.comports()]

def open_serial_connection():
    global ser

    if ser is not None and ser.is_open:
        print("Serial connection is already open.")
        return
    ser = serial.Serial()
    ser.port=selected_port
    ser.baudrate=115200
    ser.bytesize=serial.EIGHTBITS
    ser.parity=serial.PARITY_NONE
    ser.stopbits=serial.STOPBITS_ONE
    ser.timeout=1

    ser.open()
    print(f"Opened serial connection on {ser.port}"
)

#rotate the circle
def rotate(angle):
    ser.write(f"ROTATE {angle}\r\n".encode())

#control the LED
def led(state):
    if state:
        ser.write(b"LED ON\r\n")
    else:
        ser.write(b"LED OFF\r\n")

#open stream called by button, creates a popup window and starts a new thread to open the stream
def open_stream():
    global stream

    if stream is not None:
        return
    
    #create connecting popup window
    popup = customtkinter.CTkToplevel(app)
    popup.title("Connecting")
    popup.geometry("1000x300")
    popup.transient(app)

    text = customtkinter.CTkLabel(popup, text="Connecting to the camera...")
    text.pack(expand=True, padx=20, pady=20)

    # Start a new thread to open the stream
    threading.Thread(
        target=open_stream_thread,
        args=(popup,),
        daemon=True
    ).start()

#error popup handler, shows a popup with the error message
def show_error_popup(message):
    error_popup = customtkinter.CTkToplevel(app)
    error_popup.title("Error")
    error_popup.geometry("400x150")
    error_popup.transient(app)

    text = customtkinter.CTkLabel(error_popup, text=message)
    text.pack(expand=True, padx=20, pady=20)

    ok_button = customtkinter.CTkButton(error_popup, text="OK", command=lambda: error_popup.destroy())
    ok_button.pack(pady=10)

#stream opening backend function, called in a separate thread to avoid blocking the main GUI thread
def open_stream_thread(popup):
    global stream, stream_running

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
            stream_running = True
            threading.Thread(target=stream_thread, daemon=True).start()

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

def stream_thread():
    global latest_frame, stream_running
    while stream_running:
        if stream is not None:
            with stream_lock:
                ret, frame = stream.read()
            if ret:
                with frame_lock:
                    latest_frame = frame


#frame update function, calls itself every 10ms to update the video frame in the GUI
def update_frame():
    global latest_frame

    # Start the stream thread if it's not already running
    with frame_lock:
        if latest_frame is not None:
            frame = latest_frame.copy()
        else:
            frame = None
            
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)

        imgtk = customtkinter.CTkImage(
            light_image=img,
            dark_image=img,
            size=(1280, 720)
        )

        video.configure(image=imgtk, text="")
        video.imgtk = imgtk

    app.after(30, update_frame)

def capture_frame():
    with frame_lock:
        if latest_frame is None:
            return False
        frame = latest_frame.copy()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    if cv2.imwrite(filename, frame):
        print(f"Captured frame saved as {filename}")
        return True
    print(f"Failed to save frame as {filename}")
    return False

def close_stream():
    global stream, stream_running
    stream_running = False
    with stream_lock:
        if stream is not None:
            stream.release()
            stream = None
            print("Stream closed.")

def combobox_callback(choice):
    global current_componenttype
    current_componenttype.set(choice)

    print(f"Selected component type: {current_componenttype.get()}")

def autocapture():
    threading.Thread(target=autocapture_thread, daemon=True).start()

def autocapture_thread():
    while True:
        capture_frame()
        print("Frame captured")
        time.sleep(2)  # Capture every 5 seconds

combobox = customtkinter.CTkComboBox(
    app, values=["Resistor", "Capacitor", "Inductor", "Diode", "Transistor"], 
    command=combobox_callback, 
    variable=current_componenttype)

serial_combobox = customtkinter.CTkComboBox(
    app, 
    values=ports,
    command=port_selection_callback)


button = customtkinter.CTkButton(app, text="Rotate 90°", width=200, height=100, command=lambda: rotate(90))
button.grid(row=0, column=0, padx=10, pady=10)

button2 = customtkinter.CTkButton(app, text="LED ON", width=200, height=100, command=lambda: led(True))
button2.grid(row=0, column=1, padx=10, pady=10)

button3 = customtkinter.CTkButton(app, text="LED OFF", width=200, height=100, command=lambda: led(False))
button3.grid(row=0, column=2, padx=10, pady=10)

stream_button = customtkinter.CTkButton( app, text="Open stream", width=200, height=100, command=open_stream )
stream_button.grid(row=0, column=3, padx=10, pady=10)

close_stream_button = customtkinter.CTkButton( app, text="Close stream", width=200, height=100, command=close_stream )
close_stream_button.grid(row=0, column=4, padx=10, pady=10)
start_autocapture_button = customtkinter.CTkButton( app, text="Start auto capture", width=200, height=100, command=autocapture )
start_autocapture_button.grid(row=3, column=0, padx=10, pady=10)

open_serial_button = customtkinter.CTkButton( app, text="Open serial connection", width=200, height=100, command=open_serial_connection )
open_serial_button.grid(row=3, column=1, padx=10, pady=10)


combobox.grid(row=1, column=0, padx=10, pady=10)
serial_combobox.grid(row=2, column=0, padx=10, pady=10)

video = customtkinter.CTkLabel(
    app,
    width=1280,
    height=720,
    text=""
)

video.grid(
    row=1,
    column=1,
    columnspan=3,
    padx=10,
    pady=10
)

#make the columns expand equally when the window is resized
app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=1)
app.grid_columnconfigure(2, weight=1)
app.grid_columnconfigure(3, weight=1)
app.grid_columnconfigure(4, weight=1)

app.mainloop()