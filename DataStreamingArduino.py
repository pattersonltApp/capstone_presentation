# Luke Patterson
# 4/14/2022
# This program will read serial out from an arduino
#   and record the information over a given timeframe.
import time
import keyboard
from datetime import datetime

import numpy as np
import pandas as pd

import serial
import serial.tools.list_ports


HANDSHAKE = 0
VOLTAGE_REQUEST = 1
ON_REQUEST = 2
STREAM = 3
READ_DAQ_DELAY = 4


def find_arduino(port=None):
    """Get the name of the port that is connected to Arduino."""
    if port is None:
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if p.manufacturer is not None and "Arduino" in p.manufacturer:
                port = p.device
    return port


def handshake_arduino(arduino, sleep_time=1, print_handshake_message=False, handshake_code=0):
    """Make sure connection is established by sending
    and receiving bytes."""
    # Close and reopen
    arduino.close()
    arduino.open()

    # Chill out while everything gets set
    time.sleep(sleep_time)

    # Set a long timeout to complete handshake
    timeout = arduino.timeout
    arduino.timeout = 2

    # Read and discard everything that may be in the input buffer
    _ = arduino.read_all()

    # Send request to Arduino
    arduino.write(bytes([handshake_code]))

    # Read in what Arduino sent
    handshake_message = arduino.read_until()

    # Send and receive request again
    arduino.write(bytes([handshake_code]))
    handshake_message = arduino.read_until()

    # Print the handshake message, if desired
    if print_handshake_message:
        print("Handshake message: " + handshake_message.decode())

    # Reset the timeout
    arduino.timeout = timeout


def parse_raw(raw):
    """Parse bytes output from Arduino."""
    raw = raw.decode()
    if raw[-1] != "\n":
        raise ValueError(
            "Input must end with newline, otherwise message is incomplete."
        )

    t, V0, V1, V2, V3 = raw.rstrip().split(",")

    return int(t), int(V0), int(V1), int(V2), int(V3)
    #return int(t), int(V) * 5 / 1023


def daq_stream(arduino, n_data=100, delay=20):
    """Obtain `n_data` data points from an Arduino stream
    with a delay of `delay` milliseconds between each."""
    # Specify delay
    arduino.write(bytes([READ_DAQ_DELAY]) + (str(delay) + "x").encode())

    # Initialize output
    time_ms = np.empty(n_data)
    signal0 = np.empty(n_data)
    signal1 = np.empty(n_data)
    signal2 = np.empty(n_data)
    signal3 = np.empty(n_data)

    # Turn on the stream
    arduino.write(bytes([STREAM]))

    # Receive data
    i = 0
    while i < n_data:
        raw = arduino.read_until()

        try:
            t, V0, V1, V2, V3 = parse_raw(raw)
            time_ms[i] = t
            signal0[i] = V0
            signal1[i] = V1
            signal2[i] = V2
            signal3[i] = V3
            i += 1
        except:
            pass

    # Turn off the stream
    arduino.write(bytes([ON_REQUEST]))

    return pd.DataFrame({'time (ms)': time_ms, 'signal 0': signal0, 'signal 1': signal1,
                         'signal 2': signal2, 'signal 3': signal3})


def main():
    port = find_arduino()
    arduino = serial.Serial(port, baudrate=115200)
    handshake_arduino(arduino, handshake_code=HANDSHAKE, print_handshake_message=True)

    while True:
        if keyboard.is_pressed('p'):
            label = input('Enter a label for record: ')
            now = datetime.now()
            filename = label + '-' + now.strftime('%m-%d_%H%M%S') + '.csv'
            arduino.flush()
            df = daq_stream(arduino, n_data=1000, delay=10) # 1000 Datapoints over 10 seconds

            df['time (sec)'] = df['time (ms)'] / 1000
            df.to_csv(filename, index=False)

            # p = bokeh.plotting.figure(
            #     x_axis_label='time (s)',
            #     y_axis_label='Signal',
            #     frame_height=175,
            #     frame_width=500,
            #     x_range=[df['time (sec)'].min(), df['time (sec)'].max()],
            # )
            # p.line(source=df, x='time (sec)', y='EMG signal')
            # bokeh.io.show(p)
            print('Press p to continue...')


if __name__ == '__main__':
    main()
