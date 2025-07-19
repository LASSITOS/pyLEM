# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 20:48:59 2025

@author: acapelli
"""

import serial
import time
import zlib
import argparse

def send_file(port, baudrate, file_path):
    with serial.Serial(port, baudrate, timeout=1) as ser:
        time.sleep(2)  # Allow Arduino reset

        with open(file_path, 'rb') as file:
            data = file.read()
            size = len(data)
            crc = zlib.crc32(data)

        print(f"Sending '{file_path}'")
        print(f"File size: {size} bytes, CRC32: {crc:#010x}")

        # Send start command
        ser.write(b'UPLOAD_SETFILE\n')
        time.sleep(0.5)

        # Send header: "size,crc"
        header = f"{size},{crc}\n".encode()
        ser.write(header)

        # Wait for READY
        ack = ser.readline().decode().strip()
        if ack != "READY":
            print("Arduino not ready or error:", ack)
            return

        # Send data
        ser.write(data)
        print("File sent.")

        # Read result
        result = ser.readline().decode().strip()
        print("Arduino says:", result)

def main():
    parser = argparse.ArgumentParser(description="Send setting file to LEM over Serial and save to SD card.")
    parser.add_argument('--port', required=True, help='Serial port (e.g., COM3 or /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--file', required=True, help='Path to the file to send')

    args = parser.parse_args()
    send_file(args.port, args.baud, args.file)

if __name__ == "__main__":
    main()
