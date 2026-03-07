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
    with serial.Serial(port, baudrate, timeout=1 ) as ser:   # for avoiding reset of Arduino at start of serial: dsrdtr=True 
        time.sleep(5)  # Allow Arduino reset

        with open(file_path, 'rb') as file:
            data = file.read()
            size = len(data)
            crc = zlib.crc32(data)

        print(f"Sending '{file_path}'")
        print(f"File size: {size} bytes, CRC32: {crc:#010x}")

        # Empty serial buffer
        ser.reset_input_buffer()
        
        # if ser.in_waiting > 0:
           # ser.read(ser.in_waiting)

        # Send start command
        # ser.write(b'UPLOAD_SETFILE\n')
        # print("COM:--UPLOAD_SETFILE--")
        # time.sleep(1)

        # # Send header: "size,crc"
        # ser.write("smalline\n".encode())
        # ser.write(b'smalline\n')
        # header = f"FSIZE:{size},{crc}\n".encode()
        # ser.write(header)
        # ser.write('\n'.encode('ascii'))
        # print("COM:--",end='')
        # print(header,end='')
        # print("--")
        # time.sleep(0.1)
        
        # Send start command with file size and CRC in one line
        header = f"UPLOAD_SETFILE:{size},{crc}*\r\n"
        print("COM:--",end='')
        print(header,end='')
        print("--")
        ser.write(header.encode('ASCII'))
        # ser.write(b'UPLOAD_SETFILE\n')
        
        time.sleep(1)
        
        # Wait for READY
        ack=ser.readline().decode()
        i=0
        while( ack.strip() != "READY"):
            i+=1
            print("Arduino not ready or error:", ack)
            ack = ser.readline().decode()
            if i>5:
                return

        # Send data
        print("Sending File")
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
