import serial
import csv
import time

def main():
    # ------------------- Configuration -------------------
    SERIAL_PORT = 'COM11'       # Replace with your Arduino's COM port
    BAUD_RATE = 115200          # Must match the Arduino's baud rate
    SAMPLE_RATE = 256           # Samples per second (Hz)
    CSV_FILENAME = 'eeg_data_basic.csv'
    BUFFER_SIZE = 256           # Set to power of 2 for FFT efficiency
    # ----------------------------------------------------

    # Attempt to connect to the serial port
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Serial connection established on {SERIAL_PORT} at {BAUD_RATE} baud.")
    except serial.SerialException as e:
        print(f"Failed to connect to {SERIAL_PORT}: {e}")
        return

    # Wait briefly to allow the serial connection to initialize
    time.sleep(2)

    # Open the CSV file for writing
    try:
        with open(CSV_FILENAME, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write CSV header
            csvwriter.writerow([
                'Time (s)',
                'Channel 1 (V)',
                'Channel 2 (V)'
            ])
            print(f"Logging data to {CSV_FILENAME}...")
            
            # Initialize data buffers
            channel1_data = []
            channel2_data = []
            
            # Record the start time
            start_time = time.time()
            
            while True:
                try:
                    # Read a line from the serial port
                    line = ser.readline().decode('utf-8').strip()
                    
                    if line:
                        # Split the line into two parts
                        data = line.split(',')
                        
                        if len(data) != 2:
                            print(f"Invalid data format: {data}")
                            continue  # Skip invalid lines
                        
                        try:
                            # Parse the voltage values
                            channel1_val = float(data[0])
                            channel2_val = float(data[1])
                        except ValueError as ve:
                            print(f"Value parsing error: {ve} | Data: {data}")
                            continue  # Skip lines with non-numeric data
                        
                        # Calculate elapsed time
                        current_time = time.time() - start_time
                        
                        # Write the data to the CSV file
                        csvwriter.writerow([
                            round(current_time, 4),
                            round(channel1_val, 4),
                            round(channel2_val, 4)
                        ])
                        csvfile.flush()  # Ensure data is written to disk
                        
                        print(f"Logged -> Time: {round(current_time, 4)}s | Ch1: {round(channel1_val, 4)}V | Ch2: {round(channel2_val, 4)}V")
                    
                except KeyboardInterrupt:
                    print("\nRecording stopped by user.")
                    break
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    break

    except IOError as e:
        print(f"Failed to open {CSV_FILENAME} for writing: {e}")
    finally:
        # Close the serial connection
        ser.close()
        print("Serial connection closed.")

if __name__ == "__main__":
    main()
