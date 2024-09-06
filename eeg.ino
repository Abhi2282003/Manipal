// EEG Data Transmission from Arduino

// Define the analog pins for two EEG channels
const int channel1Pin = A0;  // First EEG sensor connected to A0
const int channel2Pin = A1;  // Second EEG sensor connected to A1

// Baud rate for serial communication
const unsigned long baudRate = 115200;

// Sampling parameters
const unsigned long samplingRate = 256;        // Desired sampling rate in Hz
const unsigned long samplingInterval = 1000000 / samplingRate;  // Interval in microseconds

unsigned long previousMicros = 0;

void setup() {
  // Initialize serial communication at the specified baud rate
  Serial.begin(baudRate);
  
  // Initialize analog pins (optional, since they're input by default)
  pinMode(channel1Pin, INPUT);
  pinMode(channel2Pin, INPUT);
  
  // Wait for serial port to connect (useful for some boards)
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  Serial.println("EEG Data Transmission Started");
}

void loop() {
  unsigned long currentMicros = micros();
  
  // Check if it's time to take a new sample
  if (currentMicros - previousMicros >= samplingInterval) {
    previousMicros = currentMicros;
    
    // Read the analog values from the two channels
    int channel1Value = analogRead(channel1Pin);
    int channel2Value = analogRead(channel2Pin);
    
    // Convert analog readings to voltage (assuming 10-bit ADC and 5V reference)
    float voltage1 = (channel1Value / 1023.0) * 5.0;
    float voltage2 = (channel2Value / 1023.0) * 5.0;
    
    // Send the two voltage values over serial in CSV format: "voltage1,voltage2"
    Serial.print(voltage1, 4);  // 4 decimal places
    Serial.print(",");
    Serial.println(voltage2, 4);
    
    // Optional: Print to Serial Monitor for debugging
    // Serial.print("Ch1: ");
    // Serial.print(voltage1, 4);
    // Serial.print(" V, Ch2: ");
    // Serial.print(voltage2, 4);
    // Serial.println(" V");
  }
}