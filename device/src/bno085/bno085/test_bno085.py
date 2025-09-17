import time
import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_MAGNETOMETER
)

# I2C-Instanz erstellen
i2c = busio.I2C(board.SCL, board.SDA)

try:
    # BNO085 initialisieren
    sensor = BNO08X_I2C(i2c)
    print("✅ BNO085 erfolgreich erkannt!")

    # Sensor-Modi aktivieren
    sensor.enable_feature(BNO_REPORT_ACCELEROMETER)  # Beschleunigung
    sensor.enable_feature(BNO_REPORT_GYROSCOPE)      # Gyroskop
    sensor.enable_feature(BNO_REPORT_MAGNETOMETER)   # Magnetometer

    # Endlosschleife: Sensordaten auslesen
    while True:
        accel_x, accel_y, accel_z = sensor.acceleration  # m/s²
        gyro_x, gyro_y, gyro_z = sensor.gyro             # rad/s
        mag_x, mag_y, mag_z = sensor.magnetic           # µT

        print(f"📊 Beschleunigung (m/s²): X={accel_x:.2f}, Y={accel_y:.2f}, Z={accel_z:.2f}")
        print(f"🔄 Gyroskop (rad/s): X={gyro_x:.2f}, Y={gyro_y:.2f}, Z={gyro_z:.2f}")
        print(f"🧭 Magnetometer (µT): X={mag_x:.2f}, Y={mag_y:.2f}, Z={mag_z:.2f}")
        print("-" * 40)
        
        time.sleep(0.5)  # 500 ms warten

except Exception as e:
    print(f"❌ Fehler: {e}")
