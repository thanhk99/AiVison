from threading import Lock


class HomeState:

    def __init__(self):
        self._lock = Lock()

        self.temperature = 0.0
        self.humidity = 0.0

        self.devices = {
            "LIGHT_LIVING": False
        }

    def update_sensor(self, temperature: float, humidity: float):
        with self._lock:
            self.temperature = temperature
            self.humidity = humidity

    def update_device(self, device: str, status: bool):
        with self._lock:
            self.devices[device] = status

    def get_state(self):
        with self._lock:
            return {
                "temperature": self.temperature,
                "humidity": self.humidity,
                "devices": self.devices.copy()
            }


home_state = HomeState()