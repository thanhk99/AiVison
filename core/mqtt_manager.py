import paho.mqtt.client as mqtt
import logging
import json
import threading
import time

import asyncio

from services.state_service import home_state
from server.dashboard_ws import dashboard_manager

logger = logging.getLogger("MqttManager")

class MqttManager:
    """Quản lý kết nối và giao tiếp MQTT."""
    
    def __init__(self, config):
        self.config = config.get("mqtt", {})
        self.enabled = self.config.get("enabled", False)
        
        if not self.enabled:
            logger.info("MQTT is disabled in config.")
            return

        self.broker = self.config.get("broker", "localhost")
        self.port = self.config.get("port", 1883)
        self.user = self.config.get("user", "")
        self.password = self.config.get("password", "")
        self.client_id = self.config.get("client_id", "ai_assistant")
        self.topics = self.config.get("topics", {})

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, self.client_id)
        
        if self.user and self.password:
            self.client.username_pw_set(self.user, self.password)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self.command_callback = None
        self._is_connected = False

    def connect(self):
        """Kết nối tới MQTT Broker."""
        if not self.enabled:
            return
        
        try:
            logger.info(f"Connecting to MQTT Broker at {self.broker}:{self.port}...")
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to connect to MQTT Broker: {e}")

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to MQTT Broker successfully.")
            self._is_connected = True
            # Subscribe to command topic
            command_topic = self.topics.get("command", "home/assistant/command")
            self.client.subscribe(command_topic)
            logger.info(f"Subscribed to topic: {command_topic}")
            
            sensor_topic = self.topics.get("sensor")

            if sensor_topic:
                self.client.subscribe(sensor_topic)

            device_status_topic = self.topics.get("device_status")
            
            # Publish initial status
            if device_status_topic:
                self.client.subscribe(device_status_topic)
            
            self.publish_status({"state": "online", "message": "Quản gia đã sẵn sàng"})
        else:
            logger.error(f"Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._is_connected = False
        logger.warning(f"Disconnected from MQTT Broker with code {rc}")

    def _on_message(self, client, userdata, msg):
        """Xử lý tin nhắn đến."""
        try:
            payload = msg.payload.decode()
            logger.info(f"Received message on {msg.topic}: {payload}")
            
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = payload # Handle plain text commands

            if msg.topic == self.topics.get("command"):
                if self.command_callback:
                    # Chạy callback trong thread riêng để không block loop của MQTT
                    threading.Thread(target=self.command_callback, args=(data,), daemon=True).start()
                return
            
            # ==========================================
            # Sensor Data
            # Topic: home/sensor
            # ==========================================
            
            if msg.topic == self.topics.get("sensor"):
                temperature = data.get("temperature",0)
                humidity = data.get("humidity",0)
                
                home_state.update_sensor(temperature,humidity)
                logger.info(f"[STATE] Temp={temperature} Hum={humidity}")
                
                self._broadcast_dashboard(
                    {
                        "type": "sensor_update",
                        "temperature": temperature,
                        "humidity": humidity
                    }
                )
                return            
            
            #==========================================
            # Device Status
            # Topic: home/device/status
            # ==========================================
            
            if msg.topic == self.topics.get("device_status"):
                device = data.get("device")
                status = data.get("status")
                if device is not None:
                # cập nhật state
                    home_state.update_device(device,status)

                # realtime dashboard
                self._broadcast_dashboard(
                    {
                        "type": "device_update",
                        "device": device,
                        "status": status
                    }
                )

                logger.info(
                    f"Device updated: "
                    f"{device} -> {status}"
                )

            return
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    def set_command_callback(self, callback):
        """Thiết lập callback để xử lý lệnh nhận được."""
        self.command_callback = callback

    def publish(self, topic_key, payload):
        """Gửi tin nhắn MQTT dựa trên topic key trong config."""
        if not self.enabled or not self._is_connected:
            return

        topic = self.topics.get(topic_key)
        if not topic:
            logger.error(f"Topic key '{topic_key}' not found in config.")
            return

        try:
            if isinstance(payload, (dict, list)):
                payload = json.dumps(payload, ensure_ascii=False)
            
            logger.info(f"[MQTT SEND] topic={topic} payload={payload}")
            self.client.publish(topic, payload, qos=1)
            # logger.debug(f"Published to {topic}: {payload}")
        except Exception as e:
            logger.error(f"Failed to publish MQTT message: {e}")

    def publish_status(self, status):
        """Gửi trạng thái hệ thống."""
        self.publish("status", status)

    def publish_event(self, event_type, details):
        """Gửi sự kiện hệ thống."""
        self.publish("event", {
            "event": event_type,
            "details": details,
            "timestamp": time.time()
        })

    def stop(self):
        """Dừng kết nối MQTT."""
        if self.enabled:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("MQTT connection stopped.")
            
    def _broadcast_dashboard(self, payload):
        try:
            asyncio.run(dashboard_manager.broadcast(payload))
        except Exception as e:
            logger.error(f"Dashboard broadcast error: {e}")
