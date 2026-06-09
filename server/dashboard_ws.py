import logging
from fastapi import WebSocket

logger = logging.getLogger("DashboardWS")


class DashboardManager:

    def __init__(self):
        self.connections = set()

    async def connect(self, websocket: WebSocket):

        await websocket.accept()

        self.connections.add(websocket)

        logger.info(
            f"Dashboard connected. Total={len(self.connections)}"
        )

    def disconnect(self, websocket: WebSocket):

        self.connections.discard(websocket)

        logger.info(
            f"Dashboard disconnected. Total={len(self.connections)}"
        )

    async def broadcast(self, payload: dict):

        disconnected = set()

        for ws in self.connections:

            try:
                await ws.send_json(payload)

            except Exception:
                disconnected.add(ws)

        for ws in disconnected:
            self.disconnect(ws)


dashboard_manager = DashboardManager()