from fastapi import APIRouter
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from services.state_service import home_state
from server.dashboard_ws import dashboard_manager

router = APIRouter(
    prefix="/api/dashboard",
    tags=["Dashboard"]
)


@router.get("")
def get_dashboard():
    return home_state.get_state()


@router.websocket("/ws")
async def dashboard_ws(websocket: WebSocket):

    await dashboard_manager.connect(websocket)

    try:

        await websocket.send_json(
            {
                "type": "initial_state",
                "state": home_state.get_state()
            }
        )

        while True:

            # giữ kết nối
            await websocket.receive_text()

    except WebSocketDisconnect:

        dashboard_manager.disconnect(websocket)

    except Exception:

        dashboard_manager.disconnect(websocket)