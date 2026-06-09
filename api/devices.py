from fastapi import APIRouter
from fastapi import HTTPException

from services.state_service import home_state

router = APIRouter(
    prefix="/api/devices",
    tags=["Devices"]
)


@router.post("/light")
def control_light(status: bool):

    # Import module server.main lazily to avoid circular import at module import time
    import server.main as main_server

    if getattr(main_server, "mqtt", None) is None:
        raise HTTPException(
            status_code=503,
            detail="Assistant chưa kết nối"
        )

    main_server.mqtt.publish(
        "iot",
        {
            "device": "LIGHT_LIVING",
            "action": "ON" if status else "OFF"
        }
    )

    # cập nhật local state
    # home_state.update_device(
    #     "LIGHT_LIVING",
    #     status
    # )

    return {
        "success": True,
        "status": status
    }