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

    if (
        main_server.assistant is None
        or main_server.assistant.mqtt is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Assistant chưa kết nối"
        )

    print("==== API LIGHT ====")
    print(main_server.assistant)
    print(main_server.assistant.mqtt)

    main_server.assistant.mqtt.publish(
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
    
@router.get("/debug")
def debug():
    import server.main as main_server

    return {
        "assistant": str(main_server.assistant),
        "has_mqtt": (
            main_server.assistant is not None
            and getattr(main_server.assistant, "mqtt", None) is not None
        )
    }