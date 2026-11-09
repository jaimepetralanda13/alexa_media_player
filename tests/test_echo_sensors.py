"""Tests for capability-based Echo motion and illuminance sensors."""

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from alexapy import (
    AlexapyConnectionError,
    AlexapyLoginCloseRequested,
    AlexapyLoginError,
)
from alexapy.errors import AlexapyTooManyRequestsError
from homeassistant.helpers.update_coordinator import UpdateFailed
import pytest

import custom_components.alexa_media as alexa_media
from custom_components.alexa_media import (
    binary_sensor as binary_sensor_platform,
    echo_sensor as echo_sensor_module,
    sensor as sensor_platform,
)
from custom_components.alexa_media.alexa_entity import (
    is_echo_sensor,
    parse_alexa_entities,
)
from custom_components.alexa_media.binary_sensor import EchoMotionSensor
from custom_components.alexa_media.echo_sensor import (
    EchoSensorCoordinator,
    get_echo_client,
    get_echo_coordinator,
)
from custom_components.alexa_media.sensor import EchoLightSensor


def capability(namespace, name):
    """Return one retrievable sensor capability."""
    return {
        "interfaceName": namespace,
        "properties": {
            "retrievable": True,
            "proactivelyReported": True,
            "supported": [{"name": name}],
        },
    }


@pytest.fixture
def echo_appliance():
    """Provide a synthetic Echo endpoint exposing three independent sensors."""
    return {
        "entityId": "endpoint-garden",
        "applianceId": "appliance-garden",
        "friendlyName": "Garden Echo",
        "friendlyDescription": "Amazon Echo",
        "manufacturerName": "Amazon",
        "modelName": "Echo",
        "applianceTypes": ["ALEXA_VOICE_ENABLED"],
        "driverIdentity": {"namespace": "AAA"},
        "alexaDeviceIdentifierList": [{"dmsDeviceSerialNumber": "echo-garden"}],
        "capabilities": [
            capability("Alexa.TemperatureSensor", "temperature"),
            capability("Alexa.MotionSensor", "detectionState"),
            capability("Alexa.LightSensor", "illuminance"),
        ],
    }


@pytest.fixture
def details():
    """Describe one sensor endpoint without user-specific identifiers."""
    return {
        "id": "endpoint-garden",
        "appliance_id": "appliance-garden",
        "name": "Garden Echo",
        "is_hue_v1": False,
        "device_serial": "echo-garden",
    }


@pytest.fixture
def client():
    """Provide the already-loaded media player for this account."""
    return SimpleNamespace(
        name="Garden Echo",
        unique_id="account-a_echo-garden",
        device_serial_number="echo-garden",
        available=True,
        device_info={"identifiers": {("alexa_media", "account-a_echo-garden")}},
    )


def coordinator_for(namespace, name, value, timestamp="2024-01-01T00:00:00Z"):
    """Model a successful poll that can report an unchanged historical state."""
    return SimpleNamespace(
        data={
            "endpoint-garden": [
                {
                    "namespace": namespace,
                    "name": name,
                    "value": value,
                    "timeOfSample": timestamp,
                }
            ]
        },
        last_update_success=True,
    )


def test_discovery_adds_independent_echo_capabilities(echo_appliance):
    """Temperature must not hide motion and light on the same endpoint."""
    result = parse_alexa_entities([echo_appliance])

    for kind in ("temperature", "motion", "illuminance"):
        assert len(result[kind]) == 1
        assert result[kind][0]["id"] == "endpoint-garden"
        assert result[kind][0]["device_serial"] == "echo-garden"


@pytest.mark.parametrize(
    ("namespace", "name", "kind"),
    [
        ("Alexa.MotionSensor", "detectionState", "motion"),
        ("Alexa.LightSensor", "illuminance", "illuminance"),
    ],
)
def test_sensor_discovery_does_not_require_temperature(
    echo_appliance, namespace, name, kind
):
    """Echo sensors remain discoverable on devices with no temperature sensor."""
    echo_appliance["capabilities"] = [capability(namespace, name)]

    result = parse_alexa_entities([echo_appliance])

    assert len(result[kind]) == 1
    assert result["temperature"] == []


@pytest.mark.parametrize(
    "namespace,name",
    [
        ("Alexa.MotionSensor", "detectionState"),
        ("Alexa.LightSensor", "illuminance"),
    ],
)
def test_discovery_rejects_skill_endpoints(echo_appliance, namespace, name):
    """Do not reimport sensors exposed to Alexa by a skill or HA bridge."""
    echo_appliance["driverIdentity"] = {"namespace": "SKILL"}
    echo_appliance["connectedVia"] = "A misleading Echo name"

    assert not is_echo_sensor(echo_appliance, namespace, name)
    result = parse_alexa_entities([echo_appliance])
    assert result["motion"] == []
    assert result["illuminance"] == []


def test_discovery_does_not_treat_acoustic_presence_as_motion(echo_appliance):
    """An old acoustic presence capability is not ultrasonic motion support."""
    echo_appliance["capabilities"] = [
        capability("Alexa.AcousticEventSensor", "humanPresenceDetectionState"),
        capability("Alexa.AcousticEventSensor", "dogBarkDetectionState"),
    ]

    result = parse_alexa_entities([echo_appliance])

    assert result["motion"] == []
    assert result["illuminance"] == []


def test_discovery_requires_reportable_capability(echo_appliance):
    """A declared but neither retrievable nor reported property is unsupported."""
    echo_appliance["capabilities"] = [
        capability("Alexa.MotionSensor", "detectionState")
    ]
    props = echo_appliance["capabilities"][0]["properties"]
    props["retrievable"] = False
    props["proactivelyReported"] = False

    assert not is_echo_sensor(echo_appliance, "Alexa.MotionSensor", "detectionState")


def test_discovery_rejects_unrelated_local_sensor(echo_appliance):
    """A third-party local motion sensor must not become an Echo sensor."""
    echo_appliance["manufacturerName"] = "Synthetic Sensor Vendor"
    echo_appliance["modelName"] = "PIR Sensor"
    echo_appliance["applianceTypes"] = ["MOTION_SENSOR"]
    echo_appliance["connectedVia"] = "Garden Echo"
    echo_appliance.pop("alexaDeviceIdentifierList")

    assert not is_echo_sensor(echo_appliance, "Alexa.MotionSensor", "detectionState")


def test_amazon_other_endpoint_requires_serial_and_bound_client(echo_appliance, client):
    """Amazon's generic sensor endpoint still needs a selected Echo to attach to."""
    echo_appliance["applianceTypes"] = ["OTHER"]
    result = parse_alexa_entities([echo_appliance])
    details = result["motion"][0]
    account = {
        "devices": {"media_player": {}},
        "entities": {"media_player": {}},
    }
    assert get_echo_client(account, details) is None
    account["devices"]["media_player"]["echo-garden"] = {}
    account["entities"]["media_player"]["echo-garden"] = client
    assert get_echo_client(account, details) is client

    echo_appliance.pop("alexaDeviceIdentifierList")
    assert not is_echo_sensor(echo_appliance, "Alexa.MotionSensor", "detectionState")


def test_empty_discovery_has_sensor_lists():
    """No network details is an empty result, not a setup failure."""
    result = parse_alexa_entities(None)
    assert result["motion"] == []
    assert result["illuminance"] == []


@pytest.mark.parametrize(
    "value,expected", [("DETECTED", True), ("NOT_DETECTED", False)]
)
def test_motion_preserves_valid_unchanged_state(details, client, value, expected):
    """Sample time describes a state change and must not impose a stale timeout."""
    coordinator = coordinator_for("Alexa.MotionSensor", "detectionState", value)
    sensor = EchoMotionSensor(coordinator, details, client)

    assert sensor.available is True
    assert sensor.is_on is expected


@pytest.mark.parametrize("timestamp", [None, 123, "bad timestamp", "2024-01-01", {}])
def test_motion_malformed_timestamp_does_not_break_valid_state(
    details, client, timestamp
):
    """Timestamp metadata must not throw or erase a valid reported motion state."""
    coordinator = coordinator_for(
        "Alexa.MotionSensor", "detectionState", "NOT_DETECTED", timestamp
    )
    sensor = EchoMotionSensor(coordinator, details, client)

    assert sensor.available is True
    assert sensor.is_on is False
    assert isinstance(sensor.extra_state_attributes, dict)


@pytest.mark.parametrize("value", [None, "UNKNOWN", "", 0, 1, True, [], {}])
def test_motion_invalid_value_is_unknown(details, client, value):
    """Malformed responses must never be presented as an unoccupied room."""
    coordinator = coordinator_for("Alexa.MotionSensor", "detectionState", value)
    sensor = EchoMotionSensor(coordinator, details, client)

    assert sensor.available is False
    assert sensor.is_on is None


@pytest.mark.parametrize("data", [None, {}, {"endpoint-garden": []}])
def test_missing_response_is_unavailable(details, client, data):
    """Missing endpoint data must not retain an apparently current sensor state."""
    coordinator = SimpleNamespace(data=data, last_update_success=True)
    for cls in (EchoMotionSensor, EchoLightSensor):
        assert cls(coordinator, details, client).available is False


@pytest.mark.parametrize(
    "cls,namespace,name,value",
    [
        (EchoMotionSensor, "Alexa.MotionSensor", "detectionState", "DETECTED"),
        (EchoLightSensor, "Alexa.LightSensor", "illuminance", 10),
    ],
)
def test_sensor_tracks_poll_and_device_availability(
    details, client, cls, namespace, name, value
):
    """Failures and device disconnects invalidate both sensor types immediately."""
    coordinator = coordinator_for(namespace, name, value)
    sensor = cls(coordinator, details, client)
    assert sensor.available is True

    coordinator.last_update_success = False
    assert sensor.available is False
    coordinator.last_update_success = True
    client.available = False
    assert sensor.available is False
    client.available = True
    assert sensor.available is True


@pytest.mark.parametrize("error", [True, {"code": "ENDPOINT_UNREACHABLE"}])
@pytest.mark.parametrize(
    "cls,namespace,name,value",
    [
        (EchoMotionSensor, "Alexa.MotionSensor", "detectionState", "DETECTED"),
        (EchoLightSensor, "Alexa.LightSensor", "illuminance", 10),
    ],
)
def test_capability_error_is_unavailable(
    details, client, cls, namespace, name, value, error
):
    """A per-capability error is not hidden by an otherwise successful poll."""
    coordinator = coordinator_for(namespace, name, value)
    coordinator.data["endpoint-garden"][0]["error"] = error
    assert cls(coordinator, details, client).available is False


def test_disabled_motion_is_unavailable_without_fabricating_vacancy(details, client):
    """Disabled detection cannot present the last reported state as a live reading."""
    coordinator = coordinator_for("Alexa.MotionSensor", "detectionState", "DETECTED")
    coordinator.data["endpoint-garden"].append(
        {
            "namespace": "Alexa.MotionSensor",
            "name": "enablement",
            "value": "DISABLED",
        }
    )
    sensor = EchoMotionSensor(coordinator, details, client)

    assert sensor.available is False
    assert sensor.is_on is True
    assert sensor.extra_state_attributes["detection_enabled"] == "DISABLED"


@pytest.mark.parametrize("value", [0, 0.0, 0.25, 25, 1605])
def test_light_accepts_finite_nonnegative_readings(details, client, value):
    """Darkness and fractional light readings are genuine data, not missing data."""
    coordinator = coordinator_for("Alexa.LightSensor", "illuminance", value)
    sensor = EchoLightSensor(coordinator, details, client)

    assert sensor.available is True
    assert sensor.native_value == value


@pytest.mark.parametrize(
    "value",
    [None, True, False, -1, "12", [], {}, float("nan"), float("inf"), -float("inf")],
)
def test_light_rejects_invalid_values(details, client, value):
    """Invalid illuminance must not become zero or poison numeric statistics."""
    coordinator = coordinator_for("Alexa.LightSensor", "illuminance", value)
    sensor = EchoLightSensor(coordinator, details, client)

    assert sensor.available is False
    assert sensor.native_value is None


def test_light_handles_integer_overflow(details, client):
    """Oversized API integers are unavailable rather than raising during validation."""
    coordinator = coordinator_for("Alexa.LightSensor", "illuminance", 10**10000)
    sensor = EchoLightSensor(coordinator, details, client)
    assert sensor.available is False
    assert sensor.native_value is None


@pytest.mark.parametrize("timestamp", [None, 123, "bad timestamp", "2024-01-01", {}])
def test_light_handles_malformed_timestamp_metadata(details, client, timestamp):
    """Valid light values survive malformed optional timestamp metadata."""
    coordinator = coordinator_for("Alexa.LightSensor", "illuminance", 0, timestamp)
    sensor = EchoLightSensor(coordinator, details, client)

    assert sensor.available is True
    assert sensor.native_value == 0
    assert sensor.extra_state_attributes["sample_time"] is None


def test_sensor_identity_is_account_scoped_and_reuses_device(details, client):
    """Two accounts exposing one endpoint must not collide or create new devices."""
    coordinator = coordinator_for("Alexa.MotionSensor", "detectionState", "DETECTED")
    other = deepcopy(client)
    other.unique_id = "account-b_echo-garden"
    other.device_info = {"identifiers": {("alexa_media", other.unique_id)}}

    for cls in (EchoMotionSensor, EchoLightSensor):
        sensor_a = cls(coordinator, details, client)
        sensor_b = cls(coordinator, details, other)
        assert sensor_a.unique_id != sensor_b.unique_id
        assert sensor_a.coordinator_context == details["id"]
        assert sensor_a.device_info["identifiers"] == client.device_info["identifiers"]
        assert sensor_b.device_info["identifiers"] == other.device_info["identifiers"]


def test_client_lookup_stays_within_current_loaded_account(details, client):
    """Excluded or missing media players must not yield unattached Echo sensors."""
    account = {
        "devices": {"media_player": {"echo-garden": {}}},
        "entities": {"media_player": {"echo-garden": client}},
    }
    assert get_echo_client(account, details) is client
    account["entities"]["media_player"].clear()
    assert get_echo_client(account, details) is None


def test_client_lookup_rejects_unselected_echo(details, client):
    """A stale media-player reference cannot bypass current device selection."""
    account = {
        "devices": {"media_player": {}},
        "entities": {"media_player": {"echo-garden": client}},
    }
    assert get_echo_client(account, details) is None


@pytest.mark.parametrize(
    "platform,expected_class",
    [
        (binary_sensor_platform, EchoMotionSensor),
        (sensor_platform, EchoLightSensor),
    ],
)
@pytest.mark.parametrize("selected", [True, False])
async def test_platform_setup_respects_media_selection_and_is_idempotent(
    monkeypatch, details, client, platform, expected_class, selected
):
    """Only selected Echoes gain sensors, including after repeated setup calls."""
    account = {
        "coordinator": coordinator_for(
            "Alexa.MotionSensor", "detectionState", "DETECTED"
        ),
        "login_obj": None,
        "options": {},
        "devices": {
            "media_player": {"echo-garden": {"capabilities": []}} if selected else {},
            "motion": [details],
            "illuminance": [details],
        },
        "entities": {
            # A stale entity reference must not override current selection.
            "media_player": {"echo-garden": client},
            "binary_sensor": [],
            "sensor": {},
        },
    }
    hass = SimpleNamespace(
        data={"alexa_media": {"accounts": {"test@example.com": account}}}
    )
    monkeypatch.setattr(binary_sensor_platform, "async_get_state", lambda *args: None)
    monkeypatch.setattr(
        platform, "get_echo_coordinator", lambda hass, account: account["coordinator"]
    )
    added = []

    def add_entities(entities, update_before_add):
        added.extend(entities)

    config = {
        "email": "test@example.com",
        "include_devices": ["Garden Echo"],
        "exclude_devices": ["Other Echo"],
    }
    assert await platform.async_setup_platform(hass, config, add_entities)
    assert await platform.async_setup_platform(hass, config, add_entities)
    assert len(added) == int(selected)
    assert all(isinstance(entity, expected_class) for entity in added)
    if selected:
        assert added[0].device_info["identifiers"] == client.device_info["identifiers"]


def account_for_polling():
    """Supply a synthetic authenticated account without any real credentials."""
    return {
        "config_entry": None,
        "coordinator": SimpleNamespace(
            data={"endpoint-garden": []},
            last_update_success=True,
            update_interval=timedelta(minutes=10),
        ),
        "login_obj": SimpleNamespace(
            email="test@example.com",
            close_requested=False,
            session=SimpleNamespace(closed=False),
            status={"login_successful": True},
        ),
    }


def motion_response():
    """Supply a usable response with a valid, unchanged motion sample."""
    return {
        "endpoint-garden": [
            {
                "namespace": "Alexa.MotionSensor",
                "name": "detectionState",
                "value": "NOT_DETECTED",
                "timeOfSample": "2024-01-01T00:00:00Z",
            }
        ]
    }


async def test_echo_coordinator_is_per_account_and_preserves_main_interval(hass):
    """Fast sensor refresh must neither reuse another account nor speed up media APIs."""
    account = account_for_polling()
    coordinator = get_echo_coordinator(hass, account)
    other = get_echo_coordinator(hass, account_for_polling())
    try:
        assert get_echo_coordinator(hass, account) is coordinator
        assert coordinator is not other
        assert coordinator.update_interval == timedelta(seconds=60)
        assert account["coordinator"].update_interval == timedelta(minutes=10)
        assert coordinator.data == account["coordinator"].data
        assert coordinator.last_update_success is True
    finally:
        await coordinator.async_shutdown()
        await other.async_shutdown()


async def test_echo_coordinator_batches_only_active_endpoints(hass, monkeypatch):
    """Two entities on one Echo use one endpoint query, even without temperature."""
    account = account_for_polling()
    coordinator = EchoSensorCoordinator(hass, account)
    response = {"endpoint-garden": [], "endpoint-hall": []}
    fetch = AsyncMock(return_value=response)
    monkeypatch.setattr(echo_sensor_module, "get_entity_data", fetch)
    remove = [
        coordinator.async_add_listener(lambda: None, context=endpoint)
        for endpoint in ("endpoint-garden", "endpoint-garden", "endpoint-hall")
    ]
    try:
        assert await coordinator._async_update_data() == response
        fetch.assert_awaited_once()
        login, endpoints = fetch.await_args.args
        assert login is account["login_obj"]
        assert set(endpoints) == {"endpoint-garden", "endpoint-hall"}
        assert len(endpoints) == 2
    finally:
        for unsubscribe in remove:
            unsubscribe()
        await coordinator.async_shutdown()


async def test_no_active_sensors_do_not_poll(hass, monkeypatch):
    """A coordinator without enabled entities must perform no network requests."""
    coordinator = EchoSensorCoordinator(hass, account_for_polling())
    fetch = AsyncMock()
    monkeypatch.setattr(echo_sensor_module, "get_entity_data", fetch)
    try:
        assert await coordinator._async_update_data() == {}
        fetch.assert_not_awaited()
    finally:
        await coordinator.async_shutdown()


@pytest.mark.parametrize(
    "failure", ["missing", "closing", "closed_session", "bad_auth", "missing_auth"]
)
async def test_unusable_login_pauses_sensor_requests(hass, monkeypatch, failure):
    """A stale login must not generate traffic while the integration reauthenticates."""
    account = account_for_polling()
    if failure == "missing":
        account["login_obj"] = None
    elif failure == "closing":
        account["login_obj"].close_requested = True
    elif failure == "closed_session":
        account["login_obj"].session.closed = True
    elif failure == "bad_auth":
        account["login_obj"].status["login_successful"] = False
    else:
        account["login_obj"].status = {}
    coordinator = EchoSensorCoordinator(hass, account)
    previous_update = datetime(2024, 1, 1, tzinfo=timezone.utc)
    coordinator.last_update = previous_update
    fetch = AsyncMock()
    relogin = Mock()
    monkeypatch.setattr(echo_sensor_module, "get_entity_data", fetch)
    monkeypatch.setattr(echo_sensor_module, "report_relogin_required", relogin)
    unsubscribe = coordinator.async_add_listener(
        lambda: None, context="endpoint-garden"
    )
    try:
        with pytest.raises(UpdateFailed):
            await coordinator._async_update_data()
        fetch.assert_not_awaited()
        relogin.assert_not_called()
        assert coordinator.update_interval == timedelta(seconds=60)
        assert coordinator.last_update == previous_update
    finally:
        unsubscribe()
        await coordinator.async_shutdown()


@pytest.mark.parametrize("error", [AlexapyLoginError, AlexapyLoginCloseRequested])
async def test_login_failure_is_reported_without_restarting_auth(
    hass, monkeypatch, error
):
    """Sensor polling reports account errors instead of starting a separate login."""
    account = account_for_polling()
    coordinator = EchoSensorCoordinator(hass, account)
    previous_update = datetime(2024, 1, 1, tzinfo=timezone.utc)
    coordinator.last_update = previous_update
    fetch = AsyncMock(side_effect=error("synthetic auth failure"))
    relogin = Mock()
    monkeypatch.setattr(echo_sensor_module, "get_entity_data", fetch)
    monkeypatch.setattr(echo_sensor_module, "report_relogin_required", relogin)
    unsubscribe = coordinator.async_add_listener(
        lambda: None, context="endpoint-garden"
    )
    try:
        with pytest.raises(UpdateFailed):
            await coordinator._async_update_data()
        fetch.assert_awaited_once()
        assert coordinator.update_interval == timedelta(seconds=120)
        assert coordinator.last_update == previous_update
        if error is AlexapyLoginError:
            relogin.assert_called_once_with(
                hass, account["login_obj"], "test@example.com"
            )
        else:
            relogin.assert_not_called()
    finally:
        unsubscribe()
        await coordinator.async_shutdown()


async def test_each_poll_uses_current_account_login(hass, monkeypatch):
    """Reauthentication replaces credentials without rebuilding sensor entities."""
    account = account_for_polling()
    coordinator = EchoSensorCoordinator(hass, account)
    fetch = AsyncMock(return_value=motion_response())
    monkeypatch.setattr(echo_sensor_module, "get_entity_data", fetch)
    unsubscribe = coordinator.async_add_listener(
        lambda: None, context="endpoint-garden"
    )
    try:
        first_login = account["login_obj"]
        await coordinator._async_update_data()
        replacement = account_for_polling()["login_obj"]
        account["login_obj"] = replacement
        await coordinator._async_update_data()
        assert fetch.await_args_list[0].args[0] is first_login
        assert fetch.await_args_list[1].args[0] is replacement
    finally:
        unsubscribe()
        await coordinator.async_shutdown()


async def test_last_unsubscribe_stops_scheduled_polling(hass, monkeypatch):
    """Disabling or removing the last native sensor stops its refresh timer."""
    coordinator = EchoSensorCoordinator(hass, account_for_polling())
    fetch = AsyncMock(return_value={})
    monkeypatch.setattr(echo_sensor_module, "get_entity_data", fetch)
    remove_motion = coordinator.async_add_listener(
        lambda: None, context="endpoint-garden"
    )
    remove_light = coordinator.async_add_listener(
        lambda: None, context="endpoint-garden"
    )
    try:
        assert coordinator._unsub_refresh is not None
        remove_motion()
        assert coordinator._unsub_refresh is not None
        remove_light()
        assert coordinator._unsub_refresh is None
        assert list(coordinator.async_contexts()) == []
        await coordinator._async_update_data()
        fetch.assert_not_awaited()
    finally:
        await coordinator.async_shutdown()


async def test_shutdown_cancels_inflight_sensor_request(hass, monkeypatch):
    """Unloading the account cancels its outstanding network operation."""
    coordinator = EchoSensorCoordinator(hass, account_for_polling())
    previous_update = datetime(2024, 1, 1, tzinfo=timezone.utc)
    coordinator.last_update = previous_update
    coordinator.update_interval = timedelta(seconds=240)
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking_fetch(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    fetch = AsyncMock(side_effect=blocking_fetch)
    monkeypatch.setattr(echo_sensor_module, "get_entity_data", fetch)
    unsubscribe = coordinator.async_add_listener(
        lambda: None, context="endpoint-garden"
    )
    refresh = asyncio.create_task(coordinator.async_refresh())
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.wait_for(coordinator.async_shutdown(), timeout=1)
        await asyncio.gather(refresh, return_exceptions=True)
        assert cancelled.is_set()
        assert coordinator.update_interval == timedelta(seconds=240)
        assert coordinator.last_update == previous_update
        await coordinator.async_refresh()
        fetch.assert_awaited_once()
    finally:
        unsubscribe()
        if not refresh.done():
            refresh.cancel()
            await asyncio.gather(refresh, return_exceptions=True)
        await coordinator.async_shutdown()


async def test_sensor_request_timeout_cancels_network_operation(hass, monkeypatch):
    """A hung Phoenix query is bounded and releases resources for the next poll."""
    assert echo_sensor_module.ECHO_SENSOR_REQUEST_TIMEOUT == 10
    monkeypatch.setattr(echo_sensor_module, "ECHO_SENSOR_REQUEST_TIMEOUT", 0.001)
    coordinator = EchoSensorCoordinator(hass, account_for_polling())
    previous_update = datetime(2024, 1, 1, tzinfo=timezone.utc)
    coordinator.last_update = previous_update
    cancelled = asyncio.Event()

    async def blocking_fetch(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(echo_sensor_module, "get_entity_data", blocking_fetch)
    unsubscribe = coordinator.async_add_listener(
        lambda: None, context="endpoint-garden"
    )
    try:
        with pytest.raises(TimeoutError):
            await coordinator._async_update_data()
        assert cancelled.is_set()
        assert coordinator._poll_task is None
        assert coordinator.update_interval == timedelta(seconds=120)
        assert coordinator.last_update == previous_update
    finally:
        unsubscribe()
        await coordinator.async_shutdown()


async def test_sensor_poll_failures_back_off_and_recover(hass, monkeypatch):
    """Repeated failures slow requests to ten minutes until usable data returns."""
    coordinator = EchoSensorCoordinator(hass, account_for_polling())
    previous_update = datetime(2024, 1, 1, tzinfo=timezone.utc)
    coordinator.last_update = previous_update
    fetch = AsyncMock(side_effect=AlexapyConnectionError("synthetic outage"))
    monkeypatch.setattr(echo_sensor_module, "get_entity_data", fetch)
    unsubscribe = coordinator.async_add_listener(
        lambda: None, context="endpoint-garden"
    )
    try:
        assert coordinator.update_interval == timedelta(seconds=60)
        for seconds in (120, 240, 480, 600, 600, 600):
            await coordinator.async_refresh()
            assert coordinator.last_update_success is False
            assert coordinator.update_interval == timedelta(seconds=seconds)
            assert coordinator.last_update == previous_update

        fetch.side_effect = None
        fetch.return_value = motion_response()
        await coordinator.async_refresh()
        assert coordinator.last_update_success is True
        assert coordinator.data == motion_response()
        assert coordinator.update_interval == timedelta(seconds=60)
        recovered_update = coordinator.last_update
        assert recovered_update > previous_update
        assert recovered_update.utcoffset() == timedelta(0)

        fetch.side_effect = AlexapyConnectionError("second synthetic outage")
        await coordinator.async_refresh()
        assert coordinator.last_update_success is False
        assert coordinator.update_interval == timedelta(seconds=120)
        assert coordinator.last_update == recovered_update
    finally:
        unsubscribe()
        await coordinator.async_shutdown()


@pytest.mark.parametrize("response", [None, {}, [], ["invalid"], "invalid", 123])
async def test_empty_or_invalid_response_does_not_advance_heartbeat(
    hass, monkeypatch, response
):
    """A successful HTTP exchange without endpoint data is not a sensor update."""
    coordinator = EchoSensorCoordinator(hass, account_for_polling())
    previous_update = datetime(2024, 1, 1, tzinfo=timezone.utc)
    coordinator.last_update = previous_update
    fetch = AsyncMock(return_value=response)
    monkeypatch.setattr(echo_sensor_module, "get_entity_data", fetch)
    unsubscribe = coordinator.async_add_listener(
        lambda: None, context="endpoint-garden"
    )
    try:
        with pytest.raises(UpdateFailed):
            await coordinator._async_update_data()
        assert coordinator.last_update == previous_update
        assert coordinator.update_interval == timedelta(seconds=120)
        assert coordinator._poll_task is None
    finally:
        unsubscribe()
        await coordinator.async_shutdown()


@pytest.mark.parametrize(
    "error",
    [TimeoutError, AlexapyConnectionError, AlexapyTooManyRequestsError, OSError],
)
async def test_request_failure_preserves_heartbeat(hass, monkeypatch, error):
    """Connection and timeout failures cannot make cached readings look fresh."""
    coordinator = EchoSensorCoordinator(hass, account_for_polling())
    previous_update = datetime(2024, 1, 1, tzinfo=timezone.utc)
    coordinator.last_update = previous_update
    fetch = AsyncMock(side_effect=error("synthetic request failure"))
    monkeypatch.setattr(echo_sensor_module, "get_entity_data", fetch)
    unsubscribe = coordinator.async_add_listener(
        lambda: None, context="endpoint-garden"
    )
    try:
        await coordinator.async_refresh()
        assert coordinator.last_update_success is False
        assert coordinator.last_update == previous_update
        assert coordinator.update_interval == timedelta(seconds=120)
        assert coordinator._poll_task is None
        fetch.assert_awaited_once()
    finally:
        unsubscribe()
        await coordinator.async_shutdown()


async def test_account_unload_stops_sensor_polling_before_closing_login(
    hass, monkeypatch
):
    """Account teardown cannot leave requests running on a session it closes."""
    order = []
    coordinator = SimpleNamespace(
        async_shutdown=AsyncMock(side_effect=lambda: order.append("shutdown"))
    )
    other_coordinator = SimpleNamespace(async_shutdown=AsyncMock())
    account = {"echo_sensor_coordinator": coordinator}
    hass.data["alexa_media"] = {
        "accounts": {
            "test@example.com": account,
            "other@example.com": {"echo_sensor_coordinator": other_coordinator},
        }
    }
    close = AsyncMock(side_effect=lambda *_: order.append("close"))
    monkeypatch.setattr(alexa_media, "close_connections", close)
    monkeypatch.setattr(alexa_media, "ALEXA_COMPONENTS", [])
    monkeypatch.setattr(alexa_media, "DEPENDENT_ALEXA_COMPONENTS", [])
    entry = SimpleNamespace(data={"email": "test@example.com"})

    assert await alexa_media.async_unload_entry(hass, entry)

    assert order == ["shutdown", "close"]
    close.assert_awaited_once_with(hass, "test@example.com")
    coordinator.async_shutdown.assert_awaited_once()
    other_coordinator.async_shutdown.assert_not_awaited()
    assert "echo_sensor_coordinator" not in account
    assert set(hass.data["alexa_media"]["accounts"]) == {"other@example.com"}
