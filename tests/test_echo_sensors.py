"""Tests for capability-based Echo motion and illuminance sensors."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from custom_components.alexa_media import (
    binary_sensor as binary_sensor_platform,
    sensor as sensor_platform,
)
from custom_components.alexa_media.alexa_entity import (
    is_echo_sensor,
    parse_alexa_entities,
)
from custom_components.alexa_media.binary_sensor import EchoMotionSensor
from custom_components.alexa_media.echo_sensor import (
    get_echo_client,
    get_enabled_echo_entities,
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


def test_light_polling_does_not_depend_on_temperature():
    """Enabled light entities must remain monitored when temperature is absent."""
    light = SimpleNamespace(enabled=True, alexa_entity_id="endpoint-garden")
    account = {"entities": {"sensor": {"echo-garden": {"Illuminance": light}}}}

    assert set(get_enabled_echo_entities(account)) == {"endpoint-garden"}
    light.enabled = False
    assert set(get_enabled_echo_entities(account)) == set()


def test_light_polling_ignores_unrelated_or_incomplete_entries():
    """Non-Echo and partly initialized sensor records must not break polling."""
    account = {
        "entities": {
            "sensor": {
                "one": None,
                "two": {},
                "three": {
                    "Temperature": SimpleNamespace(enabled=True, alexa_entity_id="temp")
                },
            }
        }
    }
    assert set(get_enabled_echo_entities(account)) == set()


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
