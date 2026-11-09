"""Shared state and device association for native Echo sensor capabilities.

SPDX-License-Identifier: Apache-2.0
"""

from datetime import datetime
from typing import Any

from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity


def get_echo_client(account_dict: dict, details: dict):
    """Return the already selected media player for this account's Echo.

    Media-player discovery applies the account's include/exclude filters. Match
    the physical serial, never a friendly name, and retain its account-scoped
    unique ID and device identifiers.
    """
    serial = details.get("device_serial")
    if not serial or serial not in account_dict["devices"]["media_player"]:
        return None
    client = account_dict["entities"]["media_player"].get(serial)
    if client is None or not (client.device_info or {}).get("identifiers"):
        return None
    return client


def get_enabled_echo_entities(account_dict: dict) -> set[str]:
    """Find light endpoints to poll even when temperature is disabled."""
    return {
        sensor.alexa_entity_id
        for sensors in account_dict["entities"]["sensor"].values()
        if isinstance(sensors, dict)
        and (sensor := sensors.get("Illuminance")) is not None
        and sensor.enabled
    }


class EchoSensor(CoordinatorEntity):
    """A sensor read through the existing account-wide Phoenix coordinator."""

    _attr_has_entity_name = True

    def __init__(self, coordinator, entity_id: str, client, suffix: str) -> None:
        """Attach to the existing Echo without creating a second device."""
        super().__init__(coordinator)
        self.alexa_entity_id = entity_id
        self._client = client
        self._attr_unique_id = f"{client.unique_id}_{suffix}"
        self._attr_device_info = DeviceInfo(
            identifiers=set(client.device_info["identifiers"])
        )

    def _reading(self, namespace: str, name: str) -> dict[str, Any] | None:
        """Read only a matching, well-formed capability from the latest poll."""
        data = self.coordinator.data
        if not isinstance(data, dict):
            return None
        readings = data.get(self.alexa_entity_id)
        if not isinstance(readings, list):
            return None
        return next(
            (
                reading
                for reading in readings
                if isinstance(reading, dict)
                and reading.get("namespace") == namespace
                and reading.get("name") == name
                and not reading.get("error")
            ),
            None,
        )

    @property
    def available(self) -> bool:
        """Require a successful poll and an online Echo."""
        return super().available and self._client.available

    @staticmethod
    def _sample_time(reading: dict[str, Any] | None) -> str | None:
        """Expose Amazon's timestamp without treating last-change time as a TTL."""
        value = (reading or {}).get("timeOfSample")
        if not isinstance(value, str):
            return None
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return stamp.isoformat() if stamp.tzinfo is not None else None
