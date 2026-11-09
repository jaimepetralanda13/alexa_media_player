"""Shared state and device association for native Echo sensor capabilities.

SPDX-License-Identifier: Apache-2.0
"""

import asyncio
from datetime import datetime, timedelta, timezone
import logging
from typing import Any

from alexapy import (
    AlexapyConnectionError,
    AlexapyLoginCloseRequested,
    AlexapyLoginError,
)
from alexapy.errors import AlexapyTooManyRequestsError
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
    UpdateFailed,
)

from .alexa_entity import get_entity_data
from .helpers import report_relogin_required

_LOGGER = logging.getLogger(__name__)
ECHO_SENSOR_SCAN_INTERVAL = timedelta(seconds=60)
ECHO_SENSOR_MAX_RETRY_INTERVAL = timedelta(minutes=10)
ECHO_SENSOR_REQUEST_TIMEOUT = 10


def get_echo_coordinator(hass, account_dict: dict):
    """Reuse one lightweight native-sensor poller per Alexa account."""
    coordinator = account_dict.get("echo_sensor_coordinator")
    if coordinator is None:
        coordinator = EchoSensorCoordinator(hass, account_dict)
        account_dict["echo_sensor_coordinator"] = coordinator
    return coordinator


class EchoSensorCoordinator(DataUpdateCoordinator):
    """Poll only enabled native Echo sensors without refreshing the account.

    Motion and light do not receive the media-player push events that slow the
    main coordinator to ten-minute polling. Batch their active endpoint contexts
    into one Phoenix read per minute using the integration's existing session.
    The standard coordinator listener lifecycle stops polling when no enabled
    entities subscribe, and the integration shuts it down before closing login.
    Failed requests back off to ten minutes until a usable response is received.
    """

    def __init__(self, hass, account_dict: dict) -> None:
        """Seed sensor states from network discovery without an extra request."""
        super().__init__(
            hass,
            _LOGGER,
            name="alexa_media_echo_sensors",
            config_entry=account_dict.get("config_entry"),
            update_interval=ECHO_SENSOR_SCAN_INTERVAL,
        )
        self._account_dict = account_dict
        self._poll_lock = asyncio.Lock()
        self._poll_task = None
        self.last_update = None
        main = account_dict["coordinator"]
        self.data = dict(main.data) if isinstance(main.data, dict) else {}
        self.last_update_success = main.last_update_success

    async def _async_update_data(self) -> dict:
        """Read active sensor endpoints in one bounded request."""
        async with self._poll_lock:
            if self._shutdown_requested:
                return {}
            endpoints = sorted(set(self.async_contexts()))
            if not endpoints:
                return {}
            login = self._account_dict.get("login_obj")
            if (
                login is None
                or login.close_requested
                or login.session.closed
                or not (login.status or {}).get("login_successful")
            ):
                raise UpdateFailed("Alexa session is not available")

            self._poll_task = asyncio.current_task()
            try:
                try:
                    data = await asyncio.wait_for(
                        get_entity_data(login, endpoints),
                        timeout=ECHO_SENSOR_REQUEST_TIMEOUT,
                    )
                except AlexapyLoginCloseRequested as err:
                    raise UpdateFailed("Alexa session is closing") from err
                except AlexapyLoginError as err:
                    report_relogin_required(self.hass, login, login.email)
                    raise UpdateFailed("Alexa authentication failed") from err
                except AlexapyTooManyRequestsError as err:
                    raise UpdateFailed(
                        "Alexa is limiting Echo sensor requests"
                    ) from err
                except AlexapyConnectionError as err:
                    raise UpdateFailed("Unable to read Echo sensors") from err
                if not isinstance(data, dict) or not data:
                    # alexapy can turn an unsuccessful HTTP response into no data.
                    # It must not advance the successful-poll heartbeat.
                    raise UpdateFailed("No Echo sensor data received")
            except Exception:
                # HA schedules the next refresh using this interval. Re-raise to
                # retain its failure handling; cancellation must bypass backoff.
                self.update_interval = min(
                    (self.update_interval or ECHO_SENSOR_SCAN_INTERVAL) * 2,
                    ECHO_SENSOR_MAX_RETRY_INTERVAL,
                )
                raise
            else:
                self.update_interval = ECHO_SENSOR_SCAN_INTERVAL
                self.last_update = datetime.now(timezone.utc)
                return data
            finally:
                self._poll_task = None

    async def async_shutdown(self) -> None:
        """Stop scheduled and in-flight reads before the account is unloaded."""
        await super().async_shutdown()
        task = self._poll_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


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


class EchoSensor(CoordinatorEntity):
    """A native sensor read through its account's shared lightweight poller."""

    _attr_has_entity_name = True

    def __init__(self, coordinator, entity_id: str, client, suffix: str) -> None:
        """Attach to the existing Echo without creating a second device."""
        super().__init__(coordinator, context=entity_id)
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
