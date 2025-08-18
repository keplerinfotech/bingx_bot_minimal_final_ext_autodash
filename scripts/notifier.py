from __future__ import annotations

"""Notifier to send alerts to Slack webhook or via SMTP email.

The module provides two small helpers and a convenience wrapper used by the
kill-switch monitor. Configuration is read from environment variables so no
secrets are stored in the repository.

Environment variables (optional):
    SLACK_WEBHOOK_URL
    ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_SMTP_USER, ALERT_SMTP_PASS, ALERT_TO
"""

import json
import os
import smtplib
import urllib.request
from email.message import EmailMessage


def send_slack(webhook_url: str, text: str, timeout: int = 5) -> bool:
    """Send a simple text message to a Slack Incoming Webhook URL.

    Returns True on success, False on failure.
    """
    try:
        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if hasattr(resp, "getcode"):
                code = resp.getcode()
            else:
                code = getattr(resp, "status", None)
            return code in (200, 204)
    except Exception:
        return False


def send_email(
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    to: str,
    subject: str,
    body: str,
    timeout: int = 10,
) -> bool:
    """Send a plaintext email via SMTP (STARTTLS).

    Returns True on success, False on failure.
    """
    msg = EmailMessage()
    msg["From"] = username
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout) as s:
            s.starttls()
            s.login(username, password)
            s.send_message(msg)
        return True
    except Exception:
        return False


def notify_on_kill(message: str) -> None:
    """Notify via configured channels when a kill-switch triggers.

    This wrapper is intentionally best-effort: failures to notify do not
    raise exceptions since the kill monitor should continue its shutdown
    procedure regardless of notification delivery.
    """
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        try:
            send_slack(webhook, message)
        except Exception:
            pass

    smtp_host = os.environ.get("ALERT_SMTP_HOST")
    smtp_port = int(os.environ.get("ALERT_SMTP_PORT", "587"))
    smtp_user = os.environ.get("ALERT_SMTP_USER")
    smtp_pass = os.environ.get("ALERT_SMTP_PASS")
    alert_to = os.environ.get("ALERT_TO")
    if smtp_host and smtp_user and smtp_pass and alert_to:
        try:
            send_email(
                smtp_host,
                smtp_port,
                smtp_user,
                smtp_pass,
                alert_to,
                "Kill-switch triggered",
                message,
            )
        except Exception:
            pass
