"""Exclusive, non-GUI WiFly provisioning shared by the desktop interface.

Read settings before writes, restrict provisioning to the supported RN-171 build,
and redact known secrets before callbacks and saved reports."""
import ipaddress
import re
import socket
from celestron_probe import Recorder
from network_info import directly_attached
from wifly_configure import Session, configure, make_plan, sanitize, SECURITY
from endpoint_lock import exclusive_endpoint


def validate_endpoint(host, port):
    """Validate a telescope IPv4 destination and its TCP port."""
    address = ipaddress.ip_address(host.strip())
    if address.is_multicast or address.is_unspecified or address.is_loopback:
        raise ValueError('Enter the telescope IP address')
    return str(address), validate_port(port)


def validate_port(port):
    """Reject invalid port inputs and return the numeric TCP port."""
    if isinstance(port, bool) or not str(port).isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError('Port must be between 1 and 65535')
    return int(port)


def clean(value, secrets):
    """Recursively redact known secrets before presenting or storing results."""
    if isinstance(value, str):
        return sanitize(value, secrets)
    if isinstance(value, dict):
        return {k: clean(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v, secrets) for v in value]
    return value


class LiveRecorder(Recorder):
    def __init__(self, callback, secrets):
        """Create the log and attach redacted GUI-event delivery."""
        super().__init__('setup-app')
        self.callback, self.secrets = callback, secrets

    def record(self, kind, **fields):
        """Append a timestamped event to the operation log."""
        fields = clean(fields, self.secrets)
        super().record(kind, **fields)
        self.callback({'event': kind, **fields})

    def finish(self, report):
        """Save the final operation report and close its event log."""
        super().finish(clean(report, self.secrets))


def check_direct_route(host):
    # 1.2.3.x is publicly allocated. Only use it when directly attached locally.
    """Require a directly attached network route for the public 1.2.3.x range."""
    if ipaddress.ip_address(host) not in ipaddress.ip_network('1.2.3.0/24'):
        return
    if not directly_attached(host):
        raise ValueError('Connect to the telescope’s SkyQLink network to use 1.2.3.4, or enter its house-network IP address.')


@exclusive_endpoint
def operate(host, port, ssid='', password='', security='WPA2-Personal (AES)',
            apply=False, callback=lambda event: None):
    """Read adapter settings and optionally verify and save an explicitly requested provisioning change."""
    host, port = validate_endpoint(host, port)
    if security not in SECURITY:
        raise ValueError('Select a supported security mode')
    if SECURITY[security] == 0:
        password = ''
    plan = make_plan(ssid, password, security) if apply else []
    secrets = [password] + [c.split(' ', 3)[-1] for c, secret in plan if secret]
    recorder = LiveRecorder(callback, secrets)
    report = {'host': host, 'port': port, 'target_ssid': ssid if apply else None,
              'security': security if apply else None, 'apply_requested': apply,
              'saved': False, 'reboot_sent': False, 'motor_commands_sent': False,
              'router_connection_verified': False}
    sock = session = None
    try:
        check_direct_route(host)
        recorder.record('stage', message=f'Connecting to {host}:{port}')
        sock = socket.create_connection((host, port), 3)
        sock.settimeout(.2)
        session = Session(sock, recorder, secrets)
        recorder.record('stage', message='Entering WiFly command mode')
        session.enter()
        report['before'] = {c: clean(session.command(c), secrets)
                            for c in ('ver', 'get wlan', 'get ip', 'get uart', 'get option')}
        if not re.search(r'\b2\.40-CEL\b.*\bRN-171\b', report['before']['ver']):
            raise RuntimeError('This tool supports the identified RN-171 / 2.40-CEL firmware only; no writes sent.')
        recorder.record('stage', message='RN-171 / WiFly 2.40-CEL identified')
        if apply:
            recorder.record('stage', message='Uploading router settings, then verifying and saving')
            configure(session, ssid, password, report, security)
            recorder.record('stage', message='Save acknowledged. Router connection requires a restart.')
    except Exception as e:
        report['error'] = sanitize(str(e), secrets)
        if report.get('ram_changes_attempted') and not report['saved']:
            report['recovery_note'] = 'Partial settings may remain in RAM. Rollback was not performed.'
        recorder.record('failure', message=report['error'])
    finally:
        if session:
            try:
                report['exit_confirmed'] = session.exit()
            except Exception as e:
                report['exit_error'] = sanitize(str(e), secrets)
        if sock:
            sock.close()
        report = clean(report, secrets)
        report['log_directory'] = str(recorder.directory)
        recorder.finish(report)
    return report
