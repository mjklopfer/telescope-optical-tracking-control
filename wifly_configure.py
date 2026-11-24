#!/usr/bin/env python3
"""Native WiFly RN-171 inspection and opt-in infrastructure provisioning.

The CLI previews by default; --apply explicitly permits configuration writes.
Command-mode entry/exit and redaction protect the shared transparent AUX stream."""
import argparse
import getpass
import re
import socket
import time
from celestron_probe import Recorder

PROMPT = re.compile(rb'(?:^|[\r\n])<[^\r\n<>]{1,40}>\s*$')
SECURITY = {'WPA2-Personal (AES)': 4, 'WPA-Personal (TKIP)': 2, 'Open (no password)': 0}


def validate_text(value, label, minimum, maximum):
    """Reject unsupported lengths and non-printable provisioning characters."""
    if not minimum <= len(value) <= maximum or any(not 32 <= ord(c) <= 126 for c in value):
        raise ValueError(f'{label} must be {minimum}-{maximum} printable ASCII characters')


def make_plan(ssid, password, security='WPA2-Personal (AES)'):
    """Build the native WiFly command sequence without transmitting it."""
    validate_text(ssid, 'SSID', 1, 32)
    if security not in SECURITY:
        raise ValueError('Select a supported security mode')
    if SECURITY[security]:
        validate_text(password, 'Wi-Fi passphrase', 8, 63)
    else:
        password = ''
    replacement = next((c for c in '%~^|!' if c not in ssid + password), None)
    if replacement is None:
        raise ValueError('No unambiguous space-encoding character available')
    def encode(s):
        """Replace spaces with the selected unambiguous WiFly escape character."""
        return s.replace(' ', replacement)
    plan = [('set opt replace ' + hex(ord(replacement)), False),
            ('set wlan ssid ' + encode(ssid), False)]
    if SECURITY[security]:
        plan.append(('set wlan phrase ' + encode(password), True))
    return plan + [('set wlan auth ' + str(SECURITY[security]), False),
            ('set wlan channel 0', False),
            ('set wlan join 1', False),
            ('set ip dhcp 1', False)]


def sanitize(text, secrets=()):
    # Process complete responses: secrets may be split across TCP chunks.
    """Redact credentials and protocol password fields from captured text."""
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        text = text.replace(secret, '[REDACTED]')
    return re.sub(r'(?im)^.*(?:passphrase|passkey|password|(?:\bpass\s*=)|(?:wlan\s+phrase)|(?:\b(?:key|phrase)\s*=)).*$',
                  '[credential field redacted]', text)


class Session:
    def __init__(self, sock, recorder, secrets=()):
        """Attach command-mode parsing and redaction to an existing adapter socket."""
        self.sock, self.recorder, self.secrets = sock, recorder, secrets
        self.entered = False

    def receive(self, timeout=3):
        """Read available socket data within the requested time window."""
        data = bytearray()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                part = self.sock.recv(4096)
            except socket.timeout:
                continue
            if not part:
                self.last_reply = data.decode('ascii', 'backslashreplace')
                if data:
                    self.recorder.record('rx_sanitized', text=sanitize(self.last_reply, self.secrets))
                raise ConnectionError('Adapter closed connection')
            data.extend(part)
            if len(data) > 65536:
                raise RuntimeError('Unexpectedly large adapter reply')
            if PROMPT.search(data):
                break
        text = data.decode('ascii', 'backslashreplace')
        self.last_reply = text
        self.recorder.record('rx_sanitized', text=sanitize(text, self.secrets))
        return text

    def enter(self):
        """Enter native WiFly command mode and verify its prompt."""
        self.receive(.6)
        time.sleep(.5)
        self.recorder.record('command_mode_escape')
        self.sock.sendall(b'$$$')
        reply = self.receive(2)
        if not re.search(r'\bCMD\b', reply):
            raise RuntimeError('Command mode not confirmed; no configuration commands sent')
        self.entered = True

    def command(self, command, secret=False, write=False):
        """Send a native command and collect its bounded response."""
        self.last_reply = ''
        self.recorder.record('tx', command='set wlan phrase [REDACTED]' if secret else sanitize(command, self.secrets))
        self.sock.sendall(command.encode('ascii') + b'\r')
        reply = self.receive()
        # A returned status such as Time=FAIL is data, not a rejected command.
        if re.search(r'^\s*(?:ERR|ERROR|FAIL)\b', reply, re.I | re.M):
            raise RuntimeError('Adapter rejected ' + ('credential command' if secret else command))
        if not PROMPT.search(reply.encode('ascii', 'backslashreplace')):
            raise RuntimeError('No complete prompt after ' + ('credential command' if secret else command))
        if write and not re.search(r'(?im)^\s*(?:AOK|OK)\s*$', reply):
            raise RuntimeError('No write acknowledgement for ' + ('credential command' if secret else command))
        return reply

    def exit(self):
        """Try to return the adapter to transparent AUX data mode."""
        if self.entered:
            reply = self._exit()
            confirmed = bool(re.search(r'\bEXIT\b', reply))
            if confirmed:
                self.entered = False
            return confirmed
        return None

    def _exit(self):
        """Perform one command-mode exit attempt and confirm the response."""
        self.recorder.record('tx', command='exit')
        self.sock.sendall(b'exit\r')
        return self.receive(1)


def configure(session, ssid, password, report, security='WPA2-Personal (AES)'):
    """Apply, read back and save the provisioning plan while restoring space encoding."""
    plan = make_plan(ssid, password, security)
    original = re.search(r'(?im)^Replace=(0x[0-9a-f]+)\s*$', report['before']['get option'])
    if not original:
        raise RuntimeError('Cannot read original space-encoding setting; no writes sent')
    report['write_plan'] = ['set wlan phrase [REDACTED]' if secret else c for c, secret in plan]
    report['applied_commands'] = []
    for command, secret in plan:
        report['ram_changes_attempted'] = True
        session.command(command, secret=secret, write=True)
        report['applied_commands'].append('set wlan phrase [REDACTED]' if secret else command)
    restore = 'set opt replace ' + original.group(1)
    session.command(restore, write=True)
    report['applied_commands'].append(restore)
    wlan = session.command('get wlan')
    ip = session.command('get ip')
    report['readback'] = {'wlan': sanitize(wlan, session.secrets), 'ip': sanitize(ip, session.secrets)}
    match = re.search(r'(?im)^SSID=(.*)\r?$', wlan)
    if not match or match.group(1).rstrip('\r') != ssid:
        raise RuntimeError('SSID readback mismatch; configuration not saved')
    if not re.search(r'(?im)^Join=0?1\s*$', wlan):
        raise RuntimeError('Auto-join readback mismatch; configuration not saved')
    if not re.search(r'(?im)^DHCP=ON\s*$', ip):
        raise RuntimeError('DHCP client readback mismatch; configuration not saved')
    expected_auth = {'WPA2-Personal (AES)': ('WPA2',), 'WPA-Personal (TKIP)': ('WPA1', 'WPA'),
                     'Open (no password)': ('OPEN',)}[security]
    auth = re.search(r'(?im)^Auth=([^\r\n]+)', wlan)
    if not auth or auth.group(1).strip().upper() not in expected_auth:
        raise RuntimeError('Security readback mismatch; configuration not saved')
    report['save_attempted'] = True
    reply = session.command('save')
    if not re.search(r'(?i)(storing.*config|\bAOK\b|\bOK\b)', reply):
        raise RuntimeError('Save acknowledgement missing; persistence is uncertain')
    report['saved'] = True


def main():
    """Parse command-line options and run this module’s entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssid')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    if args.apply and not args.ssid:
        parser.error('--apply requires --ssid')
    password = ''
    plan = []
    if args.ssid:
        validate_text(args.ssid, 'SSID', 1, 32)
    if args.apply:
        password = getpass.getpass('Router WPA2 password (not saved in logs): ')
        plan = make_plan(args.ssid, password)
    elif not args.inspect:
        print('Preview only. Use --inspect to read the adapter; --apply --ssid NAME to configure.')
        print('Plan: enter $$$ command mode; set space encoding; router SSID/passphrase; WPA2;')
        print('scan channels; auto-join; DHCP client; verify readback; save; exit. No reboot or motor commands.')
        return 0
    recorder = Recorder('wifly-configure' if args.apply else 'wifly-inspect')
    secrets = [password]
    secrets += [c.split(' ', 3)[-1] for c, secret in plan if secret]
    report = {'host': '1.2.3.4', 'port': 2000, 'target_ssid': args.ssid,
              'apply_requested': args.apply, 'saved': False, 'reboot_sent': False,
              'motor_commands_sent': False}
    session = None
    sock = None
    try:
        from network_info import directly_attached
        if not directly_attached('1.2.3.4'):
            raise RuntimeError('Connect this computer to the telescope SkyQLink network first')
        sock = socket.create_connection(('1.2.3.4', 2000), 3)
        sock.settimeout(.2)
        session = Session(sock, recorder, secrets)
        session.enter()
        report['before'] = {c: sanitize(session.command(c), secrets) for c in ('ver', 'get wlan', 'get ip', 'get uart', 'get option')}
        if not re.search(r'\b2\.40-CEL\b.*\bRN-171\b', report['before']['ver']):
            raise RuntimeError('Adapter is not the identified 2.40-CEL RN-171; no writes sent')
        if args.apply:
            configure(session, args.ssid, password, report)
    except (Exception, KeyboardInterrupt) as e:
        report['error'] = sanitize(str(e) or type(e).__name__, secrets)
        if report.get('ram_changes_attempted') and not report['saved']:
            report['recovery_note'] = 'Partial RAM settings may exist. No successful save confirmed. Do not assume settings were rolled back.'
    finally:
        if session:
            try:
                report['exit_confirmed'] = session.exit()
            except Exception as e:
                report['exit_error'] = sanitize(str(e), secrets)
        if sock:
            sock.close()
        recorder.finish(report)
    return 1 if report.get('error') else 0


if __name__ == '__main__':
    raise SystemExit(main())
