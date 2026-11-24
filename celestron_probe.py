#!/usr/bin/env python3
"""Read-only AUX framing, discovery and diagnostic recording.

Only whitelisted, destination-specific read queries are allowed. psutil supplies portable
network inventory; runtime reports use a writable user data directory."""
import argparse
import os
import concurrent.futures
import datetime as dt
import ipaddress
import json
from pathlib import Path
import socket
import subprocess
import time
import uuid
from network_info import network_interfaces, route_source_address, neighbor_entries, parse_neighbors

ROOT = Path(os.environ.get('TELESCOPE_CONTROL_DATA_DIR', os.environ.get(
    'COSMOS90GT_DATA_DIR', str(Path.home() / '.telescope-control')))).expanduser()
NODES = {0x01: 'Main board', 0x04: 'Hand controller', 0x0d: 'Hand controller plus',
         0x10: 'Azimuth motor', 0x11: 'Elevation motor', 0x12: 'Focuser',
         0xb0: 'GPS', 0xb2: 'RTC (model-dependent)', 0xb5: 'Wi-Fi adapter', 0xb6: 'Battery',
         0xb7: 'Charger', 0xbf: 'Light'}
PORTS = (23, 80, 443, 2000, 3000, 4030, 8080)


def run(*args):
    """Run a bounded OS helper and return output or a captured execution error."""
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return {'argv': list(args), 'exit_code': p.returncode,
                'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'argv': list(args), 'error': str(e)}


class Recorder:
    def __init__(self, kind):
        """Initialize Recorder state; external work starts only in the corresponding operation."""
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.directory = ROOT / 'logs' / (stamp + '-' + kind + '-' + uuid.uuid4().hex[:6])
        self.directory.mkdir(parents=True)
        self.events = (self.directory / 'events.jsonl').open('a')

    def record(self, kind, **fields):
        """Append a timestamped event to the operation log."""
        entry = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                 'monotonic': time.monotonic(), 'event': kind, **fields}
        self.events.write(json.dumps(entry) + '\n')
        self.events.flush()

    def finish(self, report):
        """Save the final operation report and close its event log."""
        (self.directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        self.events.close()
        if os.environ.get('COSMOS90GT_QUIET_REPORTS') != '1':
            print(json.dumps(report, indent=2))
            print('Saved:', self.directory)


def packet(destination, command):
    # Command IDs are destination-specific: GPS GET_YEAR 0x04 must never
    # become motor SET_POSITION 0x04. No payload or write command is accepted.
    """Build a checksummed AUX request permitted by this module."""
    allowed = ({0x01, 0x02, 0x03, 0x04, 0x33, 0x36, 0x37, 0x07, 0x08, 0x55, 0xa0, 0xfe}
               if destination == 0xb0 else {0x03, 0x04, 0x33, 0xfe}
               if destination == 0xb2 else ({0x01, 0x13, 0x15, 0x40, 0x41, 0x47, 0xfc, 0xfe}
                     | ({0x18, 0x3b, 0x3c} if destination == 0x10 else {0x12}))
               if destination in (0x10, 0x11) else {0xfe} if destination in NODES else set())
    if command not in allowed:
        raise ValueError('Probe only permits read commands')
    body = bytes((3, 0x20, destination, command))
    return b'\x3b' + body + bytes((-sum(body) & 255,))


class FrameParser:
    def __init__(self):
        """Initialize FrameParser state; external work starts only in the corresponding operation."""
        self.buffer = bytearray()

    def feed(self, data):
        """Buffer partial bytes, validate AUX checksums and return complete frames."""
        self.buffer.extend(data)
        frames = []
        while len(self.buffer) >= 2:
            if self.buffer[0] != 0x3b or not 3 <= self.buffer[1] <= 64:
                del self.buffer[0]
                continue
            size = self.buffer[1] + 3
            if len(self.buffer) < size:
                break
            candidate = bytes(self.buffer[:size])
            if sum(candidate[1:]) & 255:
                del self.buffer[0]
                continue
            del self.buffer[:size]
            frames.append(candidate)
        return frames


class AuxProbe:
    def __init__(self, host, port, recorder):
        """Initialize AuxProbe state; external work starts only in the corresponding operation."""
        self.recorder = recorder
        self.parser = FrameParser()
        self.sock = socket.create_connection((host, port), 3)
        self.sock.settimeout(.2)
        recorder.record('connected', host=host, port=port)

    def receive(self, seconds, axis=None, command=None):
        """Read available socket data within the requested time window."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                data = self.sock.recv(4096)
            except socket.timeout:
                continue
            if not data:
                raise ConnectionError('Peer closed TCP stream')
            self.recorder.record('rx', hex=data.hex(), ascii=data.decode('ascii', 'backslashreplace'))
            for frame in self.parser.feed(data):
                self.recorder.record('valid_aux_frame', hex=frame.hex())
                if frame[2] == axis and frame[3] == 0x20 and frame[4] == command:
                    return frame[5:-1]
        return None

    def query(self, axis, command, attempts=2):
        """Send a read-only AUX query and await the matching controller reply."""
        for attempt in range(attempts):
            # Drain existing stream before sending so an old reply isn't accepted.
            self.receive(.25)
            data = packet(axis, command)
            self.recorder.record('tx_read_only', hex=data.hex(), attempt=attempt + 1)
            self.sock.sendall(data)
            reply = self.receive(1.2, axis, command)
            if reply is not None:
                return reply
        return None

    def close(self):
        """Release this object’s resources and complete its shutdown path."""
        self.sock.close()
        self.recorder.record('disconnected')


def diagnose(args, recorder):
    """Collect read-only controller diagnostics through a recorded probe session."""
    report = {'host': args.host, 'port': args.port, 'read_only': True, 'nodes': []}
    probe = None
    try:
        probe = AuxProbe(args.host, args.port, recorder)
        probe.receive(.6)  # Record any WiFly greeting separately.
        nodes = NODES if args.map_aux else {k: NODES[k] for k in (0x10, 0x11, 0xb5)}
        for axis, name in nodes.items():
            entry = {'address': f'0x{axis:02x}', 'role': name}
            reply = probe.query(axis, 0xfe)
            entry['firmware_reply_hex'] = None if reply is None else reply.hex()
            entry['responded'] = reply is not None
            if axis in (0x10, 0x11):
                reply = probe.query(axis, 0x01)
                entry['position_reply_hex'] = None if reply is None else reply.hex()
                if reply is not None and len(reply) == 3:
                    entry['encoder'] = int.from_bytes(reply, 'big')
                    entry['degrees_raw'] = entry['encoder'] * 360 / (1 << 24)
            report['nodes'].append(entry)
            print(f"{name}: firmware reply {entry['firmware_reply_hex']}", flush=True)
    except (OSError, ConnectionError) as e:
        report['error'] = str(e)
    finally:
        if probe:
            probe.close()
    report['caution'] = 'No reply does not establish absence. AUX roles are protocol addresses, not model identification.'
    return report


def ping(host):
    """Check host reachability with host-appropriate ping options and a timeout."""
    try:
        command = ['ping', '-n', '1', '-w', '1000', host] if os.name == 'nt' else ['ping', '-c', '1', host]
        p = subprocess.run(command,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1.5)
        return host, p.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return host, False


def check_port(item):
    """Test a TCP endpoint and report whether a connection can be opened."""
    host, port = item
    try:
        with socket.create_connection((host, port), .6):
            return host, port, True
    except OSError:
        return host, port, False


def arp_entries(output):
    """Normalize neighbor output into IPv4 and MAC records."""
    return parse_neighbors(output)


def discover(args, recorder):
    """Inspect supported attached networks and candidate TCP endpoints without movement."""
    network = ipaddress.ip_network(args.subnet)
    direct = network == ipaddress.ip_network('1.2.3.0/24')
    if network.version != 4 or network.prefixlen < 24 or (not network.is_private and not direct):
        raise ValueError('Discovery requires an attached private IPv4 /24 or smaller')
    rows = [row for row in network_interfaces() if row['state'] == 'active' and row.get('netmask')
            and row['ip'] != '—' and ipaddress.ip_network(f'{row["ip"]}/{row["netmask"]}', strict=False) == network
            and (not args.interface or row['interface'] == args.interface)]
    source = route_source_address(str(network.network_address + 4))
    row = next((row for row in rows if row['ip'] == source), None)
    if row is None:
        raise ValueError('Requested subnet must match the selected interface and active route')
    address = row['ip']
    args.interface = row['interface']
    report = {'subnet': str(network), 'interface': args.interface, 'local_ip': address, 'devices': []}
    recorder.record('discovery_start', subnet=str(network), interface=args.interface)
    hosts = ['1.2.3.4'] if direct else [str(h) for h in network.hosts() if str(h) != address]
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        replies = dict(executor.map(ping, hosts))
    neighbors = neighbor_entries()
    report['neighbors'] = neighbors
    seen = {e['ip']: e for e in neighbors if (not e['interface'] or e['interface'] == args.interface) and ipaddress.ip_address(e['ip']) in network}
    live = sorted(set(h for h, ok in replies.items() if ok) | set(seen), key=ipaddress.ip_address)
    # One connection per endpoint, with no application payload.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        ports = list(executor.map(check_port, [(h, p) for h in live for p in PORTS]))
    for host in live:
        report['devices'].append({'ip': host, 'mac': seen.get(host, {}).get('mac'),
                                  'ping_reply': replies.get(host, False),
                                  'open_tcp_ports': [p for h, p, ok in ports if h == host and ok]})
    report['other_interface_neighbors'] = [e for e in neighbors if e['interface'] != args.interface and not ipaddress.ip_address(e['ip']).is_multicast]
    report['limitations'] = ['Reachability is not the Wi-Fi association/client list.',
                             'Client isolation and sleeping devices can hide peers.',
                             'Only the listed TCP ports were checked; UDP services were not scanned.',
                             'ARP entries on other interfaces are cached neighbors, not an exhaustive inventory.']
    report['tcp_ports_checked'] = PORTS
    recorder.record('discovery_complete', devices=report['devices'])
    return report


def main():
    """Parse command-line options and run this module’s entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    d = commands.add_parser('discover')
    d.add_argument('--interface', help='Optional native network-interface name')
    d.add_argument('--subnet', default='1.2.3.0/24')
    a = commands.add_parser('diagnose')
    a.add_argument('--host', default='1.2.3.4')
    a.add_argument('--port', type=int, default=2000)
    a.add_argument('--map-aux', action='store_true')
    args = parser.parse_args()
    recorder = Recorder(args.mode)
    try:
        report = discover(args, recorder) if args.mode == 'discover' else diagnose(args, recorder)
    except Exception as e:
        report = {'error': str(e)}
        recorder.record('error', message=str(e))
    recorder.finish(report)
    return 1 if 'error' in report else 0


if __name__ == '__main__':
    raise SystemExit(main())
