"""Portable IPv4 interfaces, route-source checks and optional neighbor hints.

psutil supplies native OS interface names. Route probes send no UDP payload;
neighbor entries prioritize scanning but never establish telescope identity."""
import ipaddress
import os
import re
import socket
import subprocess
import sys

import psutil


def network_interfaces():
    """Return configured IPv4 addresses and inactive adapters using native OS names."""
    statistics = psutil.net_if_stats()
    rows = []
    for name, addresses in psutil.net_if_addrs().items():
        ipv4 = [entry for entry in addresses if entry.family == socket.AF_INET]
        mac = next((entry.address for entry in addresses if entry.family == psutil.AF_LINK), None)
        stat = statistics.get(name)
        for entry in ipv4 or [None]:
            address = entry.address if entry else '—'
            loopback = entry is not None and ipaddress.ip_address(address).is_loopback
            rows.append({'interface': name, 'name': name, 'ip': address,
                         'netmask': entry.netmask if entry else None,
                         'mac': mac, 'state': 'loopback' if loopback else
                         'active' if stat and stat.isup else 'inactive'})
    return rows


def route_source_address(host, port=2000):
    """Ask the OS for its selected IPv4 source; UDP connect sends no datagram."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect((host, port))
        return probe.getsockname()[0]


def directly_attached(host):
    """Require a route source and interface mask covering this destination."""
    address = ipaddress.ip_address(host)
    source = route_source_address(host)
    return any(row['ip'] == source and row.get('netmask') and row['state'] == 'active'
               and address in ipaddress.ip_network(f'{source}/{row["netmask"]}', strict=False)
               for row in network_interfaces())


def connected_lans():
    """Bound automatic discovery to attached private /24 or smaller IPv4 LANs."""
    candidates, seen = [], set()
    for row in network_interfaces():
        if row['state'] != 'active' or not row.get('netmask') or row['ip'] == '—':
            continue
        network = ipaddress.ip_network(f'{row["ip"]}/{row["netmask"]}', strict=False)
        direct = network == ipaddress.ip_network('1.2.3.0/24')
        if (network.version != 4 or network.prefixlen < 24 or network.is_link_local
                or network.is_loopback or (not network.is_private and not direct)):
            continue
        if network not in seen:
            candidates.append((row['interface'], row['ip'], network))
            seen.add(network)
    return candidates


def parse_neighbors(output):
    """Parse common arp/ip-neigh formats. Neighbor hints never prove identity."""
    entries = []
    for line in output.splitlines():
        ip = re.search(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])', line)
        mac = re.search(r'(?i)\b(?:[0-9a-f]{1,2}[:-]){5}[0-9a-f]{1,2}\b', line)
        if not ip or not mac:
            continue
        try:
            ipaddress.IPv4Address(ip.group())
        except ValueError:
            continue
        interface = re.search(r'\b(?:on|dev)\s+(\S+)', line)
        entries.append({'ip': ip.group(), 'mac': ':'.join(part.zfill(2) for part in
                        mac.group().lower().replace('-', ':').split(':')),
                        'interface': interface.group(1) if interface else ''})
    return entries


def neighbor_entries():
    """Read optional OS neighbor hints; absence is not a discovery failure."""
    commands = [['arp', '-a']] if os.name == 'nt' else [['ip', 'neigh'], ['arp', '-an']]
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5,
                                    errors='replace')
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            rows = parse_neighbors(result.stdout)
            if rows:
                return rows
    return []


def open_directory(path):
    """Use the host's file browser without invoking a shell."""
    path = os.path.abspath(path)
    if os.name == 'nt':
        os.startfile(path)
    else:
        subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', path])
