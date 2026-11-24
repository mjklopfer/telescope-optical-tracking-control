"""Authenticated loopback target stream with simulation and a gated experimental backend.

Simulation mode never connects to hardware. The optional controller forwards to
the separately armed, experimental native-target backend. MCP supervises it; an
external producer POSTs JSON to /target on the returned loopback URL using
Authorization: Bearer <token from the returned token_file>. Packet format:
  {"sequence": 1, "utc": <Unix UTC seconds>, "azimuth_deg": 123.4,
   "elevation_deg": 35.6}
Send current targets at 10-20 Hz initially. UTC clocks must agree. Out-of-order,
stale, nonfinite, over-rate or out-of-envelope input is rejected. No AI/MCP call
belongs in the time-critical update loop. The 50 Hz loop here is a scheduling
target, NOT a hard real-time timing guarantee on a desktop OS.

Only localhost accepts connections. For remote producers a separately reviewed
authenticated transport/adapter is needed; do not expose this server to a LAN.
Orbitron-specific integration/orbit propagation is not implemented. An adapter
must convert its output into this schema and supply current observing-site data.

Hardware tracking is unverified and disabled by default: validate AUX rate/GOTO semantics,
both-axis concurrent actuation, Stop on communication loss, mechanical envelopes,
encoder freshness and end-to-end latency first. Existing timed jog calibration
is not sufficient to claim closed-loop satellite/launch tracking capability.
"""
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import math
import secrets
import threading
import time
from celestron_probe import ROOT


def az_delta(target, current):
    return (target - current + 180) % 360 - 180


class TrackingStream:
    def __init__(self, min_elevation=0.0, max_elevation=80.0,
                 max_az_rate=2.5, max_el_rate=2.5, stale_seconds=.5, controller=None):
        limits = (min_elevation, max_elevation, max_az_rate, max_el_rate, stale_seconds)
        if (not all(math.isfinite(x) for x in limits)
                or not -90 <= min_elevation < max_elevation <= 90
                or not 0 < max_az_rate <= 3 or not 0 < max_el_rate <= 3
                or not .1 <= stale_seconds <= 2):
            raise ValueError('Invalid simulation limits')
        self.controller = controller
        self.input_fault = False
        self.min_el, self.max_el = min_elevation, max_elevation
        self.max_rates = (max_az_rate, max_el_rate)
        self.stale_seconds = stale_seconds
        self.samples = deque(maxlen=2)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.position = [0.0, (min_elevation + max_elevation) / 2]
        self.rates = [0.0, 0.0]
        self.state = 'WAITING_FOR_STREAM'
        self.rejected = 0
        self.accepted = 0
        self.recoveries = 0
        self.link_down_until = 0.0
        self.last_received = 0.0
        self.server = None
        self.thread = None
        self.token = secrets.token_urlsafe(32)
        self.token_file = None

    def submit(self, sample, now=None):
        now = time.time() if now is None else now
        try:
            seq = sample['sequence']
            utc, az, el = (float(sample[k]) for k in ('utc', 'azimuth_deg', 'elevation_deg'))
            if (type(seq) is not int or seq < 0 or not all(math.isfinite(x) for x in (utc, az, el))
                    or not 0 <= az < 360 or not self.min_el <= el <= self.max_el
                    or not -.2 <= now - utc <= self.stale_seconds):
                raise ValueError('Invalid, stale, future or out-of-envelope target')
            with self.lock:
                if self.samples:
                    old = self.samples[-1]
                    if seq <= old['sequence'] or utc <= old['utc']:
                        raise ValueError('Target sequence/time must increase')
                    elapsed = utc - old['utc']
                    velocities = (abs(az_delta(az, old['azimuth_deg'])) / elapsed,
                                  abs(el - old['elevation_deg']) / elapsed)
                    if any(v > cap for v, cap in zip(velocities, self.max_rates, strict=True)):
                        raise ValueError('Requested trajectory exceeds configured slew rate')
                if self.controller:
                    self.controller.submit(az, el, seq, utc, ttl=self.stale_seconds)
                self.input_fault = False
                self.samples.append(dict(sequence=seq, utc=utc, azimuth_deg=az, elevation_deg=el))
                self.last_received = time.monotonic()
                self.accepted += 1
        except (ValueError, KeyError, TypeError) as error:
            with self.lock:
                self.rejected += 1
                self.input_fault = True
                if self.controller:
                    self.controller.pause()
                self.rates = [0.0, 0.0]
                self.state = 'REJECTED_TARGET'
            raise ValueError(str(error)) from error

    def step(self, elapsed, now=None):
        """Simulate bounded simultaneous-axis control; no network motor commands."""
        now = time.time() if now is None else now
        with self.lock:
            if self.controller or self.input_fault:
                return
            if self.link_down_until > time.monotonic():
                self.state = 'RECOVERING_SIMULATED_LINK'
                self.rates = [0.0, 0.0]
                return
            if self.link_down_until:
                self.link_down_until = 0
                self.recoveries += 1
            if not self.samples or now - self.samples[-1]['utc'] > self.stale_seconds:
                self.state = 'STALE_STREAM_STOPPED' if self.samples else 'WAITING_FOR_STREAM'
                self.rates = [0.0, 0.0]
                return
            last = self.samples[-1]
            target = [last['azimuth_deg'], last['elevation_deg']]
            velocity = [0.0, 0.0]
            if len(self.samples) == 2:
                first = self.samples[0]
                delta_t = last['utc'] - first['utc']
                velocity = [az_delta(target[0], first['azimuth_deg']) / delta_t,
                            (target[1] - first['elevation_deg']) / delta_t]
                prediction = max(0, min(.2, now - last['utc']))
                target = [(target[0] + velocity[0] * prediction) % 360,
                          max(self.min_el, min(self.max_el, target[1] + velocity[1] * prediction))]
            errors = [az_delta(target[0], self.position[0]), target[1] - self.position[1]]
            self.state = 'SIMULATED_CATCH_UP' if max(abs(x) for x in errors) > .2 else 'SIMULATED_TRACKING'
            self.rates = [max(-cap, min(cap, speed + 2 * error))
                          for cap, speed, error in zip(self.max_rates, velocity, errors, strict=True)]
            elapsed = max(0, min(.1, elapsed))  # never leap over a missed scheduler interval
            self.position[0] = (self.position[0] + self.rates[0] * elapsed) % 360
            self.position[1] = max(self.min_el, min(self.max_el, self.position[1] + self.rates[1] * elapsed))

    def status(self):
        with self.lock:
            if self.controller:
                return {'mode': 'experimental_hardware', 'controller': self.controller.status(),
                        'input_fault': self.input_fault, 'accepted_samples': self.accepted, 'rejected_samples': self.rejected}
            return {'mode': 'simulation_only', 'hardware_motion_enabled': False,
                    'state': self.state, 'input_live': bool(self.samples and time.time() - self.samples[-1]['utc'] <= self.stale_seconds),
                    'simulated_position': list(self.position), 'simulated_rates_deg_s': list(self.rates),
                    'accepted_samples': self.accepted, 'rejected_samples': self.rejected,
                    'simulated_recoveries': self.recoveries,
                    'latest_target': dict(self.samples[-1]) if self.samples else None}

    def start(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # no tokens or target streams in HTTP access logs

            def do_POST(self):
                auth = self.headers.get('Authorization', '')
                if not hmac.compare_digest(auth, 'Bearer ' + owner.token):
                    self.send_error(401)
                    return
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if self.path != '/target' or not 0 < size <= 4096:
                        raise ValueError('Use POST /target with one JSON target, max 4096 bytes')
                    owner.submit(json.loads(self.rfile.read(size)))
                    payload, code = {'accepted': True}, 200
                except (ValueError, TypeError) as error:
                    payload, code = {'error': str(error)}, 400
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        port = self.server.server_port
        ROOT.mkdir(parents=True, exist_ok=True)
        self.token_file = ROOT / f'tracking-stream-token-{port}.txt'
        self.token_file.write_text(self.token, encoding='utf-8')
        self.token_file.chmod(0o600)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        def run():
            previous = time.monotonic()
            while not self.stop_event.wait(.02):
                current = time.monotonic()
                self.step(current - previous)
                previous = current
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return {'url': f'http://127.0.0.1:{port}/target', 'token_file': str(self.token_file),
                'mode': 'experimental_hardware' if self.controller else 'simulation_only', 'hardware_motion_enabled': bool(self.controller and self.controller.armed)}

    def stop(self):
        if self.controller:
            self.controller.pause()
        self.stop_event.set()
        if self.thread:
            self.thread.join(2)
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.token_file:
            self.token_file.unlink(missing_ok=True)
        with self.lock:
            self.state = 'STOPPED'
            self.rates = [0.0, 0.0]
