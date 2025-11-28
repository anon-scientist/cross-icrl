import os
import sys
import time
import pickle
import psutil
import pprint
import signal
import argparse

from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--interval', type=float, default=10,                          help='')
parser.add_argument('--chunk',    type=int,   default=1000,                        help='')
parser.add_argument('--cpu',      type=str,   default='no', choices=['yes', 'no'], help='')
parser.add_argument('--mem',      type=str,   default='no', choices=['yes', 'no'], help='')
parser.add_argument('--gpu',      type=str,   default='no', choices=['yes', 'no'], help='')
parser.add_argument('--blk',      type=str,   default='no', choices=['yes', 'no'], help='')
parser.add_argument('--net',      type=str,   default='no', choices=['yes', 'no'], help='')
parser.add_argument('--sens',     type=str,   default='no', choices=['yes', 'no'], help='')
parser.add_argument('--pids',     type=str,   default='no', choices=['yes', 'no'], help='')
args = parser.parse_args()


class Profiler():
    def __init__(self, config):
        self.interval = config.interval
        self.chunk = config.chunk

        self.cpu = True if config.cpu == 'yes' else False
        self.mem = True if config.mem == 'yes' else False
        self.gpu = True if config.gpu == 'yes' else False
        self.blk = True if config.blk == 'yes' else False
        self.net = True if config.net == 'yes' else False
        self.sens = True if config.sens == 'yes' else False
        self.pids = True if config.pids == 'yes' else False

        self.condition = None

        self.uids = os.getresuid()
        self.gids = os.getresgid()

        self.data = []
        self.counter = 0

        self.base_path = os.path.join("./results", 'test', 'stats', Profiler.__name__)

        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def profile_cpu(self, separated=True):
        cpu = psutil.cpu_percent(percpu=separated)
        times = psutil.cpu_times(percpu=separated)
        return {'cpu': cpu, 'times': times}

    def profile_mem(self):
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {'mem': mem, 'swap': swap}

    def profile_gpu(self):
        gpu = None
        return {'gpu': gpu}

    def profile_blk(self, separated=True):
        blk = psutil.disk_io_counters(perdisk=separated)
        return {'blk': blk}

    def profile_net(self, separated=True):
        net = psutil.net_io_counters(pernic=separated)
        return {'net': net}

    def profile_sens(self):
        temps = psutil.sensors_temperatures()
        fans = psutil.sensors_fans()
        return {'temps': temps, 'fans': fans}

    def profile_pids(self, use_uid=True, use_gid=True):
        pids = psutil.process_iter(['uids', 'gids', 'name', 'create_time', 'status', 'cpu_percent', 'cpu_times', 'memory_percent', 'memory_info', 'num_threads', 'num_ctx_switches', 'num_fds', 'io_counters', 'connections'])
        if use_uid: pids = filter(lambda pid: any(uid in pid.uids() for uid in self.uids), pids)
        if use_gid: pids = filter(lambda pid: any(gid in pid.gids() for gid in self.gids), pids)
        return {'pids': list(pids)}

    def dump_profile(self):
        with open(os.path.join(self.base_path, f'dump_{self.counter}.pkl'), 'wb') as fp:
            pickle.dump(self.data, fp)

    def run(self):
        while self.condition:
            last = time.time()

            entry = {'timestamp': datetime.now()}
            if self.cpu: entry.update(self.profile_cpu())
            if self.mem: entry.update(self.profile_mem())
            if self.gpu: entry.update(self.profile_gpu())
            if self.blk: entry.update(self.profile_blk())
            if self.net: entry.update(self.profile_net())
            if self.sens: entry.update(self.profile_sens())
            if self.pids: entry.update(self.profile_pids())
            self.data.append(entry)

            if len(self.data) >= self.chunk:
                self.dump_profile()
                self.data.clear()
                self.counter += 1

            duration = time.time() - last
            if self.interval > duration:
                time.sleep(self.interval - duration)


P = Profiler(args)

def handler(signum, frame):
    P.condition = False
    P.dump_profile()
    sys.exit()

signal.signal(signal.SIGTERM, handler) # default kill
signal.signal(signal.SIGINT, handler) # STRG + C

P.condition = True
P.run()
sys.exit()
