"""
Production-ready asynchronous dispersion runner
Addresses all critical and performance issues
"""

import csv
import pickle
import os
import time
import threading
import logging
import psutil
import gc
import signal
import ctypes
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional, Set, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError, Future
from queue import Queue, Empty, PriorityQueue
from collections import defaultdict, deque
import numpy as np
import pandas as pd

from pypws.enums import ResultCode
from py_lopa.calcs import helpers
from py_lopa.calcs.consts import Consts
from py_lopa.data.tables import Tables
from py_lopa.phast_io.phast_prep import prep_state, flash_calc
from py_lopa.calcs.thermo_pio import vpress_pa_and_vapor_phase_comp_and_component_vapor_pressures
from py_lopa.model_interface import Model_Interface

csv_lock = threading.Lock()

@dataclass
class FlashSpec:
    """Specification for a flash calculation"""
    flash_key: str
    material: str
    cas_no: str
    chem_name: str
    press_psig: float
    temp_k: float


@dataclass
class DispersionSpec:
    """Specification for a dispersion calculation"""
    task_id: str
    idx: int
    flash_key: str
    row_data: Dict[str, Any]
    requeue_count: int = 0
    max_retries: int = 3


class MemoryMonitor:
    """Real-time memory monitoring with improved trend analysis"""
    
    def __init__(self, threshold_gb: float, warning_gb: float):
        if warning_gb >= threshold_gb:
            raise ValueError("Warning threshold must be less than hard threshold")
        
        self.threshold = threshold_gb * 1024 * 1024 * 1024
        self.warning_threshold = warning_gb * 1024 * 1024 * 1024
        self.samples = deque(maxlen=120)  # 2 minutes at 1-second samples
        self.lock = threading.Lock()
        
    def check_memory(self) -> Dict[str, Any]:
        """Get current memory status without caching"""
        memory = psutil.virtual_memory()
        used = memory.used
        timestamp = time.time()
        
        with self.lock:
            self.samples.append((timestamp, used))
        
        return {
            'used_gb': used / 1024**3,
            'percent': memory.percent,
            'available_gb': memory.available / 1024**3,
            'under_threshold': used < self.threshold,
            'under_warning': used < self.warning_threshold,
            'trend': self._calculate_trend(),
            'slope': self._calculate_slope()
        }
    
    def _calculate_trend(self) -> str:
        """Calculate memory usage trend using moving averages"""
        if len(self.samples) < 20:
            return "insufficient_data"
        
        samples_list = list(self.samples)
        
        # Use moving averages to smooth out noise
        window_size = min(10, len(samples_list) // 2)
        recent_avg = np.mean([s[1] for s in samples_list[-window_size:]])
        older_avg = np.mean([s[1] for s in samples_list[-2*window_size:-window_size]])
        
        change_pct = (recent_avg - older_avg) / older_avg
        
        if change_pct > 0.05:  # 5% increase
            return "increasing"
        elif change_pct < -0.05:  # 5% decrease
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_slope(self) -> float:
        """Calculate memory usage slope (bytes per second)"""
        if len(self.samples) < 10:
            return 0.0
        
        samples_list = list(self.samples)[-20:]  # Last 20 seconds
        times = np.array([s[0] for s in samples_list])
        values = np.array([s[1] for s in samples_list])
        
        # Linear regression slope
        slope = np.polyfit(times - times[0], values, 1)[0]
        return slope


class SmartTaskScheduler:
    """Intelligent task scheduling with condition variables"""
    
    def __init__(self, max_ready_queue_size: int = 500):
        if max_ready_queue_size <= 0:
            raise ValueError("Max ready queue size must be positive")
        
        self.max_ready_queue_size = max_ready_queue_size
        
        # Task storage
        self.pending_by_flash = defaultdict(list)
        self.ready_queue = PriorityQueue()
        self.all_flash_specs = {}
        self.flash_work_queue = Queue()
        
        # Status tracking
        self.flash_results = {}
        self.flash_failures = set()
        self.flash_in_progress = set()
        
        # Synchronization
        self.lock = threading.RLock()
        self.flash_available = threading.Condition(self.lock)
        self.dispersion_available = threading.Condition(self.lock)
        
        # Statistics
        self.stats = {
            'dispersions_pending': 0,
            'dispersions_ready': 0,
            'dispersions_requeued': 0,
            'dispersions_max_requeues': 0,
            'flash_requests': 0,
            'unique_flash_keys': 0,
            'all_tasks_added': False
        }
    
    def add_flash_spec(self, flash_spec: FlashSpec):
        """Add a flash specification"""
        with self.lock:
            if flash_spec.flash_key not in self.all_flash_specs:
                self.all_flash_specs[flash_spec.flash_key] = flash_spec
                self.flash_work_queue.put(flash_spec)
                self.stats['unique_flash_keys'] += 1
                self.flash_available.notify()
    
    def add_dispersion_spec(self, disp_spec: DispersionSpec):
        """Add dispersion spec - only queue if flash is ready"""
        with self.lock:
            flash_key = disp_spec.flash_key
            
            if flash_key in self.flash_results:
                if self.ready_queue.qsize() < self.max_ready_queue_size:
                    priority = disp_spec.idx
                    self.ready_queue.put((priority, disp_spec))
                    self.stats['dispersions_ready'] += 1
                    self.dispersion_available.notify()
                else:
                    self.pending_by_flash[flash_key].append(disp_spec)
                    self.stats['dispersions_pending'] += 1
            elif flash_key in self.flash_failures:
                return False
            else:
                self.pending_by_flash[flash_key].append(disp_spec)
                self.stats['dispersions_pending'] += 1
                
        return True
    
    def get_next_flash_request(self, timeout: float = 5.0) -> Optional[FlashSpec]:
        """Get next flash calculation with blocking wait"""
        try:
            return self.flash_work_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def complete_flash(self, flash_key: str, result: Any = None, failed: bool = False):
        """Mark flash as complete and promote waiting dispersions"""
        with self.lock:
            self.flash_in_progress.discard(flash_key)
            
            if failed:
                self.flash_failures.add(flash_key)
                skipped_count = len(self.pending_by_flash.pop(flash_key, []))
                self.stats['dispersions_pending'] -= skipped_count
                return skipped_count
            else:
                self.flash_results[flash_key] = result
                
                pending_tasks = self.pending_by_flash.pop(flash_key, [])
                promoted_count = 0
                
                for disp_spec in pending_tasks:
                    if self.ready_queue.qsize() < self.max_ready_queue_size:
                        priority = disp_spec.idx
                        self.ready_queue.put((priority, disp_spec))
                        promoted_count += 1
                    else:
                        self.pending_by_flash[flash_key].append(disp_spec)
                
                self.stats['dispersions_pending'] -= len(pending_tasks)
                self.stats['dispersions_ready'] += promoted_count
                
                if promoted_count > 0:
                    self.dispersion_available.notify_all()
                
                return promoted_count
    
    def get_ready_dispersion(self, timeout: float = 5.0) -> Optional[DispersionSpec]:
        """Get next ready dispersion task with blocking wait"""
        with self.dispersion_available:
            while self.ready_queue.empty() and not self._is_complete():
                self.dispersion_available.wait(timeout=timeout)
                if self.ready_queue.empty():
                    return None
            
            if not self.ready_queue.empty():
                try:
                    priority, disp_spec = self.ready_queue.get_nowait()
                    self.stats['dispersions_ready'] -= 1
                    return disp_spec
                except Empty:
                    pass
        
        return None
    
    def requeue_dispersion(self, disp_spec: DispersionSpec, use_exponential_backoff: bool = True):
        """Requeue a dispersion with exponential backoff"""
        if disp_spec.requeue_count >= disp_spec.max_retries:
            return False
        
        disp_spec.requeue_count += 1
        
        with self.lock:
            self.stats['dispersions_requeued'] += 1
            self.stats['dispersions_max_requeues'] = max(
                self.stats['dispersions_max_requeues'], 
                disp_spec.requeue_count
            )
            
            if disp_spec.flash_key in self.flash_results:
                if use_exponential_backoff:
                    delay_factor = 2 ** disp_spec.requeue_count
                    priority = disp_spec.idx + (delay_factor * 10000)
                else:
                    priority = disp_spec.idx + (disp_spec.requeue_count * 10000)
                
                self.ready_queue.put((priority, disp_spec))
                self.stats['dispersions_ready'] += 1
                self.dispersion_available.notify()
            else:
                self.pending_by_flash[disp_spec.flash_key].append(disp_spec)
                self.stats['dispersions_pending'] += 1
        
        return True
    
    def mark_complete(self):
        """Mark that all tasks have been added"""
        with self.lock:
            self.stats['all_tasks_added'] = True
            self.flash_available.notify_all()
            self.dispersion_available.notify_all()
    
    def _is_complete(self) -> bool:
        """Check if all work is complete (must be called under lock)"""
        return (self.stats['all_tasks_added'] and 
                self.stats['dispersions_pending'] == 0 and 
                self.stats['dispersions_ready'] == 0 and 
                len(self.flash_in_progress) == 0 and
                self.flash_work_queue.empty())


class InterruptibleDispersionCalculator:
    """Wrapper for dispersion calculations with proper thread interruption"""
    
    @staticmethod
    def _thread_id():
        """Get current thread ID"""
        return threading.current_thread().ident
    
    @staticmethod
    def _raise_exception_in_thread(thread_id, exception_type):
        """Raise exception in target thread"""
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread_id), 
            ctypes.py_object(exception_type)
        )
        if ret > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, None)
    
    def __init__(self, cd, cheminfo):
        self.cd = cd
        self.cheminfo = cheminfo
        self.interrupted = threading.Event()
    
    def run_with_interrupt(self, disp_spec: DispersionSpec, flashresult, timeout_seconds: int):
        """Run dispersion calculation with interruptible timeout"""
        result_container = {'result': None, 'error': None, 'thread_id': None}
        
        def calculation_thread():
            try:
                result_container['thread_id'] = self._thread_id()
                result = self._run_dispersion(disp_spec, flashresult)
                if not self.interrupted.is_set():
                    result_container['result'] = result
            except Exception as e:
                if not self.interrupted.is_set():
                    result_container['error'] = str(e)
        
        # Start calculation thread
        calc_thread = threading.Thread(target=calculation_thread)
        calc_thread.start()
        
        # Wait with timeout
        calc_thread.join(timeout=timeout_seconds)
        
        if calc_thread.is_alive():
            # Force interrupt the calculation
            self.interrupted.set()
            if result_container['thread_id']:
                try:
                    self._raise_exception_in_thread(
                        result_container['thread_id'], 
                        KeyboardInterrupt
                    )
                except:
                    pass
            
            # Give it a moment to clean up
            calc_thread.join(timeout=2)
            
            raise TimeoutError(f"Calculation timed out after {timeout_seconds} seconds")
        
        if result_container['error']:
            raise Exception(result_container['error'])
        
        return result_container['result']
    
    def _run_dispersion(self, disp_spec: DispersionSpec, flashresult):
        """Run a single dispersion calculation"""
        row_data = disp_spec.row_data.copy()
        row_data['flashresult'] = flashresult
        row_data['idx'] = disp_spec.idx
        
        m_io = self._get_m_io(row_data)
        
        # Check for interruption periodically
        if self.interrupted.is_set():
            raise KeyboardInterrupt("Calculation interrupted")
        
        results = self._process_m_io(m_io)
        
        if results:
            row_data.update(results)
            return row_data
        
        return None
    
    def _get_m_io(self, row_data):
        """Create Model_Interface object"""
        m_io = Model_Interface()
        m_io.material = row_data['material']
        m_io.flashresult = row_data['flashresult']
        m_io.inputs['chemical_mix'] = [row_data['cas_no']]
        m_io.inputs['composition'] = [1]
        m_io.inputs['pressure_psig'] = row_data['press_psig']
        m_io.inputs['temp_deg_c'] = row_data['temp_k'] - 273.15
        m_io.inputs['storage_mass_kg'] = row_data['release_mass_kg']
        m_io.inputs['release_elevation_m'] = row_data['elev_m']
        m_io.inputs['release_angle_degrees_from_horizontal'] = row_data['orientation']
        m_io.inputs['flash_fire'] = False
        m_io.inputs['inhalation'] = True
        m_io.inputs['max_hole_size_in'] = row_data['hole_size_in']
        m_io.inputs['bldg_hts_low_med_high'] = [0, 0, 0]
        m_io.inputs['use_multicomponent_method'] = False
        return m_io
    
    def _process_m_io(self, m_io):
        """Process model and return results"""
        if self.interrupted.is_set():
            raise KeyboardInterrupt("Calculation interrupted")
        
        res = m_io.run()
        if res != ResultCode.SUCCESS:
            return None
        
        ca_dict = m_io.mc.ca_dict
        output = {}
        
        for rel_dur_sec, ca_data_at_dur in ca_dict.items():
            for wx, ca_at_dur_at_wx in ca_data_at_dur[rel_dur_sec].items():
                for haz, ca_at_dur_at_wx_at_haz in ca_at_dur_at_wx[wx].items():
                    ca = ca_at_dur_at_wx_at_haz[haz]
                    conseq_list = ca.conseq_list
                    for con_dict in conseq_list:
                        if con_dict[self.cd.CAT_TITLE] in [self.cd.CAT_SERIOUS, self.cd.CAT_MODERATE]:
                            output[f'{con_dict[self.cd.CAT_TITLE]}_{self.cd.IMPACT_DISTANCE_M}'] = con_dict[self.cd.IMPACT_DISTANCE_M]
                            output[f'{con_dict[self.cd.CAT_TITLE]}_{self.cd.IMPACT_AREA_M2}'] = con_dict[self.cd.IMPACT_AREA_M2]
        
        return output


class ProductionAsyncDispersionRunner:
    """Production-ready async dispersion runner"""
    
    def __init__(self, flash_workers=6, dispersion_workers=4, 
                 memory_threshold_gb=7.0, memory_warning_gb=6.5,
                 output_file='model_run_output.csv', batch_size=100, 
                 timeout_seconds=900, max_ready_queue_size=500):
        
        # Validation
        if flash_workers <= 0 or dispersion_workers <= 0:
            raise ValueError("Worker counts must be positive")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if memory_warning_gb >= memory_threshold_gb:
            raise ValueError("Warning threshold must be less than hard threshold")
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        
        self.flash_workers = flash_workers
        self.dispersion_workers = dispersion_workers
        self.output_file = output_file
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        
        # Core components
        self.memory_monitor = MemoryMonitor(memory_threshold_gb, memory_warning_gb)
        self.scheduler = SmartTaskScheduler(max_ready_queue_size)
        
        # Thread synchronization
        self.output_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Single executor with exact worker count
        self.dispersion_executor = ThreadPoolExecutor(
            max_workers=dispersion_workers,
            thread_name_prefix="DispCalc"
        )
        
        # Calculator instance
        self.cd = Consts().CONSEQUENCE_DATA
        self.cheminfo = helpers.get_dataframe_from_csv(Tables().CHEM_INFO)
        self.calculator = InterruptibleDispersionCalculator(self.cd, self.cheminfo)
        
        # Output buffering
        self.output_buffer = []
        self.header_written = False
        
        # Statistics
        self.stats = {
            'flash_completed': 0,
            'flash_failed': 0,
            'dispersion_completed': 0,
            'dispersion_failed': 0,
            'dispersion_timeout': 0,
            'dispersion_skipped_flash_failed': 0,
            'dispersion_skipped_memory': 0,
            'memory_warnings': 0,
            'start_time': time.time()
        }
        
        # Setup
        self.logger = self._setup_logging()
        self.resume_from_idx = self._get_starting_idx()
        
        # Load cache
        self.flash_cache_file = 'flash_results_cache.pickle'
        self._load_flash_cache()
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_event.set()
    
    def _setup_logging(self):
        """Setup logging"""
        logger = logging.getLogger('production_dispersion_runner')
        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _get_starting_idx(self):
        """Get starting index"""
        try:
            with open('curr_idx.txt', 'r') as f:
                return int(f.read().strip())
        except:
            return -1
    
    def _load_flash_cache(self):
        """Load flash cache"""
        try:
            with open(self.flash_cache_file, 'rb') as f:
                cache = pickle.load(f)
                
                loaded_results = 0
                loaded_failures = 0
                
                if isinstance(cache, dict):
                    for key, value in cache.items():
                        if value is None:
                            self.scheduler.flash_failures.add(key)
                            loaded_failures += 1
                        else:
                            self.scheduler.flash_results[key] = value
                            loaded_results += 1
                
                self.logger.info(f"Loaded {loaded_results} flash results, {loaded_failures} failures")
        except Exception as e:
            self.logger.info(f"No flash cache loaded: {e}")
    
    def _save_flash_cache(self):
        """Save flash cache"""
        try:
            cache_data = {}
            
            with self.scheduler.lock:
                cache_data.update(self.scheduler.flash_results)
                for key in self.scheduler.flash_failures:
                    cache_data[key] = None
            
            temp_file = self.flash_cache_file + '.tmp'
            with open(temp_file, 'wb') as f:
                pickle.dump(cache_data, f)
            os.replace(temp_file, self.flash_cache_file)
            
            self.logger.info(f"Saved {len(cache_data)} flash cache entries")
        except Exception as e:
            self.logger.error(f"Failed to save flash cache: {e}")
    
    def flash_worker(self):
        """Flash calculation worker"""
        while not self.shutdown_event.is_set():
            try:
                flash_spec = self.scheduler.get_next_flash_request(timeout=5.0)
                if flash_spec is None:
                    continue
                
                with self.scheduler.lock:
                    if (flash_spec.flash_key in self.scheduler.flash_results or 
                        flash_spec.flash_key in self.scheduler.flash_failures):
                        continue
                    self.scheduler.flash_in_progress.add(flash_spec.flash_key)
                
                try:
                    press_pa = (flash_spec.press_psig + 14.6959) * 101325 / 14.6959
                    state = prep_state(press_pa=press_pa, temp_K=flash_spec.temp_k, 
                                     use_multicomponent_modeling=False)
                    flashresult = flash_calc(state, material=flash_spec.material)
                    
                    promoted = self.scheduler.complete_flash(flash_spec.flash_key, flashresult)
                    self.stats['flash_completed'] += 1
                    
                except Exception as e:
                    skipped = self.scheduler.complete_flash(flash_spec.flash_key, failed=True)
                    self.stats['flash_failed'] += 1
                    self.stats['dispersion_skipped_flash_failed'] += skipped
                
            except Exception as e:
                self.logger.error(f"Flash worker error: {e}")
    
    def dispersion_worker(self):
        """Dispersion calculation worker"""
        while not self.shutdown_event.is_set():
            try:
                disp_spec = self.scheduler.get_ready_dispersion(timeout=5.0)
                if disp_spec is None:
                    continue
                
                if disp_spec.idx <= self.resume_from_idx:
                    continue
                
                memory_status = self.memory_monitor.check_memory()
                
                if not memory_status['under_warning']:
                    if (memory_status['trend'] == 'increasing' or 
                        memory_status['slope'] > 100*1024*1024):  # 100MB/sec increase
                        
                        if not self.scheduler.requeue_dispersion(disp_spec):
                            self.stats['dispersion_failed'] += 1
                        else:
                            self.stats['dispersion_skipped_memory'] += 1
                        continue
                    
                    if not memory_status['under_threshold']:
                        gc.collect()
                        memory_status = self.memory_monitor.check_memory()
                        if not memory_status['under_threshold']:
                            if not self.scheduler.requeue_dispersion(disp_spec):
                                self.stats['dispersion_failed'] += 1
                            else:
                                self.stats['memory_warnings'] += 1
                                self.logger.warning(f"Memory at {memory_status['used_gb']:.1f}GB")
                            continue
                
                with self.scheduler.lock:
                    flashresult = self.scheduler.flash_results.get(disp_spec.flash_key)
                
                if flashresult is None:
                    if not self.scheduler.requeue_dispersion(disp_spec):
                        self.stats['dispersion_failed'] += 1
                    continue
                
                try:
                    result = self.calculator.run_with_interrupt(
                        disp_spec, flashresult, self.timeout_seconds
                    )
                    
                    if result:
                        self._buffer_result(result)
                        self.stats['dispersion_completed'] += 1
                    else:
                        self.stats['dispersion_failed'] += 1
                
                except TimeoutError:
                    self.logger.warning(f"Dispersion {disp_spec.task_id} timed out")
                    self.stats['dispersion_timeout'] += 1
                except Exception as e:
                    self.logger.error(f"Dispersion {disp_spec.task_id} failed: {e}")
                    self.stats['dispersion_failed'] += 1
                
            except Exception as e:
                self.logger.error(f"Dispersion worker error: {e}")
    
    def _buffer_result(self, row_data):
        """Buffer results for batch writing"""
        with self.output_lock:
            self.output_buffer.append(row_data)
            
            if len(self.output_buffer) >= self.batch_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Write buffered results to file"""
        if not self.output_buffer:
            return
        
        try:

            
            f = open(self.output_file, 'a', newline='')
            writer = csv.DictWriter(f, fieldnames=list(self.output_buffer[0].keys()))
            
            if not self.header_written:
                if self.resume_from_idx < 0:
                    with csv_lock:
                        writer.writeheader()
                self.header_written = True
            with csv_lock:
                writer.writerows(self.output_buffer)
            
            f.close()
            
            max_idx = max(row.get('idx', -1) for row in self.output_buffer)
            if max_idx >= 0:
                self._update_idx(max_idx)
            
            self.output_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Failed to write results: {e}")
    
    def _update_idx(self, idx):
        """Update current index file"""
        try:
            with open('curr_idx.txt', 'w') as f:
                f.write(str(idx))
        except Exception as e:
            self.logger.warning(f"Could not update index: {e}")
    
    def generate_and_schedule_tasks(self):
        """Generate tasks and schedule them intelligently"""
        mat_and_props_list = helpers.load_pickled_object('data/materials.pickle')
        
        elevs_m = np.linspace(0, 6, 4)
        orientations = [0, 90]
        pressures_psig = np.linspace(0, 500, 101)
        pressures_psig[0] = 1
        release_kmoles = [100, 300, 1000, 3000, 10000, 30000]
        hole_sizes_in = np.linspace(0.1, 10, 11)
        
        flash_count = 0
        dispersion_count = 0
        idx = -1
        
        for material_row in mat_and_props_list:
            if self.shutdown_event.is_set():
                break
            
            material = material_row['material']
            nbp_deg_k = max(200, material_row['nbp_deg_k'])
            cas_no = material_row['cas_no']
            mw = helpers.get_mw(cas_no, cheminfo=self.cheminfo)
            chem_name = material_row['chem_name']
            
            min_t = max(100, nbp_deg_k - 100)
            max_t = max(400, nbp_deg_k + 100)
            temps_k = np.linspace(min_t, max_t, 21)
            
            args_for_vp_calc = {
                'cheminfo': self.cheminfo,
                'mixture_cas_nos': [cas_no],
                'mixture_molfs': [1],
            }
            
            vps = [vpress_pa_and_vapor_phase_comp_and_component_vapor_pressures(
                temp_k=temp_k, args=args_for_vp_calc)['vpress_pa'] for temp_k in temps_k]
            vps = np.array([max(0, vp) for vp in vps])
            
            for press_psig in pressures_psig:
                for temp_k, vp in zip(temps_k, vps):
                    flash_key = f"{material}_{press_psig:.1f}_{temp_k:.1f}"
                    
                    flash_spec = FlashSpec(
                        flash_key=flash_key,
                        material=material,
                        cas_no=cas_no,
                        chem_name=chem_name,
                        press_psig=press_psig,
                        temp_k=temp_k
                    )
                    self.scheduler.add_flash_spec(flash_spec)
                    flash_count += 1
                    
                    base_row = {
                        **material_row,
                        'mw': mw,
                        'press_psig': press_psig,
                        'temp_k': temp_k,
                        'vp_pa': vp,
                        'flash_calc': pd.NA
                    }
                    
                    for release_kmol in release_kmoles:
                        release_mass_kg = release_kmol * mw
                        
                        for elev_m in elevs_m:
                            for orientation in orientations:
                                for hole_size_in in hole_sizes_in:
                                    idx += 1
                                    
                                    if idx <= self.resume_from_idx:
                                        continue
                                    
                                    row_data = {
                                        **base_row,
                                        'release_mass_kg': release_mass_kg,
                                        'elev_m': elev_m,
                                        'orientation': orientation,
                                        'hole_size_in': hole_size_in
                                    }
                                    
                                    disp_spec = DispersionSpec(
                                        task_id=f"disp_{idx:06d}",
                                        idx=idx,
                                        flash_key=flash_key,
                                        row_data=row_data
                                    )
                                    
                                    if self.scheduler.add_dispersion_spec(disp_spec):
                                        dispersion_count += 1
                
                if flash_count % 1000 == 0:
                    self.logger.info(f"Generated {flash_count} flash, {dispersion_count} dispersion tasks")
        
        self.scheduler.mark_complete()
        self.logger.info(f"Task generation complete: {flash_count} flash, {dispersion_count} dispersion")
    
    def monitor_progress(self):
        """Enhanced progress monitoring"""
        while not self.shutdown_event.is_set():
            time.sleep(15)
            
            current_time = time.time()
            elapsed = current_time - self.stats['start_time']
            
            stats = self.stats.copy()
            
            with self.scheduler.lock:
                scheduler_stats = self.scheduler.stats.copy()
            
            memory_status = self.memory_monitor.check_memory()
            
            flash_rate = stats['flash_completed'] / elapsed * 3600
            disp_rate = stats['dispersion_completed'] / elapsed * 3600
            
            self.logger.info(
                f"Progress [{elapsed/3600:.1f}h] - "
                f"Flash: {stats['flash_completed']}/{scheduler_stats['unique_flash_keys']} ({flash_rate:.0f}/hr), "
                f"Disp: {stats['dispersion_completed']} ({disp_rate:.0f}/hr), "
                f"Pending: {scheduler_stats['dispersions_pending']}, "
                f"Ready: {scheduler_stats['dispersions_ready']}, "
                f"Memory: {memory_status['used_gb']:.1f}GB ({memory_status['trend']})"
            )
            
            if stats['dispersion_timeout'] > 0:
                self.logger.warning(f"Timeouts: {stats['dispersion_timeout']}")
            if stats['memory_warnings'] > 0:
                self.logger.warning(f"Memory warnings: {stats['memory_warnings']}")
            if scheduler_stats['dispersions_requeued'] > 0:
                self.logger.info(f"Requeued: {scheduler_stats['dispersions_requeued']} (max: {scheduler_stats['dispersions_max_requeues']})")
    
    def run_study(self):
        """Run the complete production study"""
        start_time = time.time()
        
        try:
            self.logger.info("Starting production async dispersion study...")
            
            workers = []
            
            for i in range(self.flash_workers):
                worker = threading.Thread(target=self.flash_worker, name=f"Flash-{i}")
                worker.start()
                workers.append(worker)
            
            for i in range(self.dispersion_workers):
                worker = threading.Thread(target=self.dispersion_worker, name=f"Disp-{i}")
                worker.start()
                workers.append(worker)
            
            monitor_thread = threading.Thread(target=self.monitor_progress, name="Monitor")
            monitor_thread.start()
            workers.append(monitor_thread)
            
            self.generate_and_schedule_tasks()
            
            self.logger.info("Task generation complete, waiting for completion...")
            
            # Wait for completion with atomic check
            while not self.shutdown_event.is_set():
                time.sleep(30)
                
                with self.scheduler.lock:
                    if self.scheduler._is_complete():
                        self.logger.info("All work complete, shutting down...")
                        break
            
            self.shutdown_event.set()
            
            for worker in workers:
                worker.join(timeout=10)
            
            self.dispersion_executor.shutdown(wait=True, timeout=30)
            
            with self.output_lock:
                self._flush_buffer()
            
        except Exception as e:
            self.logger.error(f"Study failed: {e}")
            self.shutdown_event.set()
        
        finally:
            self._save_flash_cache()
            
            total_time = time.time() - start_time
            stats = self.stats.copy()
            
            with self.scheduler.lock:
                scheduler_stats = self.scheduler.stats.copy()
            
            self.logger.info("="*70)
            self.logger.info("PRODUCTION STUDY COMPLETE")
            self.logger.info("="*70)
            self.logger.info(f"Total time: {total_time/3600:.1f} hours")
            self.logger.info(f"Flash calculations:")
            self.logger.info(f"  Completed: {stats['flash_completed']}")
            self.logger.info(f"  Failed: {stats['flash_failed']}")
            self.logger.info(f"Dispersion calculations:")
            self.logger.info(f"  Completed: {stats['dispersion_completed']}")
            self.logger.info(f"  Failed: {stats['dispersion_failed']}")
            self.logger.info(f"  Timed out: {stats['dispersion_timeout']}")
            self.logger.info(f"  Skipped (flash failed): {stats['dispersion_skipped_flash_failed']}")
            self.logger.info(f"  Skipped (memory): {stats['dispersion_skipped_memory']}")
            self.logger.info(f"Efficiency metrics:")
            self.logger.info(f"  Tasks requeued: {scheduler_stats['dispersions_requeued']} (max: {scheduler_stats['dispersions_max_requeues']})")
            self.logger.info(f"  Memory warnings: {stats['memory_warnings']}")
            self.logger.info(f"  Average rates: {stats['dispersion_completed']/(total_time/3600):.0f} disp/hr, {stats['flash_completed']/(total_time/3600):.0f} flash/hr")
            self.logger.info("="*70)


def run_production_study(flash_workers=6, dispersion_workers=4, timeout_minutes=15, 
                        memory_threshold_gb=7.0, memory_warning_gb=6.5, 
                        batch_size=100, max_ready_queue_size=500):
    """
    Run the production async study
    
    Args:
        flash_workers: Number of flash calculation workers
        dispersion_workers: Number of dispersion calculation workers  
        timeout_minutes: Timeout for individual dispersion calculations
        memory_threshold_gb: Hard memory limit
        memory_warning_gb: Soft memory limit
        batch_size: Number of results to batch for file writing
        max_ready_queue_size: Maximum ready queue size to control memory
    """
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    runner = ProductionAsyncDispersionRunner(
        flash_workers=flash_workers,
        dispersion_workers=dispersion_workers,
        memory_threshold_gb=memory_threshold_gb,
        memory_warning_gb=memory_warning_gb,
        output_file='model_run_output.csv',
        batch_size=batch_size,
        timeout_seconds=timeout_minutes * 60,
        max_ready_queue_size=max_ready_queue_size
    )
    
    print("="*70)
    print("PRODUCTION ASYNCHRONOUS DISPERSION STUDY")
    print("="*70)
    print(f"Flash workers: {flash_workers}")
    print(f"Dispersion workers: {dispersion_workers}")
    print(f"Timeout: {timeout_minutes} minutes")
    print(f"Memory threshold: {memory_threshold_gb}GB (warning: {memory_warning_gb}GB)")
    print(f"Batch size: {batch_size}")
    print(f"Max ready queue: {max_ready_queue_size}")
    print(f"Resume from index: {runner.resume_from_idx}")
    print("="*70)
    
    runner.run_study()


if __name__ == "__main__":
    run_production_study(
        flash_workers=6, 
        dispersion_workers=4, 
        timeout_minutes=15,
        memory_threshold_gb=7.0,
        memory_warning_gb=6.5,
        batch_size=100,
        max_ready_queue_size=500
    )
