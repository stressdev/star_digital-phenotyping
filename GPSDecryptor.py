import os
import shutil
import multiprocessing
import cryptease as crypt
import getpass as gp
import tempfile
import re
import logging
from datetime import datetime
from glob import glob
from forest.jasmine.traj2stats import Frequency, gps_stats_main
from forest.oak.base import run
import pandas as pd
import pickle
from contextlib import contextmanager

# Compatibility between Python 2 and 3 for the 'input' function
try:
    input = raw_input
except NameError:
    pass

# Compatibility between Python 2 and 3 for the 'urlparse' function
try:
    import urlparse as up
except ImportError:
    import urllib.parse as up

class PrefixAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter that prefixes messages.
    """
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['prefix'], msg), kwargs
    
    
class GPSDecryptor:
    """
    Utility for decrypting and summarizing GPS data.
    """
    def __init__(self, pid: int, cred_file, input_dir, out_dir, logger=None):
        """
        Initialize GPSDecryptor object.

        Args:
            pid (int): Participant ID.
            cred_file (str): Path to the file containing the passphrase.
            input_dir (str): Path to the input directory.
            out_dir (str): Path to the output directory.
            logger (logging.Logger, optional): Custom logger instance.
        """
        self.pid = pid
        self.cred_file = cred_file
        self.input_dir = input_dir
        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.handlers:
            # create console handler and set level to debug
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            # create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # add formatter to ch
            ch.setFormatter(formatter)
            # add ch to logger
            self.logger.addHandler(ch)
        self.logger = PrefixAdapter(self.logger, {'prefix': f"PID {self.pid}"})
        gps_base_dir = os.path.join(input_dir, 'PROTECTED', 'STAR', str(self.pid))
        phone_base_dir = os.path.join(input_dir, 'GENERAL', 'STAR', str(self.pid))
        with open(cred_file, 'r') as f:
            self.passphrase = f.readline().strip()
        gps_dir_re = re.compile(r".*/gps$")
        accel_dir_re = re.compile(r".*/accelerometer$")
        self.logger.info(f"Looking for GPS data in {gps_base_dir}")
        self.gps_dir = [x[0] for x in os.walk(gps_base_dir) if bool(gps_dir_re.match(x[0]))][0]
        self.logger.info(f"Looking for accelerometer data in {phone_base_dir}")
        self.accel_dir = [x[0] for x in os.walk(phone_base_dir) if bool(accel_dir_re.match(x[0]))][0]
        self.gps_summary = None
        self.all_memory_dict = None
        self.all_bv_set = None
        self.pid_dir = os.path.join(out_dir, f'sub-{pid}')

    def __str__(self):
        """
        Return a string representation of the GPSDecryptor object.
        """
        return f"""
GPSDecryptor object:
    pid: {self.pid}
    cred_file: {self.cred_file}
    input_dir: {self.input_dir}
    gps_dir: {self.gps_dir}
    accel_dir: {self.accel_dir}
    gps_summary: {self.gps_summary}
    all_memory_dict: {self.all_memory_dict}
    all_bv_set: {self.all_bv_set}
    pid_dir: {self.pid_dir}"""

    def get(self, f):
        '''
        Retrieve file content from a given URI.

        Args:
            f (str): File URI.

        Returns:
            file content: Content of the file.
        '''
        # Parse the URI
        uri = up.urlparse(f)
        username = uri.username
        
        # Retrieve content based on URI scheme
        if not username:
            username = gp.getuser()
        if uri.scheme in ['', 'file']:
            content = open(uri.path, 'rb')
        elif uri.scheme == 'ssh':
            with paramiko.SSHClient() as client:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())
                client.load_system_host_keys()
                try:
                    client.connect(uri.hostname, username=username, password='')
                except paramiko.ssh_exception.AuthenticationException:
                    password = gp.getpass('enter password for {}@{}: '.format(username, uri.hostname))
                    client.connect(uri.hostname, username=username, password=password)
                chan = client.get_transport().open_session()
                chan.exec_command("cat '{}'".format(uri.path))
                content = chan.makefile('r', 1024)
                stderr = chan.makefile_stderr('r', 1024)
                exit_status = chan.recv_exit_status()
                if exit_status != 0:
                    raise SSHError(stderr.read())
        else:
            raise URIError('unexpected scheme: {}'.format(uri.scheme))
        return content

    def decrypt_file(self, file_path, tmpdirname):
        '''
        Decrypt a single encrypted file.

        Args:
            file_path (str): Path to the encrypted file.
            tmpdirname (str): Temporary directory path where decrypted files are stored.
        '''
        
        pattern_dir = re.compile(r".*(\d{4})/.*")
        output_fn = os.path.basename(file_path).replace('.csv.lock', '.csv')
        output_dir = os.path.join(tmpdirname, pattern_dir.search(os.path.dirname(file_path)).group(1), 'gps')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_fn)
        raw = self.get(file_path)
        key = crypt.key_from_file(raw, self.passphrase)
        crypt.decrypt(raw, key, filename=output_file)
    
    @contextmanager
    def decrypt(self, start_date=datetime(1900, 1, 1, 0, 0, 0), end_date=datetime(9900, 1, 1, 0, 0, 0)):
        '''
        Context manager for decrypting all files in a given date range.

        Args:
            start_date (datetime, optional): Start date for decryption range. Defaults to far past.
            end_date (datetime, optional): End date for decryption range. Defaults to far future.
        
        Yields:
            str: Temporary directory where decrypted files are stored.
        '''
        tmpdirname = tempfile.mkdtemp()
        try:
            self.logger.info(f"Created temporary directory {tmpdirname}")            
            self.logger.info(f"Decrypting...")
            
            gps_files = os.listdir(self.gps_dir)
            if start_date or end_date:
                #need to be datetime objects for now
                gps_files_to_decrypt = []
                gps_file_date_re = re.compile(r"^(\d{4}-\d{1,2}-\d{1,2} \d{1,2}_\d{1,2}_\d{1,2}).*\.csv\.lock$")
                for gps_file in gps_files:
                    gps_file_date = gps_file_date_re.match(gps_file)[1]
                    filetime = datetime.strptime(gps_file_date, "%Y-%m-%d %H_%M_%S")
                    if filetime > start_date and filetime < end_date:
                        gps_files_to_decrypt.append(gps_file)
            else:
                gps_files_to_decrypt = gps_files
            
            file_paths = [os.path.join(self.gps_dir, f) for f in gps_files_to_decrypt]
            
            # Determine the number of processes based on SLURM_CPUS_PER_TASK or available CPU cores
            cpus_per_task = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
            self.logger.info(f"Files to decrypt: {len(file_paths)} - CPUs: {cpus_per_task}")
            num_processes = min(cpus_per_task, len(file_paths))

            with multiprocessing.Pool(processes=num_processes) as pool:
                pool.starmap(self.decrypt_file, zip(file_paths, [tmpdirname] * len(file_paths)))

            self.logger.info('Decrypting done.')
            yield tmpdirname
        finally:
            shutil.rmtree(tmpdirname)

    def format_ses(self, s):
        # Extract numeric and optional letter parts
        numeric_part = ''.join([ch for ch in s if ch.isdigit()])
        letter_part = ''.join([ch for ch in s if ch.isalpha()])
        # Format numeric part with two digits
        formatted_numeric = f"{int(numeric_part):02d}"
        return f"{formatted_numeric}{letter_part}"
    
    def format_pid(self, pid):
        if isinstance(pid, int):
            pid_str = f"{pid:04}"
        elif isinstance(pid, str) and pid.isdigit():
            pid_str = f"{int(pid):04}"
        else:
            pid_str = pid
        return(pid_str)

    def summarize_gps(self, tmpdirname, ses:str, start_date=datetime(1900, 1, 1, 0, 0, 0), end_date=datetime(9900, 1, 1, 0, 0, 0)):
        '''
        Generate a summary for GPS data from decrypted files.

        Args:
            tmpdirname (str): Temporary directory where decrypted files are located.
            ses (str): Session identifier.
            start_date (datetime, optional): Start date for summarizing. Defaults to far past.
            end_date (datetime, optional): End date for summarizing. Defaults to far future.
        '''

        formatted_ses = self.format_ses(ses)
        formatted_pid = self.pid
        self.logger.info('Generating GPS stats...')
        output_dir = os.path.join(self.pid_dir, f'ses-{start_date:%y%m%d}STAR{formatted_pid}{formatted_ses}')
        self.logger.info(f'Summarizing into {output_dir}')

        # all_memory_dict_fn = os.path.join(output_dir, 'all_memory_dict.pkl')
        # all_bv_set_fn = os.path.join(output_dir, 'all_bv_set.pkl')

        all_memory_dict = None
        all_bv_set = None

        tz_str = "America/New_York"
        self.logger.info(f"Using time zone: {tz_str}")
        # Generate summary metrics e.g. Frequency.HOURLY, Frequency.DAILY or Frequency.HOURLY_AND_DAILY (see Frequency class in traj2stats.py)
        frequency = Frequency.DAILY
        # Save imputed trajectories?
        save_traj = True
        # Hyperparameters class for imputation (default leave None), from forest.jasmine.traj2stats import Hyperparameters
        parameters = None
        # list of locations to track if visited, leave None if don't want these summary statistics
        places_of_interest = None
        # True if want to save a log of all locations and attributes of those locations visited
        save_osm_log = False
        # threshold of time spent in a location to count as being in that location, in minutes
        threshold = 0
        # 3. Impute location data and generate mobility summary metrics using the simulated data above
        gps_stats_main(tmpdirname, 
                       output_dir, 
                       tz_str, 
                       frequency, 
                       save_traj, 
                       parameters, 
                       places_of_interest, 
                       save_osm_log, 
                       threshold,
                       all_memory_dict=all_memory_dict,
                       all_bv_set=all_bv_set
                      )
        self.logger.info('GPS stats generation completed.')

    def process_gps(self, ses=0, start_date=datetime(1900, 1, 1, 0, 0, 0), end_date=datetime(9900, 1, 1, 0, 0, 0)):
        '''
        Process GPS data by decrypting and summarizing it.

        Args:
            ses (int or str, optional): Session identifier. Defaults to 0.
            start_date (datetime, optional): Start date for processing. Defaults to far past.
            end_date (datetime, optional): End date for processing. Defaults to far future.
        '''
        with self.decrypt(start_date=start_date, end_date=end_date) as decrypt_dir:
            self.logger.info(f"Processing data for {start_date} to {end_date}")
            self.summarize_gps(decrypt_dir, ses=ses, start_date=start_date, end_date=end_date)
    
    def process_accel(self, ses="0", start_date=datetime(1900, 1, 1, 0, 0, 0), end_date=datetime(9900, 1, 1, 0, 0, 0)):
        '''
        Process call and text data by summarizing it.

        Args:
            ses (int or str, optional): Session identifier. Defaults to 0.
            start_date (datetime, optional): Start date for processing. Defaults to far past.
            end_date (datetime, optional): End date for processing. Defaults to far future.
        '''
        self.logger.info(f"Processing call and text data for {start_date} to {end_date}")
        formatted_ses = self.format_ses(ses)
        formatted_pid = self.pid
        output_dir = os.path.join(self.pid_dir, f'ses-{start_date:%y%m%d}STAR{formatted_pid}{formatted_ses}')
        self.logger.info(f'Summarizing into {output_dir}')

        tz_str = "America/New_York"
        self.logger.info(f"Using time zone: {tz_str}")
        # Generate summary metrics e.g. Frequency.HOURLY, Frequency.DAILY or Frequency.HOURLY_AND_DAILY (see Frequency class in traj2stats.py)
        frequency = Frequency.DAILY
        
        run(self.accel_dir, output_dir, tz_str, frequency, start_date, end_date)