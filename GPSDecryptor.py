from forest.jasmine.traj2stats import Frequency, gps_stats_main
from forest.oak.base import run
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
import pandas as pd
import pickle
from contextlib import contextmanager
from distutils.dir_util import copy_tree

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
        self.gps_base_dir = os.path.join(input_dir, 'PROTECTED', 'STAR', str(self.pid), 'phone', 'raw')
        self.phone_base_dir = os.path.join(input_dir, 'GENERAL', 'STAR', str(self.pid), 'phone', 'raw')
        with open(cred_file, 'r') as f:
            self.passphrase = f.readline().strip()
        gps_dir_re = re.compile(r".*/gps$")
        accel_dir_re = re.compile(r".*/accelerometer$")
        self.logger.info(f"Looking for bewie id in {self.phone_base_dir}")
        self.bewie_id = os.listdir(self.phone_base_dir)[0]
        if self.bewie_id != os.listdir(self.gps_base_dir)[0]:
            raise ValueError('Bewie ID not the same for PROTECTED and GENERAL')
        self.accel_dir = os.path.join(self.phone_base_dir, self.bewie_id, 'accelerometer')
        self.gps_dir = os.path.join(self.gps_base_dir, self.bewie_id, 'gps')
        self.logger.info(f"bewie id: {self.bewie_id}")
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
    """

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

    def decrypt_file(self, file_path, output_dir):
        '''
        Decrypt a single encrypted file.

        Args:
            file_path (str): Path to the encrypted file.
            tmpdirname (str): Temporary directory path where decrypted files are stored.
        '''
        
        output_fn = os.path.basename(file_path).replace('.csv.lock', '.csv')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_fn)
        raw = self.get(file_path)
        key = crypt.key_from_file(raw, self.passphrase)
        crypt.decrypt(raw, key, filename=output_file)
        return(output_file)
            
    def get_file_list_for_dates(self, 
                                file_dir, 
                                start_date=datetime(1900, 1, 1, 0, 0, 0), 
                                end_date=datetime(9900, 1, 1, 0, 0, 0), 
                                file_suffix = r'\.csv\.lock'):
        self.logger.info(f"Collecting filenames...")
        try:
            files = os.listdir(file_dir)
            if start_date or end_date:
                #need to be datetime objects for now
                files_to_return = []
                file_date_re = re.compile(rf"^(\d{{4}}-\d{{1,2}}-\d{{1,2}} \d{{1,2}}_\d{{1,2}}_\d{{1,2}}).*{file_suffix}$")
                for file in files:
                    file_date = file_date_re.match(file)[1]
                    filetime = datetime.strptime(file_date, "%Y-%m-%d %H_%M_%S")
                    if filetime > start_date and filetime < end_date:
                        files_to_return.append(file)
            else:
                files_to_return = files
            
            file_paths = [os.path.join(file_dir, f) for f in files_to_return]
        except Exception as e:
            raise(e)
        return(file_paths)
    
    def decrypt(self, file_paths, output_dir):
        # Determine the number of processes based on SLURM_CPUS_PER_TASK or available CPU cores
        self.logger.info(f"Starting to decrypt into {output_dir}")
        self.logger.info(f"Files to decrypt: {len(file_paths)}")
        
        for file_path in file_paths:
            self.decrypt_file(file_path, output_dir)

        self.logger.info('Decrypting done.')
        return
    
    def copy_files(self, file_paths, output_dir):
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        except Exception as e:
            self.logger.error(f'Could not make dir: {e}')
            
        for file in file_paths:
            try:
                if os.path.isfile(file):
                    shutil.copy(file, output_dir)
                else:
                    self.logger.info(f"Skipping non-existent or non-file path: {file}")
            except Exception as e:
                self.logger.error(f'Could not copy files: {e}')
                raise

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
        # list of locations to track if visited, leave None if don't want these summary statistics
        places_of_interest = None
        self.logger.info(f"Study dir is: {tmpdirname}")
        gps_stats_main(study_folder=tmpdirname, 
                    output_folder=output_dir, 
                    tz_str=tz_str, 
                    frequency=frequency, 
                    save_traj=save_traj, 
                    places_of_interest=places_of_interest, 
                    participant_ids = [f"{self.pid}"],
                    all_memory_dict=all_memory_dict,
                    all_bv_set=all_bv_set)
        self.logger.info('GPS stats generation completed.')
        for file in os.listdir(output_dir):
            self.logger.info(f"Output dir: {output_dir}")
            self.logger.info("Files: ")
            self.logger.info(f"  {file}")
        

    def process_gps(self, ses=0, start_date=datetime(1900, 1, 1, 0, 0, 0), end_date=datetime(9900, 1, 1, 0, 0, 0)):
        '''
        Process GPS data by decrypting and summarizing it.

        Args:
            ses (int or str, optional): Session identifier. Defaults to 0.
            start_date (datetime, optional): Start date for processing. Defaults to far past.
            end_date (datetime, optional): End date for processing. Defaults to far future.
        '''
        formatted_ses = self.format_ses(ses)
        formatted_pid = self.pid
        processed_output_dir = os.path.join(self.pid_dir, f'ses-{start_date:%y%m%d}STAR{formatted_pid}{formatted_ses}')
        if not os.path.exists(os.path.join(processed_output_dir, f"{self.pid}.csv")) or not os.path.exists(os.path.join(processed_output_dir, 'trajectory', f"{self.pid}.csv")): 
            with tempfile.TemporaryDirectory() as tempdir:
                output_dir = os.path.join(tempdir, f"{self.pid}", 'gps')
                self.logger.info("Collecting file paths...")
                file_paths = self.get_file_list_for_dates(file_dir=self.gps_dir, start_date=start_date, end_date=end_date)
                self.decrypt(file_paths, output_dir)
                self.logger.info(f"Processing data for {start_date} to {end_date}")
                self.summarize_gps(tempdir, ses=ses, start_date=start_date, end_date=end_date)
        else:
            self.logger.info('Skipping, output already exists.')
    
    def summarize_accel(self, tmpdirname, ses:str, start_date=datetime(1900, 1, 1, 0, 0, 0), end_date=datetime(9900, 1, 1, 0, 0, 0)):
        self.logger.info(f"Processing accelerometer data for {start_date} to {end_date}")
        formatted_ses = self.format_ses(ses)
        formatted_pid = self.pid
        output_dir = os.path.join(self.pid_dir, f'ses-{start_date:%y%m%d}STAR{formatted_pid}{formatted_ses}')
        self.logger.info(f'Summarizing into {output_dir}')

        start_date_str = start_date.strftime("%Y-%m-%d %H_%M_%S")
        end_date_str = end_date.strftime("%Y-%m-%d %H_%M_%S")
        tz_str = "America/New_York"
        self.logger.info(f"Using time zone: {tz_str}")
        frequency = Frequency.DAILY

        run(tmpdirname, output_dir, tz_str, frequency, start_date_str, end_date_str)
        self.logger.info('Accelerometer stats generation completed.')
        for file in os.listdir(output_dir):
            self.logger.info(f"Output dir: {output_dir}")
            self.logger.info("Files: ")
            self.logger.info(f"  {file}")

    def process_accel(self, ses="0", start_date=datetime(1900, 1, 1, 0, 0, 0), end_date=datetime(9900, 1, 1, 0, 0, 0)):
        formatted_ses = self.format_ses(ses)
        formatted_pid = self.pid
        processed_output_dir = output_dir = os.path.join(self.pid_dir, f'ses-{start_date:%y%m%d}STAR{formatted_pid}{formatted_ses}')
        if not os.path.exists(os.path.join(processed_output_dir, 'daily', f"{self.pid}_gait_daily.csv")): 
            with tempfile.TemporaryDirectory() as tempdir:
                self.logger.info("Collecting file paths...")
                file_paths = self.get_file_list_for_dates(file_dir=self.accel_dir, 
                                                        start_date=start_date, 
                                                        end_date=end_date, 
                                                        file_suffix=r'\.csv')
                output_dir = os.path.join(tempdir, f"{self.pid}", 'accelerometer')
                self.logger.info(f"Copying {len(file_paths)} files to {output_dir}")
                self.copy_files(file_paths, output_dir)
                self.logger.info(f"Processing data for {start_date} to {end_date}")
                self.summarize_accel(tempdir, ses=ses, start_date=start_date, end_date=end_date)
        else:
            self.logger.info('Skipping, output already exists.')
            
        