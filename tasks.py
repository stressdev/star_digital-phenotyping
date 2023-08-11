from invoke import task, Collection
import os
import re
import shutil
import logging
import pandas as pd
from GPSDecryptor import GPSDecryptor
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def setup_logging(log_file: str = None, log_level: str ='INFO'):
    """
    Sets up and returns the logger based on the provided log_file and log_level.
    
    Args:
        log_file (str): Path of the file to log to. Defaults to None.
        log_level (str): The logging level (e.g. 'INFO', 'DEBUG'). Defaults to 'INFO'.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(__name__)

    # Clear existing handlers if present
    if logger.hasHandlers(): 
        logger.handlers = []

    # Set the logging level based on the input argument
    logging_level = getattr(logging, log_level)
    logger.setLevel(logging_level)

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    
    # If log_file is provided, log to file
    if log_file:
        log_file_handler = logging.FileHandler(log_file)
        log_file_handler.setFormatter(log_formatter)
        logger.addHandler(log_file_handler)
    else:
        # Otherwise, create a console handler and attach the formatter
        log_console_handler = logging.StreamHandler()
        log_console_handler.setFormatter(log_formatter)
        logger.addHandler(log_console_handler)
    
    return logger

def get_session_dates(pid: int, ses_file, logger=None):
    """
    Returns session dates for a given participant ID.

    Args:
        pid (int): Participant ID.
        ses_file: File path containing session information.
        logger (logging.Logger, optional): Logger object for debugging and logging purposes.

    Returns:
        pd.DataFrame: A DataFrame containing session dates and related information.
    """
    def not_is_notneeded(col_name):
        return( not re.match(r"^(Unnamed|Scan Date).*", col_name) )

    # Filter out unnecessary columns and filter sessions by pid
    sessions = pd.read_csv(ses_file, usecols=not_is_notneeded, skiprows=[1])
    sessions = sessions[sessions.fevd == int(pid)]
    logger.debug(f"Raw sessions shape after filtering by fevd=={pid}: {sessions.shape}")
    
    # Melting and transformations for a cleaner dataset
    sessions_l = sessions.melt(id_vars = 'fevd')
    sessions_l[['variable', 'sub_session', 'session']] = sessions_l['variable'].str.extract('(Session|Scan Date) (\d+[AB]*)* *\(M(\d+)\)')
    sessions_l.dropna(subset=['value'], how='all', inplace=True)
    sessions_l['value'] = pd.to_datetime(sessions_l['value'], format='mixed', errors='coerce')
    sessions_l.rename(columns = {'value': 'start', 'fevd': 'pid'}, inplace=True)
    sessions_l['end'] = sessions_l['start'].shift(-1)
    sessions_l['ndays'] = sessions_l['end'] - sessions_l['start']
    
    logger.debug(f"Long sessions shape after transforms: {sessions_l.shape}")
    
    return( sessions_l[['pid', 'start', 'end', 'session', 'sub_session', 'ndays']].dropna(subset='ndays') )

def gps_for_pid(pid: int, ses_file: str, cred_file=None, input_dir=None, out_dir=None, logger=None):
    """
    Process GPS data for a particular participant ID.

    Args:
        pid (int): Participant ID.
        ses_file (str): Session file path.
        cred_file (str, optional): Path to credentials file. Defaults to None.
        input_dir (str, optional): Directory where input files reside. Defaults to None.
        out_dir (str, optional): Directory to save output data. Defaults to None.
        logger (logging.Logger, optional): Logger object for debugging and logging purposes.
    """
    logger.info(f"Processing GPS data for {pid}")
    try:
        sessions = get_session_dates(pid, ses_file, logger=logger)
    except Exception as e:
        logger.exception(f"Can't get session dates: {e}")
    
    logger.debug(f"Sessions shape for {pid}: {sessions.shape}")
    logger.debug(f"Sessions for {pid}: {sessions}")
    
    gps_decryptor = GPSDecryptor(pid, cred_file, input_dir, out_dir, logger=logger)
    
    # Decrypting GPS data for each session
    for ses in sessions[['start', 'end', 'ndays', 'sub_session']].itertuples():
        try:
            gps_decryptor.process_gps(ses = ses.sub_session, start_date = ses.start, end_date = ses.end)
        except Exception as e:
            logger.exception(f"Couldn't decrypt {pid} between {ses.start:%Y-%m-%d} to {ses.end:%Y-%m-%d}: {e}")

@task
def gps(c, pid: int=None, allpid: bool=False, cred_file: str = '.credentials'):
    """
    Invokable GPS processing task.
    Processes either a single PID's GPS data or all PIDs based on the provided options.
    
    Args:
        c: Invoke context.
        pid (int, optional): Participant ID to process. Defaults to None.
        allpid (bool, optional): Whether to process all PIDs. Defaults to False.
        cred_file (str): Path to credentials file. Defaults to '.credentials'.
    """
    logger = setup_logging(log_file = c.log_file, log_level = c.log_level)
    if not pid and not allpid:
        raise ValueError("Must either provide a PID or specify `--allpid`")
    if allpid:
        try: 
            dirs = os.listdir(c.gps_dir)
            id_list = [d for d in dirs if re.match(r'1\d{3}', d) and os.path.isdir(os.path.join(c.gps_dir, d))]
        except Exception as e:
            logger.exception(f"Couldn't list gps data directories: {e}")
        if not id_list:
            raise ValueError(f"Could not find any PID directories in {c.gps_dir}")
        
        # Batch processing logic using sbatch
        sbatch_template = f"""
{c.sbatch_header}
id_array=({' '.join(id_list)})
id=\${{id_array[\$SLURM_ARRAY_TASK_ID]}}
. PYTHON_MODULES.txt
mamba activate star_diph
echo "Processing \$id"
invoke gps --pid \$id
EOF
"""
        cmd = f"sbatch --array=0-{len(id_list)-1} <<EOF {sbatch_template}"
        logger.debug(f"Command:\n\n{cmd}")
        sbatch_result = c.run(cmd)
        logger.info(f"{sbatch_result.stdout}\n{sbatch_result.stderr}")
    else:
        input_dir = os.path.join(c.gps_dir, f"{pid}")    
        gps_for_pid(pid, ses_file=c.ses_file, cred_file=cred_file, input_dir=input_dir, out_dir=c.out_dir, logger=logger)

@task
def summarize_gps(c, pid: int=None, allpid: bool=False, cred_file: str = '.credentials'):
    """
    Invokable task to summarize GPS data.
    Placeholder for future implementation.
    
    Args:
        c: Invoke context.
        pid (int, optional): Participant ID. Defaults to None.
        allpid (bool, optional): Whether to summarize for all PIDs. Defaults to False.
        cred_file (str): Path to credentials file. Defaults to '.credentials'.
    """
    pass
    

#ensure we run in the location of this file.
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))    
os.chdir(dir_of_this_script)

# Collection of invoke tasks with default configuration
ns = Collection(gps)
ns.configure({'log_level': "INFO", 
              'log_file': "invoke.log", 
              'gps_dir': "gps", 
              'out_dir': "out",
              'ses_file': "sessions.csv",
              'sbatch_header': """#!/bin/bash
echo PLEASE DEFINE THIS IN invoke.yaml
"""})  