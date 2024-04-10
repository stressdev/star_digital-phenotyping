from invoke import task, Collection
import os, sys
import re
import shutil
import logging
import pandas as pd
from GPSDecryptor import GPSDecryptor
from datetime import datetime
from string import Template
import multiprocessing
import json

# When data acquisition falls between sessions, how do we assign which session it belongs to? In these scripts
# I assign the data collected between session N and N+1 to session N. That is, if the data were collected between
# on or after Session 1 (2022-01-01) and before Session 2 (2022-02-01), it would be assigned to session 1. 
# If the data were collected on or after Session 2 (2022-02-01) and before Session 3 (2022-03-01), it would be 
# assigned to session 2. It will be up to the user to make adjustments to the data if they feel a variable collected
# at Session N should be examine with respect to the data collected before or after that session, knowing that the digital
# phenotyping data collected *after* session N will be assigned to that session.


class MyTemplate(Template):
    delimiter = '@@'

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

def format_ses(s):
    # Extract numeric and optional letter parts
    numeric_part = ''.join([ch for ch in s if ch.isdigit()])
    letter_part = ''.join([ch for ch in s if ch.isalpha()])
    # Format numeric part with two digits
    formatted_numeric = f"{int(numeric_part):02d}"
    return f"{formatted_numeric}{letter_part}"

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
    logger.debug(sessions.columns)
    sessions = sessions[sessions.fevd == int(pid)]
    logger.debug(f"Raw sessions shape after filtering by fevd=={pid}: {sessions.shape}")
    
    # Melting and transformations for a cleaner dataset
    sessions_l = sessions.melt(id_vars = 'fevd')
    sessions_l[['variable', 'sub_session', 'session']] = sessions_l['variable'].str.extract('(Session|Scan Date) (\d+[AB]*)* *\(M(\d+)\)')
    sessions_l.dropna(subset=['value'], how='all', inplace=True)
    sessions_l['value'] = pd.to_datetime(sessions_l['value'], format='mixed', errors='coerce')
    sessions_l.rename(columns = {'value': 'start', 'fevd': 'pid'}, inplace=True)
    sessions_l['end'] = sessions_l['start'].shift(-1)
    sessions_l['sub_session_lag1'] = sessions_l['sub_session'].shift(-1)
    sessions_l['ndays'] = sessions_l['end'] - sessions_l['start']
    
    logger.debug(f"Long sessions shape after transforms: {sessions_l.shape}")
    
    return( sessions_l[['pid', 'start', 'end', 'session', 'sub_session', 'ndays', 'sub_session_lag1']].dropna(subset='ndays') )

def process_session(pid, ses_sub_session, ses_start, ses_end, method_name, cred_file, input_dir, out_dir, logger):
    logger.debug(f"Working on {method_name} for {pid} between {ses_start:%Y-%m-%d} to {ses_end:%Y-%m-%d}")
    
    try:
        processor = GPSDecryptor(pid, cred_file, input_dir, out_dir, logger=logger)
        # Dynamically get the method to call
        process_method = getattr(processor, method_name)
        # Call the method
        process_method(ses=ses_sub_session, start_date=ses_start, end_date=ses_end)
    except Exception as e:
        logger.exception(f"Error {method_name} for {pid} between {ses_start:%Y-%m-%d} to {ses_end:%Y-%m-%d}: {e}")

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
    
    # Determine the number of CPUs allocated by Slurm or default to 1
    num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 1))
    logger.info(f"Number of CPUs allocated: {num_cpus}")
    
    method_name = 'process_gps'
    session_args = [(pid, ses.sub_session, ses.start, ses.end, method_name, cred_file, input_dir, out_dir, logger) 
                    for ses in sessions[['start', 'end', 'sub_session']].itertuples(index=False)]

    # Use multiprocessing Pool for parallel processing
    with multiprocessing.Pool(processes=num_cpus) as pool:
        pool.starmap(process_session, session_args)

def find_files(directory, regex_pattern):
    # Compile the regex pattern
    regex = re.compile(regex_pattern)

    # List all files in the directory
    for filename in os.listdir(directory):
        # If the filename matches the regex pattern, yield it
        if regex.match(filename):
            yield filename

def read_file(file, logger):
    # Set the read method based on the file extension
    if file.endswith('.csv'):
        read_method = pd.read_csv
    elif file.endswith('.xlsx'):
        read_method = pd.read_excel
    else:
        return None

    # Check if the first row produces unnamed columns
    df = read_method(file, nrows=0)  # Read only the header row
    logger.debug(f'df: {df.shape}')
    unnamed_cols = df.columns.str.contains('^Unnamed')
    if unnamed_cols.any() or not df.shape[1] > 1:
        logger.info(f"File {file} has unnamed columns in the header row.")
        df = read_method(file, skiprows=1)  # Skip the first line
    else:
        df = read_method(file)
    return df

def drop_rows(df, threshold, logger):
    # Remove rows where most of the columns have missing data
    rows_before = df.shape[0]
    df = df.dropna(thresh=threshold)
    rows_after = df.shape[0]

    # Calculate and accumulate the number of dropped rows
    dropped_rows = rows_before - rows_after
    logger.info(f"Dropped {dropped_rows} of {rows_before} rows due to NA cols.")

    return df, dropped_rows

def process_files(data_dir, file_regex, threshold, logger):
    dfs = []
    total_dropped_rows = 0
    logger.debug(f"Looking for files in {data_dir} using {file_regex}")
    for file in find_files(data_dir, file_regex):
        logger.debug(f'file: {file}')
        file = os.path.join(data_dir, file)
        try:
            df = read_file(file, logger)
            if df is not None:
                df, dropped_rows = drop_rows(df, threshold, logger)
                total_dropped_rows += dropped_rows
                dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")

    combined_df = pd.concat(dfs, ignore_index=True)

    # Remove rows where the timestamp is not unique in the combined DataFrame
    rows_before = combined_df.shape[0]
    # Create a mask for the duplicated rows
    duplicated_mask = combined_df.duplicated(keep=False)

    # Select the duplicated rows
    duplicated_rows = combined_df[duplicated_mask]
    combined_df = combined_df.drop_duplicates(keep=False)
    rows_after = combined_df.shape[0]

    # Calculate and log the number of dropped rows
    dropped_rows = rows_before - rows_after
    total_dropped_rows += dropped_rows
    logger.info(f"Dropped {dropped_rows} of {rows_before} rows from the combined DataFrame due to duplicate timestamps.")
    logger.info(f"Total dropped rows: {total_dropped_rows}")

    return combined_df, duplicated_rows

def session_tag_phone_logs(phone_df, sessions, logger):
    #Find first and last session
    
    sub_session_first = sessions[sessions['start'] == min(sessions['start'])].iloc[0].sub_session
    sub_session_last = sessions[sessions['start'] == max(sessions['start'])].iloc[0].sub_session
    for index, row in sessions.iterrows():
        # For the first session, we assign anything before the second session to the first session.
        if row.sub_session == sub_session_first:
            within_session_index = (phone_df['timestamp'] < row.end.timestamp())    
        # For the last session, we assign anthing after the last session to the last session.
        elif row.sub_session == sub_session_last:
            within_session_index = (phone_df['timestamp'] >= row.start.timestamp())
        else:
            within_session_index = (phone_df['timestamp'] >= row.start.timestamp()) & (phone_df['timestamp'] < row.end.timestamp())
        # Use the index that marks which phone data belong to the session with the session and sub_session number
        phone_df.loc[within_session_index, 'session'] = row.session
        phone_df.loc[within_session_index, 'sub_session'] = row.sub_session
    if phone_df[phone_df.sub_session.isna()].shape[0] > 0:
        logger.warning('Some rows not assinged session number')
        logger.warning(f"{phone_df[phone_df.sub_session.isna()]}")
    return(phone_df.sort_values(by='timestamp'))

def summarize_phone_logs(log_df, groupby_cols, lengthcol, logger):
    for col in groupby_cols:
        if col not in log_df.columns:
            error_text = f"Column '{col}' not found in phone log dataframe."
            logger.error(error_text)
            raise ValueError(error_text)
    log_sum_df = log_df.groupby(groupby_cols).agg(
        N=pd.NamedAgg(column='timestamp', aggfunc='count'),
        N_length_zero=pd.NamedAgg(column=lengthcol, aggfunc=lambda x: sum(~(x > 0))), 
        N_unique_phone_numbers=pd.NamedAgg(column='hashed phone number', aggfunc=lambda x: x.nunique()),
        length_sum=pd.NamedAgg(column=lengthcol, aggfunc='sum')
    ).reset_index()
    return(log_sum_df)

def get_phone_data(pid: int, data_type: str, input_dir: str, out_dir: str, logger=None):
    if data_type not in ['call', 'texts']:
        logger.error(f"data_type must be either 'call' or 'texts'. Got: {data_type}.")
        raise ValueError("data_type must be either 'call' or 'texts'")
    regex = f"{pid}_\d+-*\d*_{data_type}.*(\.csv|\.xlsx)$"
    data_dir_name = 'iMazing CALL LOGS' if data_type == 'call' else 'iMazing TEXT LOGS'
    data_dir = os.path.join(input_dir, data_dir_name)
    threshold = 3  # We require at least 3 columns to consider the row legitimate
    df, dupes_df = process_files(data_dir, regex, threshold, logger)
    dupes_df_out_fn = os.path.join(out_dir, f"sub-{pid}", f"sub-{pid}_{data_type}_duplicates.csv")
    dupes_df.to_csv(dupes_df_out_fn, index=False)
    df.rename(columns={'call type' if data_type == 'call' else 'sent vs received': 'type'}, inplace=True)
    return df

def save_logs(pid: int, phone_logs: dict, sessions: pd.DataFrame, out_dir: str, logger=None):
    if not set(phone_logs.keys()).issubset({'calls', 'txts'}):
        error_msg = f"Phone log keys must be either 'call' or 'txts'. Got: {phone_logs.keys()}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    pid_dir = os.path.join(out_dir, f"sub-{pid}")
    lengthcol = {'calls' : 'duration in seconds', 'txts' : 'message length'}
    logger.info(f"Saving text and call logs into session directories and creating monthly summaries for sub-{pid}")
    for key, log in phone_logs.items():
        log['timestamp'] = log['timestamp'].astype(float)
        log = session_tag_phone_logs(log, sessions, logger)
        monthly_data_outfile_name = os.path.join(pid_dir, f"sub-{pid}_{key}_monthly.csv")
        monthly_log = summarize_phone_logs(log, groupby_cols=['session', 'sub_session', 'type'], lengthcol=lengthcol[key], logger=logger)
        monthly_log.to_csv(monthly_data_outfile_name, index=False)
        grouped_log = log.groupby(['session', 'sub_session'])
        for group_name, group_df in grouped_log:
            ses_start_date = sessions[(sessions.session == group_name[0]) & (sessions.sub_session == group_name[1])].start.iloc[0]
            formatted_ses = format_ses(group_name[1])
            ses_output_dir = os.path.join(pid_dir, f'ses-{ses_start_date:%y%m%d}STAR{pid}{formatted_ses}')
            ses_output_path = os.path.join(ses_output_dir, f"sub-{pid}_ses-{formatted_ses}_{key}.csv")
            group_df.to_csv(ses_output_path, index=False)

def phone_for_pid(pid: int, ses_file: str, cred_file=None, input_dir=None, out_dir=None, logger=None):
    logger.info(f"Processing phone data for {pid}")
    try:
        sessions = get_session_dates(pid, ses_file, logger=logger)
    except Exception as e:
        logger.exception(f"Can't get session dates: {e}")

    # Process CALL LOGS
    call_df = get_phone_data(pid, 'call', input_dir, out_dir, logger)

    # Process TXT data
    txt_df = get_phone_data(pid, 'texts', input_dir, out_dir, logger)

    phone_logs = {'calls': call_df, 'txts': txt_df}

    save_logs(pid, phone_logs, sessions, out_dir, logger)

def accel_for_pid(pid: int, ses_file: str, cred_file=None, input_dir=None, out_dir=None, logger=None):
    logger.info(f"Processing accelerometer data for {pid}")
    try:
        sessions = get_session_dates(pid, ses_file, logger=logger)
    except Exception as e:
        logger.exception(f"Can't get session dates: {e}")
    
    logger.debug(f"Sessions shape for {pid}: {sessions.shape}")
    logger.debug(f"Sessions for {pid}: {sessions}")
    
    # Determine the number of CPUs allocated by Slurm or default to 1
    num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 1))
    logger.info(f"Number of CPUs allocated: {num_cpus}")
    
    method_name = 'process_accel'
    session_args = [(pid, ses.sub_session, ses.start, ses.end, method_name, cred_file, input_dir, out_dir, logger) 
                    for ses in sessions[['start', 'end', 'sub_session']].itertuples(index=False)]

    # Use multiprocessing Pool for parallel processing
    with multiprocessing.Pool(processes=num_cpus) as pool:
        pool.starmap(process_session, session_args)
            
def summarize_for_pid(pid: int, dtype: str, csv_file: str=None, out_dir=None, logger=None):
    """
    Summarizes GPS data across days by session for a given participant ID (pid).
    Output is a csv file with averages per month.
    
    Args:
    - pid (int): participant ID
    - out_dir (str): output directory
    - logger (Logger): logger object
    
    Returns:
    - None
    
    Raises:
    - ValueError: if pid is None
    """
    if pid is None:
        logger.error("PID is not given")
        raise ValueError("`pid` is None")
    logger.info(f"Summarizing gps data by session for {pid}...")
    
    def ses_first(x):
        return(x.iloc[0])
    
    idcols = ['ID', 'Session']
    date_cols = ['year', 'month', 'day', 'date']

    pid_dir = os.path.join(out_dir, f"sub-{pid}")
    ses_dirs = [(os.path.join(pid_dir, d), d) for d in os.listdir(pid_dir) if re.match('ses-.*', d)]
    dfs = []

    for ses_dir, ses in ses_dirs:
        ses = re.match('ses-\d+STAR\d{4}(\d{2})[AB]*', ses)[1]
        file = os.path.join(ses_dir, csv_file)
        try:
            df = pd.read_csv(file)
            df['ID'] = pid
            df['Session'] = int(ses)
            dfs.append(df)
        except FileNotFoundError:
            logger.warning(f"File {file} was not found.")
        except pd.errors.EmptyDataError:
            logger.warning(f"File {file} is empty.")
        except pd.errors.ParserError:
            logger.error(f"Error parsing {file}.")
        except Exception as e:
            logger.exception(f"An unexpected error occurred with {file}.")

    # Concatenate all successfully read dataframes
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        logger.exception("No files were successfully read.")
        return

    df_all.sort_values(idcols, inplace=True)

    agg_funcs = {col: ['mean', 'median', 'count'] for col in df.columns if col not in idcols and col not in date_cols}
    agg_funcs.update({col: ses_first for col in date_cols if col in df.columns})

    df_agg = df_all.groupby(idcols).agg(agg_funcs)
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    outfile = os.path.join(pid_dir, f"sub-{pid}_{dtype}_monthly.csv")
    
    logger.info(f"Writing aggregated data to {outfile}")
    df_agg.to_csv(outfile)
            
def report_for_pid(c, pid: int=None, outdir=None, logger=None):
    """
    Generate a report for a specific participant based on their GPS and session data.

    The function fetches the GPS data of a participant (identified by their PID) and 
    structures the data into a standardized report template. The report template embeds 
    various visualizations like Lat/Long histograms and GPS trajectory. Once structured, 
    the report is saved as a `.qmd` file and is also converted to HTML for easy viewing 
    in web browsers.

    Parameters:
    - c: The context or environment for executing commands (like file operations or running other scripts).
    - pid (int, optional): The Participant ID for which the report needs to be generated. If not provided, an error is raised.
    - outdir (str, optional): The directory where the output report will be saved. If not provided, a default directory structure is used.
    - logger (optional): Logger instance to capture and log various stages of report generation. Useful for debugging.

    Raises:
    - ValueError: If `pid` is not provided (i.e., is None).

    Notes:
    - The main report structure is fetched from 'report_template.qmd'.
    - The session data structure is defined within the function as `ses_plots_template`.
    - The output report gets saved in two formats: `.qmd` and `.html`.

    Example usage:
    >>> report_for_pid(c=context, pid=12345, logger=myLogger)
    """
    if pid is None:
        logger.error("PID is not given")
        raise ValueError("`pid` is None")
    # The session plots template is structured to provide visualizations of GPS data.
    # It's designed to integrate with the main report seamlessly.

    logger.info(f"Creating report for {pid}")
    logger.debug("Openning template")
    
    # Load the main report template. This serves as a base structure where session data will be embedded.
    with open('report_template.qmd', 'r') as file:
        template = MyTemplate(file.read())

    # Default output directory structure is designed for consistency and easy lookup.
    if outdir is None:
        outdir = os.path.join(c.out_dir, f"sub-{pid}")
    
    qmd_file = f"report_sub-{pid}.qmd"
    out_qmd = os.path.join(outdir, qmd_file)
    
    # Embed the participant ID into the main template.
    logger.debug("Formatting template")
    qmd = template.substitute(sub=f"sub-{pid}")
    
    # Once the report structure is ready, it's saved as a .qmd file, which can later be converted to other formats.
    logger.debug("Writing qmd")
    with open(out_qmd, 'w') as file:
        file.write(qmd)
    
    # Conversion to HTML format is for easy viewing in web browsers.
    logger.debug("Rendering qmd to html")
    with c.prefix('OVERLAY="$SCRATCH/LABS/mclaughlin_lab/Users/jflournoy/$(uuidgen).img"'):
        with c.prefix('singularity overlay create --size 2512 "${OVERLAY}"'):
            run_info = c.run(f"singularity exec -B /ncf --overlay ${{OVERLAY}} {c.R_singularity_image} quarto render {out_qmd} --execute-debug --to html", echo=True, echo_stdin=True,
                             env={'XDG_RUNTIME_DIR': '/n/holyscratch01/LABS/mclaughlin_lab/Users/jflournoy/'}) 
            logger.debug(run_info)

def build_pid(c, pid: int, ses_file: str, cred_file=None, input_dir=None, out_dir=None, logger=None):
    """
    Builds a digital phenotyping output for a given patient ID.

    Args:
        c (object): Invoke context object.
        pid (int): The participant ID.
        ses_file (str): The file with session data.
        cred_file (str, optional): The credential file. Defaults to None.
        input_dir (str, optional): The input directory. Defaults to None.
        out_dir (str, optional): The output directory. Defaults to None.
        logger (object, optional): A logger object. Defaults to None.

    Raises:
        ValueError: If `pid` is None.

    Returns:
        None
    """
    if pid is None:
        logger.error("PID is not given")
        raise ValueError("`pid` is None")
    logger.info(f"Building {pid}...")

    try:
        gps_for_pid(pid=pid, ses_file=ses_file, cred_file=cred_file, input_dir=input_dir, out_dir=out_dir, logger=logger)
    except Exception as e:
        logger.error(f"Error processing gps for {pid}: {e}")
    
    try:
        summarize_for_pid(pid=pid, dtype='gps', csv_file=f"{pid}.csv", out_dir=out_dir, logger=logger)
    except Exception as e:
        logger.error(f"Error summarizing GPS for {pid}: {e}")
    
    try:
        accel_for_pid(pid=pid, ses_file=ses_file, cred_file=cred_file, input_dir=input_dir, out_dir=out_dir, logger=logger)
    except Exception as e:
        logger.error(f"Error processing accelerometer for {pid}: {e}")

    try:
        summarize_for_pid(pid=pid, dtype='accel', csv_file=os.path.join('daily', f"{pid}_gait_daily.csv"), out_dir=out_dir, logger=logger)
    except Exception as e:
        logger.error(f"Error summarizing accelerometer for {pid}: {e}")
                          
    try:
        phone_input_dir = c.phone_data_fasse_dir
        phone_for_pid(pid=pid, ses_file=ses_file, cred_file=cred_file, input_dir=phone_input_dir, out_dir=out_dir, logger=logger)
    except Exception as e:
        logger.error(f"Error processing phone data for {pid}: {e}")
    
    try:
        report_for_pid(c, pid=pid, outdir=None, logger=logger)
    except Exception as e:
        logger.error(f"Error generating report for {pid}: {e}")
    
def sbatch_invoke_allpid(c, invoke_task, logger, test=False):
    """
    Submit a batch job to run an invoke task for all participant ids.
    For each pid, the `invoke_task` command is invoked with the `--pid` option for each PID in the array.
    The job is submitted using the `sbatch` command with the given `c.sbatch_header` options.
    If `test` is True, only the first two directories are processed (for testing purposes).
    The `logger` object is used to log debug and info messages.

    Args:
        c (Config): Invoke context object.
        invoke_task (str): the name of the command to invoke for each PID.
        logger (Logger): the logger object to use for logging.
        test (bool): whether to run a test job with only the first two directories.

    Raises:
        ValueError: if no PID directories are found in `c.gps_dir`.
    """
    try: 
        sessions = pd.read_csv(c.ses_file)
        id_list = [f"{d:0.0f}" for d in sessions.fevd.dropna()]
        id_list = [d for d in id_list if re.match(r'1\d{3}', d) and os.path.isdir(os.path.join(c.gps_dir, str(d)))]
    except Exception as e:
        logger.exception(f"Couldn't confirm list of gps data directories: {e}")
    if not id_list:
        raise ValueError(f"Could not find any PID directories in {c.gps_dir}")

    # Batch processing logic using sbatch
    sbatch_template = f"""
{c.sbatch_header}
id_array=({' '.join(id_list)})
id=\${{id_array[\$SLURM_ARRAY_TASK_ID]}}

echo $CONDA_DEFAULT_ENV
echo $CONDA_SHLVL
echo $CONDA_PYTHON_EXE
echo "Processing \$id"
invoke {invoke_task} --pid \$id
EOF
"""
    if test:
        cmd = f"sbatch --array=0-1 <<EOF {sbatch_template}"
    else:
        cmd = f"sbatch --array=0-{len(id_list)-1} <<EOF {sbatch_template}"
    logger.debug(f"Command:\n\n{cmd}")
    sbatch_result = c.run(cmd)
    logger.info(f"{sbatch_result.stdout}\n{sbatch_result.stderr}")


@task
def gps(c, pid: int=None, allpid: bool=False, cred_file: str = '.credentials'):
    """
    GPS data processing.
    Processes either a single PID's GPS data or all PIDs based on the provided options.
    
    Args:
        c: Invoke context.
        pid (int, optional): Participant ID to process. Defaults to None.
        allpid (bool, optional): Whether to process all PIDs. Defaults to False.
        cred_file (str, optional): Path to credentials file. Defaults to '.credentials'.
        noreport (bool, optional): Do not make a report. Defaults to False.
    """
    logger = setup_logging(log_file = c.log_file, log_level = c.log_level)
    if not pid and not allpid:
        raise ValueError("Must either provide a PID or specify `--allpid`")
    if allpid:
        sbatch_invoke_allpid(c, "gps", logger)
    else:
        input_dir = os.path.join(c.input_dir)    
        try:
            gps_for_pid(pid, ses_file=c.ses_file, cred_file=cred_file, input_dir=input_dir, out_dir=c.out_dir, logger=logger)
        except Exception as e:
            logger.error(f"Problem computing gps trajectories: {e}")

@task
def accel(c, pid: int=None, allpid: bool=False, cred_file: str = '.credentials'):
    """
    Accelerometer data processing.
    Processes either a single PID's accelerometer data or all PIDs based on the provided options.
    
    Args:
        c: Invoke context.
        pid (int, optional): Participant ID to process. Defaults to None.
        allpid (bool, optional): Whether to process all PIDs. Defaults to False.
        cred_file (str, optional): Path to credentials file. Defaults to '.credentials'.
        noreport (bool, optional): Do not make a report. Defaults to False.
    """
    logger = setup_logging(log_file = c.log_file, log_level = c.log_level)
    if not pid and not allpid:
        raise ValueError("Must either provide a PID or specify `--allpid`")
    if allpid:
        sbatch_invoke_allpid(c, "accel", logger)
    else:
        input_dir = os.path.join(c.input_dir)    
        try:
            accel_for_pid(pid, ses_file=c.ses_file, cred_file=cred_file, input_dir=input_dir, out_dir=c.out_dir, logger=logger)
        except Exception as e:
            logger.error(f"Problem computing accelerometer data: {e}")

@task
def phone(c, pid: int=None, allpid: bool=False):
    logger = setup_logging(log_file = c.log_file, log_level = c.log_level)
    if not pid and not allpid:
        raise ValueError("Must either provide a PID or specify `--allpid`")
    if allpid:
        sbatch_invoke_allpid(c, "phone", logger)
    else:
        input_dir = os.path.join(c.phone_data_fasse_dir)    
        try:
            phone_for_pid(pid, ses_file=c.ses_file, input_dir=input_dir, out_dir=c.out_dir, logger=logger)
        except Exception as e:
            logger.error(f"Problem processing raw phone data: {e}")

@task
def summarize_gps(c, pid: int=None, allpid: bool=False, cred_file: str = '.credentials'):
    """
    Summarizes GPS data for a given participant ID or for all participants if allpid is True.
    
    Args:
        c (object): Invoke context object.
        pid (int, optional): The participant ID for which to summarize GPS data. Defaults to None.
        allpid (bool, optional): If True, summarizes GPS data for all participants. Defaults to False.
        cred_file (str, optional): The path to the credentials file. Defaults to '.credentials'.
    """
    
    logger = setup_logging(log_file = c.log_file, log_level = c.log_level)
    
    if allpid:
        sbatch_invoke_allpid(c, "summarize-gps", logger)
    else:
        summarize_for_pid(pid=pid, dtype='gps', csv_file=f"{pid}.csv", out_dir=c.out_dir, logger=logger)
        logger.info(f"Done with {pid}")

@task
def summarize_accel(c, pid: int=None, allpid: bool=False, cred_file: str = '.credentials'):
    """
    Summarizes accelerometer data for a given participant ID or for all participants if allpid is True.
    
    Args:
        c (object): Invoke context object.
        pid (int, optional): The participant ID for which to summarize GPS data. Defaults to None.
        allpid (bool, optional): If True, summarizes GPS data for all participants. Defaults to False.
        cred_file (str, optional): The path to the credentials file. Defaults to '.credentials'.
    """
    
    logger = setup_logging(log_file = c.log_file, log_level = c.log_level)
    
    if allpid:
        sbatch_invoke_allpid(c, "summarize-accel", logger)
    else:
        summarize_for_pid(pid=pid, dtype='accel', csv_file=os.path.join('daily', f"{pid}_gait_daily.csv"), out_dir=c.out_dir, logger=logger)
        logger.info(f"Done with {pid}")
        
@task
def make_report(c, pid: int=None, outdir: str=None, allpid: bool=False, test: bool=False):
    """
    Generate data report.
    Creates a report based on the provided PID (Participant ID). It uses templates to 
    generate various sections of the report.
    
    Args:
    - c: context object.
    - pid (int, optional): Participant ID. Defaults to None.
    - outdir (str, optional): Directory where the report will be saved. If not specified,
                              it uses the default directory derived from the context object.
    - allpid (bool, optional): Run for all PID? Default is False.
    """
    # Logging is vital to trace the report creation process and troubleshoot if required.
    logger = setup_logging(log_file = c.log_file, log_level = c.log_level)
    
    if allpid:
        sbatch_invoke_allpid(c, "make-report", logger=logger, test=test)
    else:
        report_for_pid(c, pid, outdir, logger)
        logger.info(f"Done with {pid}")
        
@task
def build(c, pid: int=None, allpid: bool=False, test: bool=False, cred_file: str = '.credentials'):
    """
    Builds the digital phenotyping pipeline for a given participant ID or for all participants if allpid is True.
    
    Args:
    - c: Invokr context object
    - pid: participant ID (int)
    - allpid: flag to indicate if the pipeline should be built for all participants (bool)
    - test: flag to indicate if the pipeline should be run in test mode (bool)
    - cred_file: path to the credentials file (str)
    
    Returns: None
    """
    logger = setup_logging(log_file = c.log_file, log_level = c.log_level)
    
    if allpid:
        sbatch_invoke_allpid(c, invoke_task="build", logger=logger, test=test)
    else:
        input_dir = os.path.join(c.input_dir)
        build_pid(c, pid, ses_file=c.ses_file, cred_file=cred_file, input_dir=input_dir, out_dir=c.out_dir, logger=logger)
        logger.info(f"Done with {pid}")

@task
def clean(c, dry_run=False):
    logger = setup_logging(log_file = c.log_file, log_level = c.log_level)
    pattern = re.compile(r'sub-1\d{3}')
    
    for entry in os.listdir(c.out_dir):
        dir_path = os.path.join(c.out_dir, entry)
        # Check if it's a directory and matches the regex
        if os.path.isdir(dir_path) and pattern.match(entry):
            logger.info(f"Deleting dir: {dir_path}")
            if not dry_run:
                shutil.rmtree(dir_path)
                    

@task
def sync_phone_data(c, dry_run=False):
    logger = setup_logging(log_file = c.log_file, log_level = c.log_level)
    dry_run_string = ""
    
    #check rclone config
    rclone_config_check = c.run('rclone config dump')
    rclone_config_keys = json.loads(rclone_config_check.stdout).keys()
    if 'sox3' not in rclone_config_keys:
        logger.error(f"sox3 not in rclone config: {rclone_config_keys}")
        raise ValueError("sox3 not in rclone config")
    if dry_run:
        dry_run_string = "--dry-run"
    rclone_cmd = f"rclone sync sox3:{c.phone_data_sox3_dir} {c.phone_data_fasse_dir} -P {dry_run_string}"
    rclone_out = c.run(rclone_cmd, echo=True)
    logger.info(f"Finished running rclone")


@task
def summarize_phone_data(c, pid: int=None, allpid: bool=False, cred_file: str = '.credentials'):
    #Not sure yet if this is needed or makes sense
    # logger = setup_logging(log_file = c.log_file, log_level = c.log_level)
    
    # if allpid:
    #     sbatch_invoke_allpid(c, "summarize-phone-data", logger)
    # else:
    #     summarize_for_pid(pid=pid, dtype='phone', csv_file=os.path.join('daily', f"{pid}_phone_daily.csv"), out_dir=c.out_dir, logger=logger)
    #     logger.info(f"Done with {pid}")
    pass
        

#--------------------------------------    

# Ensuring a consistent runtime environment, regardless of where the script is invoked from.
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))    
os.chdir(dir_of_this_script)

# Task collections make it possible to group multiple related operations, simplifying the orchestration.
ns = Collection(gps, make_report, summarize_gps, summarize_accel, build, accel, clean, sync_phone_data, phone)

# Default configurations are set to ensure smooth operations even if specific settings aren't provided.
ns.configure({'log_level': "INFO", 
              'log_file': "invoke.log", 
              'gps_dir': "gps", 
              'out_dir': "out",
              'ses_file': "sessions.csv",
              'sbatch_header': """#!/bin/bash
echo PLEASE DEFINE THIS IN invoke.yaml
"""})  

