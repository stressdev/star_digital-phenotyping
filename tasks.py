from invoke import task, Collection
import os, sys
import re
import shutil
import logging
import pandas as pd
from GPSDecryptor import GPSDecryptor
from datetime import datetime
from string import Template

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

def summarize_gps_for_pid(pid: int=None, out_dir=None, logger=None):
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
    date_cols = ['year', 'month', 'day']

    pid_dir = os.path.join(out_dir, f"sub-{pid}")
    ses_dirs = [(os.path.join(pid_dir, d), d) for d in os.listdir(pid_dir) if re.match('ses-.*', d)]
    dfs = []

    for ses_dir, ses in ses_dirs:
        ses = re.match('ses-\d+STAR\d{4}(\d{2})[AB]*', ses)[1]
        file = os.path.join(ses_dir, f"{pid}.csv")
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
    agg_funcs.update({col: ses_first for col in date_cols})

    df_agg = df_all.groupby(idcols).agg(agg_funcs)
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    outfile = os.path.join(pid_dir, f"sub-{pid}_monthly.csv")
    
    logger.info(f"Writing aggregated data to {outfile}")
    df_agg.to_csv(outfile)
                

def phone_for_pid(pid: int, ses_file: str, cred_file=None, input_dir=None, out_dir=None, logger=None):
    logger.info(f"Processing call and text data for {pid}")
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
            logger.exception(f"Couldn't process calls and texts for {pid} between {ses.start:%Y-%m-%d} to {ses.end:%Y-%m-%d}: {e}")

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
    ses_plots_template = Template("""
## $ses

::: {.panel-tabset}

### GPS Trajectory

```{python}
ses = "$ses"
traj = rf.get_traj(sub, ses)
if traj is not None:
    plot_map(traj)
```

:::
""")

    logger.info(f"Creating report for {pid}")
    logger.debug("Openning template")
    
    # Load the main report template. This serves as a base structure where session data will be embedded.
    with open('report_template.qmd', 'r') as file:
        template = Template(file.read())

    # Default output directory structure is designed for consistency and easy lookup.
    if outdir is None:
        outdir = os.path.join(c.out_dir, f"sub-{pid}")
    
    # Standardized naming convention for report files ensures uniformity.
    qmd_file = f"report_sub-{pid}.qmd"
    out_qmd = os.path.join(outdir, qmd_file)
    
    # Embed the participant ID into the main template.
    logger.debug("Formatting template")
    qmd = template.substitute(sub=f"sub-{pid}")
    
    # Fetching all session directories to report on any data that has been processed.
    ses_dirs = [d for d in os.listdir(f'/ncf/mclaughlin_lab_tier1/STAR/8_digital/sub-{pid}/') if re.match('ses-.*', d)]
    for ses_dir in ses_dirs:
        qmd += ses_plots_template.substitute(ses = ses_dir)
    
    # Once the report structure is ready, it's saved as a .qmd file, which can later be converted to other formats.
    logger.debug("Writing qmd")
    with open(out_qmd, 'w') as file:
        file.write(qmd)
    
    # Conversion to HTML format is for easy viewing in web browsers.
    logger.debug("Rendering qmd to html")
    logger.debug(f"Current conda env: {os.environ['CONDA_DEFAULT_ENV']}")
    with c.prefix('. /ncf/mclaughlin/users/jflournoy/code/spack/share/spack/setup-env.sh'):
        with c.prefix('spack load quarto-cli'):
            logger.debug(f"Current conda env: {os.environ['CONDA_DEFAULT_ENV']}")
            run_info = c.run(f"quarto check", echo=True, echo_stdin=True,
                             env={'XDG_RUNTIME_DIR': '/n/holyscratch01/LABS/mclaughlin_lab/Users/jflournoy/'}) 
            logger.debug(run_info)
    with c.prefix('. /ncf/mclaughlin/users/jflournoy/code/spack/share/spack/setup-env.sh'):
        with c.prefix('spack load quarto-cli'):
            logger.debug(f"Current conda env: {os.environ['CONDA_DEFAULT_ENV']}")
            run_info = c.run(f"quarto render {out_qmd} --execute-debug --to html", echo=True, echo_stdin=True,
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
        logger.error(f"Error processing gps for {pid}.")
    
    try:
        summarize_gps_for_pid(pid=pid, out_dir=out_dir, logger=logger)
    except Exception as e:
        logger.error(f"Error summarizing GPS for {pid}.")
    
    try:
        phone_for_pid(pid=pid, ses_file=ses_file, cred_file=cred_file, input_dir=input_dir, out_dir=out_dir, logger=logger)
    except Exception as e:
        logger.error(f"Error processing gps for {pid}.")
    
    try:
        summarize_phone_for_pid(pid=pid, out_dir=out_dir, logger=logger)
    except Exception as e:
        logger.error(f"Error summarizing GPS for {pid}.")

    try:
        report_for_pid(c, pid=pid, outdir=None, logger=logger)
    except Exception as e:
        logger.error(f"Error generating report for {pid}.")
    
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
        input_dir = os.path.join(c.gps_dir, f"{pid}")    
        try:
            gps_for_pid(pid, ses_file=c.ses_file, cred_file=cred_file, input_dir=input_dir, out_dir=c.out_dir, logger=logger)
        except Exception as e:
            logger.error(f"Problem computing gps trajectories: {e}")

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
        summarize_gps_for_pid(pid=pid, outdir=c.out_dir, logger=logger)
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

    
#--------------------------------------    

# Ensuring a consistent runtime environment, regardless of where the script is invoked from.
dir_of_this_script = os.path.dirname(os.path.abspath(__file__))    
os.chdir(dir_of_this_script)

# Task collections make it possible to group multiple related operations, simplifying the orchestration.
ns = Collection(gps, make_report, summarize_gps, build)

# Default configurations are set to ensure smooth operations even if specific settings aren't provided.
ns.configure({'log_level': "INFO", 
              'log_file': "invoke.log", 
              'gps_dir': "gps", 
              'out_dir': "out",
              'ses_file': "sessions.csv",
              'sbatch_header': """#!/bin/bash
echo PLEASE DEFINE THIS IN invoke.yaml
"""})  
