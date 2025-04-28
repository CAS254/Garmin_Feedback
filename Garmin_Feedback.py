# This script creates feedback for participants that have worn a Garmin device.
# Version: 1.0
# Date: 19-03-2025
# Author: CAS254

import os
from heapq import merge

import pandas as pd
from colorama import Fore
from colorama import init as colorama_init
import numpy as np
from fontTools.misc.plistlib import end_date
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
import matplotlib.dates as mdates
from datetime import time
from PIL import Image
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import argparse
import time


colorama_init(autoreset=True)

# Folder path for where the Garmin data is saved
data_dir = 'Q:/Data/BRC_Projects/PP04 - Thyroid/_data/Garmin/_data'
feedback_folder = 'Q:/Data/BRC_Projects/PP04 - Thyroid/Participant_feedback/feedback'
collapsed_data = 'Q:/Data/BRC_Projects/PP04 - Thyroid/Participant_feedback/collapsed_data'
plots_path = 'Q:/Data/BRC_Projects/PP04 - Thyroid/Participant_feedback/plots'


# --- THE CODE STARTS BELOW --- #

# --- Creating and exporting PDF with participant feedback --- #
def create_pdf(output_dir, id, num_days, height_mm):
    pdf = FPDF()
    pdf.add_page()

    # Creating title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 10, "Participant Feedback from Garmin", ln=1)
    pdf.line(10, 20, 200, 20)

    # Inserting plot displaying activity and heart rate
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 20, 'Activity and heart rate data', ln=1)

    # Insert participant ID and summary of HR data
    pdf.set_font('Arial', '', 10)
    label_width = 60
    value_width = 60

    pdf.cell(label_width, 5, 'Participant ID:', 1, 0)
    pdf.cell(value_width, 5, str(id), 1, ln=1)
    pdf.cell(label_width, 5, 'Average heart rate:', 1, 0)
    pdf.cell(value_width, 5, f'{mean_hr} bpm', 1, ln=1)
    pdf.cell(label_width, 5, 'Minimum heart rate:', 1, 0)
    pdf.cell(value_width, 5, f'{min_hr} bpm', 1, ln=1)
    pdf.cell(label_width, 5, 'Maximum heart rate:', 1, 0)
    pdf.cell(value_width, 5, f'{max_hr} bpm', 1, ln=1)

    pdf.ln(10)

    # Inserting the activity and heart rate plot
    path = os.path.join(plots_path, f'activity_heartrate/{id}_plot.png')
    y = pdf.get_y()
    pdf.image(path, x=10, y=y, w=180)

    if os.path.exists(path):
        os.remove(path)

    # Setting the y (curser) to the height of the activity and heartrate plot (so that next plot is below this)
    pdf.set_y(y + height_mm + 10)

    # Inserting plot displaying daily steps
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, "Daily steps", ln=1)
    step_path = os.path.join(plots_path, f'steps/{id}_steps.png')
    if os.path.exists(step_path):
        y = pdf.get_y()
        pdf.image(step_path, x=10, y=y + 1, w=120)
        os.remove(step_path)
    else:
        pass

    # Outputting the PDF
    pdf.output(os.path.join(output_dir, f"{id}_feedback.pdf"))


# --- Searching for participant specific data folder --- #
def read_file(list_ids, feedback):
    """
    Searching in data folder for folders for specified participants listed in list_ids or making list of all participants if it is specified to create feedback for all participants
    :param list_ids: ID's to create feedback for
    :param feedback: Specified if feedback should be created for all participants or only for specified IDS
    :return: Dictionary with participant ID and folder path for this/these participants data
    """
    participant_paths = {}
    sub_folders = ['CRF02', 'CRF149', 'CRF400']

    # Get list of IDs that already have feedback (and stripping filenames to get IDs only)
    existing_feedback_ids = set()
    if feedback == 'REMAINING':
        for file in os.listdir(feedback_folder):
            if file.endswith('.pdf'):
                existing_id = file.split('_feedback')[0]
                existing_feedback_ids.add(existing_id)

    # Creating paths for all participant folders within the subfolders
    for sub_dir in sub_folders:
        sub_dir_path = os.path.join(data_dir, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue

        for participant in os.listdir(sub_dir_path):
            full_path = os.path.join(sub_dir_path, participant)
            if os.path.isdir(full_path):

                # Creating dictionary with paths to all participant folders
                if feedback == 'ALL':
                    participant_paths[participant] = full_path

                # Creating dictionary with specified participant folders
                elif feedback == 'IDS' and participant in list_ids:
                    participant_paths[participant] = full_path

                # Creating dictionary with paths to participant folder that HAVEN'T already had feedback created
                elif feedback == 'REMAINING' and participant not in existing_feedback_ids:
                    participant_paths[participant] = full_path

    # Creating list if IDS if creating feedback for all participants
    if feedback == 'ALL' or feedback == 'REMAINING':
        list_ids = list(participant_paths.keys())

    return participant_paths, list_ids


# --- Finding the heart rate start wearing time --- #
def heartrate_wear_time(file_path):
    """
    The function finds the wear start and end time based on HR values.
    :param participant_paths:
    :return: wear start end end time and heart rate dataframe.
    """
    hr_file_path = os.path.join(file_path, f'{id}_heartrate.csv')
    heartrate_df = pd.read_csv(hr_file_path, usecols=['Start Time (Local)', 'Heart Rate (bpm)'], dtype={'Heart Rate (bpm)': 'int16'})

    # Replacing HR values 0, -1, 1 and 255 with missing
    values_to_replace = [0, -1, 1, 255]
    heartrate_df['Heart Rate (bpm)'] = heartrate_df['Heart Rate (bpm)'].replace(values_to_replace, np.nan)

    # Finding the first place in the hr file with 600 consecutive rows that are NOT missing
    rolling_sum = heartrate_df['Heart Rate (bpm)'].notna().rolling(window=600).sum()
    valid_indices = rolling_sum[rolling_sum == 600].index

    rolling_sum_end = heartrate_df['Heart Rate (bpm)'].notna().rolling(window=600).sum()
    end_index = rolling_sum_end[rolling_sum_end == 600].index[-1]
    if valid_indices.empty:
        return None, None, heartrate_df

    # Using the first series of 600 consecutive rows that are not 255 and finding the rows index for the start of this sequence to indicate the start wearing time
    # Finding the Start and end wear time (Local) based on the index
    start_time = heartrate_df.loc[valid_indices[0] - 599, 'Start Time (Local)']
    end_time = heartrate_df.loc[end_index, 'Start Time (Local)']

    return start_time, end_time, heartrate_df


# --- Reading accelerometer data --- #
def read_acc_file(file_path):
    acc_file_path = os.path.join(file_path, f'{id}_accelerometer.csv')
    acc_df = pd.read_csv(acc_file_path, usecols=['First Name', 'Start Time (Local)', 'Start Time (Local)', 'X', 'Y', 'Z'],
                                                 dtype={'X': 'float32', 'Y': 'float32', 'Z': 'float32'})
    return acc_df


# --- Trimming file according to wear times --- #
def trim_file(df, start_time, end_time):
    """
    The function trims a specified file (hr or acc file) according to specified wear start and end times (from HR file)
    :param df:
    :param wear_start_time:
    :param wear_end_time:
    :return:
    """
    # Formatting the Start Time (Local) variable
    df['Start Time (Local)'] = pd.to_datetime(df['Start Time (Local)'])
    start_time = pd.to_datetime(start_time).round('min')
    end_time = pd.to_datetime(end_time).floor('min')

    # Trimming the dataframe to the wear times
    trimmed_df = df.loc[
        (df['Start Time (Local)'] >= start_time) & (df['Start Time (Local)'] <= end_time)].copy()

    # Formatting the Start time variable back to a string
    trimmed_df['Start Time (Local)'] = trimmed_df['Start Time (Local)'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    trimmed_df['Start Time (Local)'] = trimmed_df['Start Time (Local)'].str[:-3]
    return trimmed_df


# --- Calculate ENMO --- #
def calculate_enmo(df):
    """
    The function calculates ENMO and collapses the accelerometer file to minute level
    :param df
    :return:
    """
    df['vektor_magnitude'] = np.sqrt(df['X'] ** 2 + df['Y'] ** 2 + df['Z'] ** 2)
    # Calculating ENMO (Substracting 1000 mg (to get into g) and truncate negative values to 0):
    df['ENMO'] = np.maximum(0, df['vektor_magnitude'] - 1000)
    df = df[['Start Time (Local)', 'ENMO']].copy()
    return df


# --- Collapse dataframe into minute level and create day variable --- #
def collapse_df(df):
    # Collapsing data to minute level :
    df['Start Time (Local)'] = pd.to_datetime(df['Start Time (Local)'])
    df.set_index('Start Time (Local)', inplace=True)
    collapsed_df = df.resample('1min').mean().reset_index()
    # Create day variable relative to start of wear
    collapsed_df['date'] = collapsed_df['Start Time (Local)'].dt.date
    collapsed_df['day'] = (collapsed_df['date'] != collapsed_df['date'].shift()).cumsum()

    return collapsed_df


# --- Removing noise from accelerometer data --- #
def remove_noise(df):
    df['Start Time (Local)'] = pd.to_datetime(df['Start Time (Local)'])
    df['hour'] = df['Start Time (Local)'].dt.hour

    # Calculating the enmo mean for hours where the SD is below 10 (expecting these hours to be asleep). The enmo mean for these hours is used to remove noise
    SD_by_hour = df.groupby('hour')['ENMO'].std()
    #print(SD_by_hour)
    still_hours = SD_by_hour[SD_by_hour < 10].index
    filtered_df = df[df['hour'].isin(still_hours)]
    still_hour_enmo_mean = filtered_df['ENMO'].mean()
    return still_hour_enmo_mean


# --- Merging HR and Acc data --- #
def merge_data(hr_df, acc_df, still_hour_enmo_mean):
    merged_df = hr_df.merge(acc_df, on=['Start Time (Local)', 'day', 'date'])

    # Removing noise from accelerometer data and truncating negative values to 0
    if not np.isnan(still_hour_enmo_mean):
        merged_df['ENMO'] = np.maximum(merged_df['ENMO'] - still_hour_enmo_mean, 0)
    file_path = os.path.join(collapsed_data, f'{id}_collapsed.csv')
    merged_df.to_csv(file_path, index=False)
    return merged_df


# --- Creating plot of HR and acc data for each day --- #
def combine_barplot_lineplot(df):
    # Formatting date time variable and creating time variable
    df['Start Time (Local)'] = pd.to_datetime(df['Start Time (Local)'])
    df['date'] = df['Start Time (Local)'].dt.date
    df['time'] = df['Start Time (Local)'].dt.time


    # Creating a range with the full time from midnight to midnight as the first day will likely start in the middle of the day
    full_times = pd.date_range('00:00', '23:59', freq="min").time
    full_df = pd.DataFrame({'time': full_times})

    # Formatting heart rate value to numeric
    df['Heart Rate (bpm)'] = pd.to_numeric(df['Heart Rate (bpm)'], errors='coerce')

    # Specifying what style to use for the plot
    sns.set_style("darkgrid")
    plt.rcParams.update({'font.size': 9})

    # Creating sub plot for each day
    unique_days = sorted(df['day'].unique())
    num_days = len(unique_days)
    fig, axes = plt.subplots(num_days, 1, figsize=(8, 1.5 * num_days), sharex=True)

    # Ensures code won't crash if only one day of data
    if num_days == 1:
        axes = [axes]

    # Looping through each day to plot it
    for i, day in enumerate(unique_days):
        ax = axes[i]

        # Merging each day with a "full day" range so that the x axis goes from midnight to midnight even if devices wasn't worn the whole duration
        day_df = df[df['day'] == day].copy()
        merged_day_df = day_df.merge(full_df, on='time', how='right')
        merged_day_df['ENMO'] = merged_day_df['ENMO'].fillna(0)


        # Creating minute variable so use a numeric time variable when plotting data over time
        merged_day_df['minutes'] = merged_day_df['time'].apply(lambda t: t.hour * 60 + t.minute)
        merged_day_df['Heart Rate (bpm)'] = merged_day_df['Heart Rate (bpm)'].replace('', np.nan)
        merged_day_df['Heart Rate (bpm)'] = pd.to_numeric(merged_day_df['Heart Rate (bpm)'], errors='coerce')
        merged_day_df['ENMO'] = pd.to_numeric(merged_day_df['ENMO'], errors='coerce')
        merged_day_df['time'] = merged_day_df['time'].apply(lambda t: t.strftime('%H:%M'))

        # Creating separate y-axis for the two plot but shared x-axis
        ax_bar = ax
        ax_line = ax_bar.twinx()

        # Creating bar plot displaying daily enmo
        enmo_color = 'black'
        heartrate_color = 'red'
        if merged_day_df['ENMO'].notna().any():
            ax_bar.bar(merged_day_df['minutes'], merged_day_df['ENMO'], color=enmo_color, width=1.0, edgecolor='black', zorder=2)

        # Creating line plot displaying daily heart rate
        if merged_day_df['Heart Rate (bpm)'].notna().any():
            x_values = merged_day_df['minutes']
            y_values = merged_day_df['Heart Rate (bpm)']
            ax_line.plot(x_values, y_values, color=heartrate_color, linewidth=0.5, zorder=3)

        # Creating shaded area where heart rate is missing
        hr_zero = merged_day_df['Heart Rate (bpm)'].isna()
        zero_blocks = []
        current_block = None

        for j, (minute, is_zero) in enumerate(zip(merged_day_df['minutes'], hr_zero)):
            if is_zero and current_block is None:
                current_block = [minute, minute]
            elif is_zero:
                current_block[1] = minute
            elif current_block:
                zero_blocks.append(current_block)
                current_block = None
        if current_block:
            zero_blocks.append(current_block)
        for block in zero_blocks:
            width = block[1] - block[0]
            rect = Rectangle((block[0], 0), width, 800, facecolor='none', edgecolor='darkgrey', hatch='////', linewidth=0, zorder=1)
            ax_bar.add_patch(rect)


        # --- FORMATTING PLOT --- #
        # Setting y labels
        ax_bar.set_ylabel('Activity', color=enmo_color, fontsize=10)
        ax_line.set_ylabel('Heart Rate (BPM)', color=heartrate_color, fontsize=10)

        # Generating date variable to display data for each sub plot
        day_date = day_df['date'].iloc[0]
        formatted_date = day_date.strftime('%d-%m-%Y')
        weekday = day_date.strftime('%A')
        ax.set_title(f'{weekday} {formatted_date}', fontsize=10)

        # Formatting the y-axis ticks
        ax_bar.tick_params(axis='y', labelcolor=enmo_color)
        ax_bar.set_ylim(0, 800)
        ax_bar.set_yticks([0, 200, 400, 600, 800])
        ax_line.tick_params(axis='y', labelcolor=heartrate_color)
        ax_line.set_ylim(0, 200)
        ax_line.set_yticks([0, 50, 100, 150, 200])

        # Setting grid lines
        plt.grid(True, linestyle='--', alpha=0.1, axis='y', zorder=0)


        # Specifying ticks on the x-axis (only on the last days sub plot):
        if i == num_days - 1:
            ax.set_xlim(0, 1439)
            desired_xticks = [0, 360, 720, 1080, 1439]
            xtick_labels = ['00:00', '06:00', '12:00', '18:00', '23:59']
            ax.set_xticks(desired_xticks)
            ax.set_xticklabels(xtick_labels, ha='center')

            ax_line.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            ax.set_xlabel('\nTime', color=enmo_color, fontsize=10)

        else:
            ax.set_xticks([])

    # Creating legends on the graphs
    activity_legend = Patch(facecolor='black', edgecolor='black', label='Activity')
    hr_legend = Line2D([0], [0], color='red', linewidth=0.5, label='Heart Rate')
    missing_hr_legend = Patch(facecolor='none', edgecolor='darkgrey', hatch='////', linewidth=0, label='Device not worn/poor signal')

    fig.legend(handles=[activity_legend, hr_legend, missing_hr_legend], loc='upper left', ncol=3, bbox_to_anchor=(0.43, 1.08), frameon=True, framealpha=1.0, facecolor='white', edgecolor='lightgrey')

    # Optimizing the layout of the subplots and adding space between each plot
    plt.tight_layout(pad=1.0)
    path = os.path.join(plots_path, f'activity_heartrate/{id}_plot.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')

    # re-opening plot to get the height for the pdf (To be able to move the curser)
    with Image.open(path) as img:
        width_px, height_px = img.size
    dpi = 96
    px_per_mm = dpi / 25.4
    height_mm = height_px / px_per_mm
    plt.close(fig)
    return num_days, height_mm


# --- Summarising heart rate data --- #
def sum_heartrate(df):
    min_hr = int(round(df['Heart Rate (bpm)'].min(), 0))
    max_hr = int(round(df['Heart Rate (bpm)'].max(), 0))
    mean_hr = int(round(df['Heart Rate (bpm)'].mean(), 0))

    return min_hr, max_hr, mean_hr


# --- Reading in dailies file --- #
def read_dailies(file_path):
    dailies_file_path = os.path.join(file_path, f'{id}_dailies.csv')
    if os.path.exists(dailies_file_path):
        dailies_df = pd.read_csv(dailies_file_path, usecols=['Calendar Date (Local)', 'Steps'], dtype={'Steps': 'int16'})

        dailies_df['Date'] = pd.to_datetime(dailies_df['Calendar Date (Local)']).dt.strftime('%d-%m-%y')

        # Clearing any existing plots
        plt.clf()

        # Creating barplot to display daily steps
        sns.set_style("darkgrid")
        plt.rcParams.update({'font.size': 9})
        fig, ax = plt.subplots()
        sns.barplot(x='Date', y='Steps', data=dailies_df, ax=ax)
        plt.xlabel('\nDate')
        plt.ylabel('Steps')
        plt.title('Daily steps count', fontsize=14, fontweight='bold')

        for index, row in dailies_df.iterrows():
            plt.text(index, row['Steps'] + 100, str(row['Steps']), ha='center', fontsize=8)

        max_steps = dailies_df['Steps'].max()
        plt.ylim(0, max_steps + 500)

        plt.grid(True, linestyle='--', alpha=0.5, axis='y')

        plt.tight_layout()
        path = os.path.join(plots_path, f'steps/{id}_steps.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        pass

# --- Calling functions --- #
if __name__ == '__main__':

    # Making script interactive, so the user needs to input arguments
    parser = argparse.ArgumentParser(description="Creating Garmin feedback for one or multiple IDs")

    # Prompt to enter if user want to create feedback for all or specified IDs only:
    parser.add_argument(
        "--number_feedback",
        type=int,
        choices=[1, 2, 3],
        help="Select an option: 1 = Create feedback for all participants, 2 = Create feedback for all participants that haven't already got feedback, 3 = Specify what IDs you want to create feedback for."
    )

    # Prompt to enter ID(s) that you want to create feedback for:
    parser.add_argument(
        "--participant_id",
        type=str,
        help="Comma-seperated list of Participant IDs to search for."
    )
    args = parser.parse_args()

    number_feedback = args.number_feedback or int(input(f"Select a number (1-3) from the list below to specify who you want to create feedback for. Press ENTER once you have made your selection: \n\n"
                                                        + Fore.YELLOW + "1 = All participants \n2 = Only participants without existing feedback \n3 = Manually specify participant IDS to create feedback for. \n\n" + Fore.RESET + "Enter a number and press ENTER:  "))


    # Creating tag, so that code can process all ID's or just specified ID's
    if number_feedback == 1:
        print(Fore.CYAN + f'Creating feedback for all participants.')
        feedback = 'ALL'
        list_ids = []
    if number_feedback == 2:
        print(Fore.CYAN + f"Creating feedback for participants who don't already have feedback.")
        feedback = 'REMAINING'
        list_ids = []
    if number_feedback == 3:
        print(Fore.CYAN + f'Creating feedback for specified participants. \n')
        participant_id = args.participant_id or input(
            "Enter the participant ID(s) for which you want to create feedback (comma separated):  ")
        feedback = 'IDS'
        list_ids = [id.strip() for id in participant_id.split(",")]

    # Creating list of participants to create feedback for
    participant_paths, list_ids = read_file(list_ids, feedback)

    # Formatting and printing text to display who feedback is created for
    count_participants = len(list_ids)
    print(Fore.CYAN + f'\nNumber of participants feedback will be created for: {count_participants}')
    print("\n" + "+" * 100 + "\n")


    # =========================================================================================================================================== #
    # --- CREATING FEEDBACK FOR IDs --- #
    for u, id in enumerate(list_ids, start=1):
        code_start = time.time()
        file_path = participant_paths.get(id)
        if not file_path or not os.path.exists(file_path):
            print(Fore.RED + f'There is no data for the following ID:' + Fore.YELLOW + f'    {id}' + Fore.RED + '.    Feedback will not be created for this ID.')

        else:
            print(Fore.CYAN + f"Creating feedback for participant {u} of {count_participants}:" + Fore.YELLOW + f'  {id}')

            # Extracting start and end wear times
            start_time, end_time, heartrate_df = heartrate_wear_time(file_path)
            #print(start_time)
            #print(end_time)

            # Collapsing HR data to minute level
            collapsed_hr_df = collapse_df(heartrate_df)

            # Trimming HR data to start and end wear times
            trimmed_hr_df = trim_file(collapsed_hr_df, start_time, end_time)

            # Reading acc data into memory
            acc_df = read_acc_file(file_path)

            # Creating ENMO variables
            acc_df = calculate_enmo(df=acc_df)

            # Collapsing acc data into minute level
            collapsed_acc_df = collapse_df(acc_df)

            # Removing noise from accelerometer data
            still_hour_enmo_mean = remove_noise(collapsed_acc_df)
            #print(still_hour_enmo_mean)

            # Trimming acc file to start and end wear times
            trimmed_acc_df = trim_file(collapsed_acc_df, start_time, end_time)


            # Merging acc and HR data
            merged_df = merge_data(trimmed_hr_df, trimmed_acc_df, still_hour_enmo_mean)

            # Plotting HR and Acc
            collapsed_file_path = os.path.join(collapsed_data, f'{id}_collapsed.csv')
            merged_df = pd.read_csv(collapsed_file_path)
            num_days, height_mm = combine_barplot_lineplot(merged_df)

            # Summarising heart rate:
            min_hr, max_hr, mean_hr = sum_heartrate(merged_df)

            # Reading in dailies file and plotting steps
            read_dailies(file_path)

            # Creating PDF with participant feedback in
            create_pdf(feedback_folder, id, num_days, height_mm)

            code_duration = time.time() - code_start
            print(f'{id} processed in {code_duration:.2f} seconds')

            print(Fore.GREEN + f"\n {'-' * 30} Feedback is created for {id} {'-' * 30}\n")
        print("\n" + "+" * 100 + "\n")
    print("\n" + Fore.YELLOW + "The script has finished running and feedback has been created for all specified participants. Press ENTER to close the script.")