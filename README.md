# Introduction
Garmin Feedback is a python project designed to generate participant feedback for participants who have worn a Garmin device as part of a research study. The project can be executed either through PyCharm or via a batch file. 

It is recommended to set up a virtual environment within a Python interpreter (PyCharm is used within the MRC Epidemiology Unit) and run the project through this environment. Follow the user guides for instructions on how to set up, prepare, and run the project using both approaches.

For detailed user guides on setting up and running the script, refer to the repository [wiki](https://github.com/CAS254/Garmin_Feedback/wiki). 

# Prerequisites
- Data from Garmin devices: Heart rate, accelerometer and dailies files in CSV format.
- Python version 3.12 or higher
- PyCharm Community Edition (optional; while the script can also be run through a batch file, installation of PyCharm is recommended as it facilitates setting up a virtual environment and managing dependencies more easily).

### Further notes 
- This process was developed on Windows and has **NOT** been tested on other operating systems, such as macOS.
- The script has been tested with Python version 3.12. Future versions may introduce incompatibilities, but testing and updates will occur as new versions are released.
- The script assumes that data from the Garmin device is stored in participant-specific subfolders, with each subfolder named after the participant's ID.
- The script expects files to follow the naming convention: ```<participant_ID>_heartrate.csv``` (or ```_accelerometer.csv```/```_dailies.csv```). Within the MRC Epidemiology Unit, a separate rename and unzip script ensures files are named correctly after being exported from Fitrockr.  

# Downloading and preparing the environment
### Set up folder structure
Create the following folder structure in your project directory:
- participant_feedback

### Download the code
1. Navigate to the Garmin Feedback repository [here](https://github.com/CAS254/Garmin_Feedback). 
2. Click ![image](https://github.com/user-attachments/assets/587012f2-735e-471e-b7c0-38e7977e36ee) and select **Download ZIP**.
3. Extract the ZIP file into the folder where you want to run the python script, ensuring it extracts to a single level of subfolders.

### Install dependencies
Certain Python modules are required to run the code. A **requirements.txt** file is included within the downloaded ZIP file and contains all necessary dependencies for the project. See the User Guides on [GitHub wiki](https://github.com/CAS254/Garmin_Feedback/wiki) for detailed instructions on installing these dependencies.

# Preparing the script to run
For a description of the Garmin Feedback script, see [Explanation of code](https://github.com/MRC-Epid/Garmin_Feedback/wiki/2.-Explanation-of-code). 

Before executing the script, some variables must be edited to match the specific settings of your study. These variables are located near the beginning of the script and are commented to explain how to edit.

# Output 
The process generates feedback files for participants based on their Garmin data. It produces feedback on their daily heart rate, activity and steps. The feedback files will be saved in the ```feedback``` folder within the ```participant_feedback``` folder and will be in PDF format. 
> Note: The ```feedback``` folder, along with other required subfolders, will be automatically created by the script if they don't already exist.
