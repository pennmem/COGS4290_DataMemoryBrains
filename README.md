# Big Data, Memory, and the Human Brain (COGS 4290/PSYC 4290)

These are the assignments and supporting material for Michael Kahana's course on "Big Data, Memory, and the Human Brain" (COGS 4290/PSYC 4290) at the University of Pennsylvania. They form a solid introduction to performing EEG analyses and include many resources for learning these tools and methods beyond the homework assignments. These assignments additionally provide a resource for getting up to speed on the research done in Mike's lab.

# Course Structure
Our goal is to familiarize you with fundamental concepts in human memory and electrophysiology as well as with programming tools needed for the large-scale computing in these fields. The psychology and neuroscience at play in these analyses will be primarily covered in the course lectures. To that end, the course outline is as follows:

* Assignment 0: **Python, Numpy, Pandas, and Plotting**

By the end of this course, you should be able to carry out EEG/iEEG/ECoG analyses, like computing spectral power and phase, and to compute statistics or to apply machine learning models to those data.

For **Rhino** compute cluster functionality:
* Introduction 1 - **Introduction to Python, Jupyter, Numpy, and pandas**
    * Useful introduction to standard tooling for using Python for data science.

These notebooks prepare you for doing in-depth multi-subject analyses with electrophysiological data. In particular, we recommend Appendix 1 for anyone unfamiliar with the Python libraries (including Numpy and Pandas) used extensively in this course. Though this course assumes a basic knowledge of Python and command line tools, we have linked recommended resources in **Introduction 1** for getting started with Python and common data analysis tools. Though this material isn't strictly part of the course, we recommend reviewing it before proceeding to the materials included here unless you are confident in your experience with numpy, pandas, scipy, and basic python syntax. If those words don't mean anything to you (or you want to brush up), please read through these resources! 

Additional Introductions and Assignments will be released as the course progresses.

# Initial Setup

To start working with any materials contained or linked here, you'll need to
set up tools for writing and running Python code. If you are affiliated with
the Computational Memory Lab and have access to Rhino, the CML computing cluster,
you can skip down to the Getting started on Rhino section. Otherwise, you
can follow the instructions below to set up python on your own computer.

## Command line access

All subsequent stages of these instructions will assume familiarity with and access
to a *NIX command line. If this is unfamiliar to you, please use the resources below
to get yourself oriented.

If you are using Rhino, an apple computer running OSX, or a Linux computer,
you will already have access to a command line. On Windows, we recommend
using Cygwin <https://www.cygwin.com/> or the Ubuntu subsystem
<https://docs.microsoft.com/en-us/windows/wsl/install-win10>.

General Introduction: https://ubuntu.com/tutorials/command-line-for-beginners#1-overview

## Getting started on Rhino

These instructions will help you access and setup your account on the Rhino computing cluster 
to the point where you can follow these notes and perform analyses. 

### Setting up your Rhino2 Account

1\. You can log in to Rhino2 in a terminal window by using any ssh client
to ssh into rhino as follows, replacing the "username" with your username:

    ssh username@rhino2.psych.upenn.edu

and then typing your temporary password when prompted. Once successfully
connected, type:

    passwd

to set your password to something only you know.

## Getting the course GitHub repository

In a terminal in the location where you would like to download these course assignment
materials, enter the following:

    git clone https://github.com/pennmem/COGS4290_DataMemoryBrains.git

    If git is not installed, you can find instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

This respository will be downloaded to a folder named DataMemoryBrain in the
same location where you ran the git clone command.

## Accessing Jupyter Lab

Once you have your password set up, check to be sure you can log in to
JupyterLab, where you'll be doing most of your coursework. If you are
connected to the internet on UPenn's campus, you only need to go to
[https://rhino2.psych.upenn.edu:9500](https://rhino2.psych.upenn.edu:9500) to
access JupyterLab. If you are connecting remotely, follow the rest of this
step. In a terminal where ssh is accessible, replace the "username" with your
username, and open an ssh tunnel by typing:

    ssh -L8000:rhino2.psych.upenn.edu:9500 username@rhino2.psych.upenn.edu

followed by entering your rhino password. In your web browser, navigate to:

[https://127.0.0.1:8000](https://127.0.0.1:8000)

and you should see the JupyterLab interface pop up!  Note that the "s" on https is critical for this to work.  Your browser might warn about this being an insecure connection or invalid certificate, given that 127.0.0.1 (direct to the ssh tunnel on your own computer) is not rhino.  Override this warning and connect anyway, because we are using ssh to provide better security here.  If the connection still fails, go back and make sure that your ssh tunnel was correctly created.

Alternatively, you can use the Penn VPN service GlobalProtect ([https://www.isc.upenn.edu/how-to/university-vpn-getting-started-guide](https://www.isc.upenn.edu/how-to/university-vpn-getting-started-guide)) to access rhino while off-campus as if you were on-campus, i.e. at [https://rhino2.psych.upenn.edu:9500](https://rhino2.psych.upenn.edu:9500). This remote connection method can be stabler than connecting via the SSH tunneling option.

## Setting up your environment (Rhino)
Good news! Working on rhino gives you access to a computing environment that already has the right software installed to do all the assignments!

In JupyterLab, open any notebook and then go to Kernel -> Change Kernel... and then select "workshop" from the dropdown! Make sure you use this kernel whenever you're opening a notebook.

## Getting started on your computer

In this course, you will have access to the CML compute cluster, Rhino. However,
if you for whatever reason need to work locally, we provide the following guidance. 
We use conda to manage the various libraries needed to perform
analyses using Python. Conda is a tool that allows Python libraries to be installed into
'environments.' This is a folder that lets you manage the needs of different projects
independently; the reasons for this may not be apparent immediately, but using some
sort of virtual environment system of some sort is a standard practice and isolates issues
when they come up. Conda is available from the [Anaconda project home](https://docs.conda.io/projects/conda/en/latest/index.html). We
recommend installing miniconda, though you can read the installation instructions and
decide for yourself which distribution is best for you.

Once you have conda set up, we need to additionally set up Jupyter notebooks. This is
a tool that makes some types of python development easier since it allows you to
run small pieces of code and immediately see the output alongside the code. 
Installation instructions and general information are available from the 
[Jupyter project home](https://jupyterlab.readthedocs.io/en/stable/index.html).

## Setting up your environment (non-Rhino / local computer)

Once you've installed the necessary tools,
you'll need to create a new virtual environment. To do so, open a terminal and run:

    conda create -y -n <environmentname> python=3.7
    NOTE: 'environmentname' is a placeholder, please replace it with a more descriptive name!

For commands to alter or refer to this environment, you'll need to activate it.
This step will be necessary any time you open a new terminal or restart your session,
but will be remembered for subsequent commands.

    conda activate <environmentname>
    NOTE: on older versions of conda, you may instead need to use source activate environmentname


Next, you'll need to install a suite of tools for EEG analysis. First,
install MNE by typing the following (be sure you're in the Anaconda
"environment" you just created in Step 1, by typing "source activate
environmentname"). Note that this may take a while, because MNE has a
lot of dependencies:

    conda install -c conda-forge mne

If this does not work at first, try `pip install mne`

Next, install PTSA, which is a set of EEG tools developed by
former lab members:

    conda install "traitlets<5"
    conda install -c pennmem ptsa=2.0.8

Install a few extra packages in use for these notes:

    conda install scikit-learn statsmodels seaborn

Finally, you'll need to link JupyterLab with your specific Python
installation.  While still logged in and in your Anaconda "environment",
type:

    conda install ipykernel

and once that's done:

    python -m ipykernel install --user --name environmentname --display-name "environmentname"

You should be all set! Next time you log in to your JupyterLab account,
you should see an option to launch a new notebook with "environmentname"
as your Python environment. If you've been logged in to JupyterLab this
whole time, you may need to log out and log back in again to see this
change take effect.

To access the data for this course outside of Rhino, contact kahana-sysadmin@sas.upenn.edu.

