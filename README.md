The installer and the service currently works for Windows.

To test the installer, simply download WindowsInstaller.exe and run it on a Windows machine. It will install as a service, but at the moment will "time out" due to no response received (this problem is being worked on).

Currently, the installer only uses two files: main.js (main.exe) and the binary for mod_sat_runner. Authentication and the updater functionality are being worked on, so the other files are works in progress.

To try building the files yourself, simply put the javascript code in a script, and use "npm install" to install the required modules. After doing so, you can use the pkg module to package the script into an executable (that's what I did).

To create the installer yourself, install Inno Setup and use the script I have in the "InnoSetup" txt file, but make sure to modify the paths to be directed to the executables on your computer. After doing so, just build the script and the installer will be in an output folder.


