const chokidar = require('chokidar');
const os = require('os');
const fs = require('fs');
const path = require('path');
const {createClient} = require("@supabase/supabase-js");


// Get the correct path for appdata in the current OS
const appDataPath = process.env.APPDATA || (os.platform() == 'darwin' ? path.join(process.env.HOME, 'Library', 'Application Support') : '/var/local');

// Define the path of the file to watch
const tokenFilePath = path.join(appDataPath, 'Node Server', 'token.txt');

// Initialize watcher
const watcher = chokidar.watch(tokenFilePath, {
    persistent: true,
    ignoreInitial: false
});

// Add event listeners
watcher
  .on('add', path => handleFileChange(path))
  .on('change', path => handleFileChange(path));

async function authenticateUser(accessToken, refreshToken, supabase) {
  try {
    const session = await supabase.auth.setSession({
      access_token: accessToken,
      refresh_token: refreshToken,
    });
    const user = session.data.user;

    console.log(user);

    // writing to file on user change
    // const usernameFilePath = path.join(appDataPath, 'Node Server', 'username.txt');
    // fs.writeFileSync(usernameFilePath, JSON.stringify(user));

  } catch (error) {
    console.error(error);
  }
}


async function handleFileChange(path) {
    // stopPythonScript;
    console.log(`File ${path} has been added or changed`);

    try {
        const data = fs.readFileSync(path, 'utf8');

        const supabase = createClient('https://jfcurpgmlzlceotuthat.supabase.co', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpmY3VycGdtbHpsY2VvdHV0aGF0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTY4MzUwMjY5NiwiZXhwIjoxOTk5MDc4Njk2fQ.qrpXSLzhYTMnjCxXWonjCScKthRvS4vJQMtEjvzcBy4');
        let lastPeriodIndex = data.lastIndexOf('.');
        let accessToken = data.substring(0, lastPeriodIndex);
        let refreshToken = data.substring(lastPeriodIndex + 1);
        console.log("Access Token: " + accessToken + "\nRefresh Token: " + refreshToken);

        try {
          let curUser = await authenticateUser(accessToken, refreshToken, supabase);
          return curUser;
        } catch (error) {
          console.error('Authentication error:', error);
        }

    } catch (err) {
        console.error(`Error reading file ${path}:`, err);
    }


    /*
    const userID = "80f5f127-149c-45f0-ba6c-8fc833d792b9";

    const supabase = createClient('https://jfcurpgmlzlceotuthat.supabase.co', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpmY3VycGdtbHpsY2VvdHV0aGF0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTY4MzUwMjY5NiwiZXhwIjoxOTk5MDc4Njk2fQ.qrpXSLzhYTMnjCxXWonjCScKthRvS4vJQMtEjvzcBy4');

    const macAddress = getMACAddress();
    const cpuCores = os.cpus().length;
    const ipAddress = getIPAddress();

    checkAndAddServer().then(() => {
        console.log("Server added")
    });

    // create a new solver queue with a max size defined by the num_instances
    const solverQueue = new SolverQueue(2);

    addSolver().then(() => {
        console.log("Solver added")
    });

    subscribeToSolverQueue().then(() => {
        console.log("Subscribed to solver_queue");
    });

    let pythonProcesses = new Map();
     */


}