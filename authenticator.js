const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');
const { BrowserWindow, app } = require('electron');

const supabaseUrl = 'https://jfcurpgmlzlceotuthat.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpmY3VycGdtbHpsY2VvdHV0aGF0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDUwODQ4ODksImV4cCI6MjAyMDY2MDg4OX0.7rAa3V9obXlEhewdRah4unY0apsEPHWEYXk5OwKYkLI';
const supabase = createClient(supabaseUrl, supabaseKey);

async function writeTokensToFile(accessToken, refreshToken) {
  try {
    const appDataDir = app.getPath('appData');
    const appSpecificDir = path.join(appDataDir, 'Node Server');

    // ensure the directory exists
    await fs.promises.mkdir(appSpecificDir, { recursive: true });

    const filePath = path.join(appSpecificDir, 'token.txt');

    await fs.promises.writeFile(filePath, accessToken + '.' + refreshToken, {
      flag: 'w',
    });
  } catch (error) {
    console.error(error);
  }
}

async function writeCookieToFile(cookie) {
  try {
    const appDataDir = app.getPath('appData');
    const appSpecificDir = path.join(appDataDir, 'Node Server');

    // ensure the directory exists
    await fs.promises.mkdir(appSpecificDir, { recursive: true });
    const filePath = path.join(appSpecificDir, 'cookie.txt');
    await fs.promises.writeFile(filePath, cookie, {
      flag: 'w',
    });

  } catch (error) {
    console.error(error);
  }
}

function isCookieEmpty() {
  const appDataDir = app.getPath('appData');
  const appSpecificDir = path.join(appDataDir, 'Node Server');
  const filePath = path.join(appSpecificDir, 'token.txt');

  // First check if the file exists
  if (fs.existsSync(filePath)) {
    try {
      const stats = fs.statSync(filePath);
      return stats.size === 0;
    } catch (error) {
      console.error('Error checking file:', error);
      return false; // or handle error as appropriate
    }
  } else {
    // File does not exist
    return true; // Or handle as appropriate for your application
  }
}


function readFileContentsSync(filePath) {
  try {
    const data = fs.readFileSync(filePath, { encoding: 'utf8' });
    return data;
  } catch (error) {
    console.error('Error reading file:', error);
    return null; // Or handle the error as needed
  }
}

function createAuthWindow() {
  const authWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      partition: 'temp',
    },
  });

  if (!isCookieEmpty()) {
    const appDataDir = app.getPath('appData');
    const appSpecificDir = path.join(appDataDir, 'Node Server');
    // const filePath = path.join(appSpecificDir, 'token.txt');
    const filePath = path.join(appSpecificDir, 'cookie.txt');
    // const filePath2 = path.join(appSpecificDir, 'cookie2.txt');
    /* const data = fs.readFileSync(filePath, 'utf8');
    let lastPeriodIndex = data.lastIndexOf('.');
    let accessToken = data.substring(0, lastPeriodIndex);
    let refreshToken = data.substring(lastPeriodIndex + 1);
    const cookie1String = `{"access_token":"${accessToken}","refresh_token":"${refreshToken}"}`;
    console.log("COOKIE 1: " + cookie1String);*/

    const cookie1 = {
      url: 'https://jit-calendar-web.vercel.app',
      name: 'sb-jfcurpgmlzlceotuthat-auth-token.0',
      // value: `{"access_token":"${accessToken}","refresh_token":"${refreshToken}"}`,
      value: readFileContentsSync(filePath),
    };
    Promise.all([
      authWindow.webContents.session.cookies.set(cookie1)
      // authWindow.webContents.session.cookies.set(cookie2)
    ])
    .then(() => {
      const authUrl = `https://jit-calendar-web.vercel.app/service/account`;
      authWindow.loadURL(authUrl);
    })
    .catch(error => {
      console.error('Error setting cookies:', error);
    });
  }
  else {
    const authUrl = `https://jit-calendar-web.vercel.app/service/account`;
    authWindow.loadURL(authUrl);
  }

  // authWindow.webContents.openDevTools();

  // Handle the response from the authentication server.
  authWindow.webContents.on('will-redirect', async (event, url) => {
    // console.log("Redirecting to:", url);
    const urlObj = new URL(url);

    // Check if the URL is the expected redirect URL.
    if (urlObj.origin + urlObj.pathname === 'https://jit-calendar-web.vercel.app/service/account') {

      try {
        const cookie1 = await authWindow.webContents.session.cookies.get({ name: "sb-jfcurpgmlzlceotuthat-auth-token.0" });
        const cookie2 = await authWindow.webContents.session.cookies.get({ name: "sb-jfcurpgmlzlceotuthat-auth-token.1" });
        const cookieStr = decodeURIComponent(cookie1[0].value) + decodeURIComponent(cookie2[0].value);
        const parsedValue = JSON.parse(cookieStr);

        await writeTokensToFile(parsedValue.access_token, parsedValue.refresh_token);
        let modifiedCookie = cookieStr.replace(/("refresh_token":.*$)/, '"refresh_token":"' + parsedValue.refresh_token + '"}');

        // await writeTokensToFile(accessToken, refreshToken, decodeURIComponent(cookies1[0].value), decodeURIComponent(cookies2[0].value));
        await writeCookieToFile(modifiedCookie);

      } catch (error) {
        console.error('Error reading cookie:', error);
      }

    }
  });
}

app.on('ready', createAuthWindow);
