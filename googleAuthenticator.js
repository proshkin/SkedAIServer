const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');
const { BrowserWindow, app } = require('electron');

const supabaseUrl = 'https://jfcurpgmlzlceotuthat.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpmY3VycGdtbHpsY2VvdHV0aGF0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTY4MzUwMjY5NiwiZXhwIjoxOTk5MDc4Njk2fQ.qrpXSLzhYTMnjCxXWonjCScKthRvS4vJQMtEjvzcBy4';
const supabase = createClient(supabaseUrl, supabaseKey);

async function authenticateUser(accessToken, refreshToken) {
  try {
    const session = await supabase.auth.setSession({
      access_token: accessToken,
      refresh_token: refreshToken,
    });
    const user = session.data.user;
    console.log(user);

    const appDataDir = app.getPath('appData');
    const appSpecificDir = path.join(appDataDir, 'Node Server');

    // ensure the directory exists
    await fs.promises.mkdir(appSpecificDir, { recursive: true });

    const filePath = path.join(appSpecificDir, 'token.txt');

    await fs.promises.writeFile(filePath, accessToken + '.' + refreshToken, {
      flag: 'wx',
    });
    console.log('File has been created successfully.');
  } catch (error) {
    console.error(error);
  }
}

function createAuthWindow() {
  const authWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      partition: 'temp', // Each window will have its own session.
    },
  });

  const authUrl = `https://jfcurpgmlzlceotuthat.supabase.co/auth/v1/authorize?provider=google`;
  authWindow.loadURL(authUrl);

  authWindow.webContents.on(
    'did-redirect-navigation',
    async (event, url) => {
      const urlObj = new URL(url);
      const hashParams = new URLSearchParams(urlObj.hash.substring(1)); // Remove the leading '#'
      const accessToken = hashParams.get('access_token');
      const refreshToken = hashParams.get('refresh_token');

      if (accessToken && refreshToken) {
        try {
          await authenticateUser(accessToken, refreshToken);
          authWindow.close();
        } catch (error) {
          console.error('Authentication error:', error);
        }
      }
    }
  );
}

app.on('ready', createAuthWindow);

