chmod 755 /usr/local/bin/main.js
chmod 755 /Applications/Authenticator.app

launchctl load /Library/LaunchDaemons/com.aitkn.skedMacInstaller.plist

echo "Post-installation steps completed."
