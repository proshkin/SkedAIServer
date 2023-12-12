const checkForUpdates = () => {
    const options = {
        hostname: 'api.github.com',
        port: 443,
        path: '/repos/{owner}/{repo}/releases',  // replace {owner} and {repo} with actual values
        method: 'GET',
        headers: {'User-Agent': 'NodeJS'}
    };

    const req = https.request(options, res => {
        let data = '';

        res.on('data', chunk => {
            data += chunk;
        });

        res.on('end', () => {
            let releases = JSON.parse(data);

            let latestRelease = releases[0].tag_name;  // The version number of the latest release

            if (latestRelease !== currentVersion) {
                console.log("There is a new version available: " + latestRelease);
                // Here you can handle the update process, like download new version and install it.
            }
            else {
                console.log("You are up to date");
            }
        });
    });

    req.on('error', error => {
        console.error('Error checking for updates:', error);
    });

    req.end();
}

// check for updates every hour
setInterval(checkForUpdates, 60 * 60 * 1000);
checkForUpdates();