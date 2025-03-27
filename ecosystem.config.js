require('dotenv').config({ path: '.env' });
const RANDOM_SUFFIX = require('child_process').execSync("cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 4 | head -n 1").toString().trim();
const PROJECT_NAME = `templar-test`;

module.exports = {
    apps: [
        {
            name: "TM0",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name miner --wallet.hotkey M0 --device cuda:0 --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"`
        },
        {
            name: "TM1",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name miner --wallet.hotkey M1 --device cuda:1 --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"`
        },
        {
            name: "TM2",
            script: "neurons/miner.py",
            interpreter: "python3",
            env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name miner --wallet.hotkey M2 --device cuda:2 --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"`
        },
        {
            name: "TV1",
            script: "neurons/validator.py",
            interpreter: "python3",                                                                                                              env: {
                ...process.env,
                PROJECT_NAME: PROJECT_NAME
            },
            args: `--wallet.name validator --wallet.hotkey default --device cuda:2 --subtensor.network local --netuid 2 --use_wandb --project "${PROJECT_NAME}"`
        }
    ]
}
