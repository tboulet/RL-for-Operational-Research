@echo off


:: Perform run
mkdir logs
cd C:\Users\33771\Documents\RL\projet\RL-for-Operational-Research

:: The different environment 
set env_tags="ks_medium_deep flp_medium_deep bp_medium_deep"

:: The algo used
set algo_tag="deep_q"

:: Whether to use wandb
set do_wandb="True"

:: Create a directory to store the logs
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"

set "datestamp=%YYYY%%MM%%DD%_%HH%%Min%%Sec%"
mkdir "logs\run_%datestamp%"

:: Iterate over each env tag
for %%a in (%env_tags%) do (
    mkdir "logs\run_%datestamp%\%algo_tag%\%%a"
    :: Iterate over seeds from 0 to 9
    for /l %%x in (0, 1, 9) do (
        :: Run the command with the current env tag, algo tag, seed, and do_wandb value
        python run.py --config-name benchmark.yaml env=%%a algo=%algo_tag% seed=%%x do_wandb=%do_wandb% > "logs\run_%datestamp%\%algo_tag%\%%a\seed_%%x.log"
    )
)