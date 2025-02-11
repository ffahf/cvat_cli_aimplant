1.Create virtual env
```
python -m venv venv
```
2. Activate virtual env (if error occur, lookup Set-ExecutionPolicy)
``` 
.\venv\Scripts\Activate.ps1
```
3. Install cvat-cli interface
``` 
pip install cvat-sdk[pytorch] cvat-cli
```
4. Enter password
```
$ENV:PASS = Read-Host -MaskInput
```
5. install other required lib eg ultralytics
6. Connect with api and using model to annotate the specified task.
```
 cvat-cli --server-host app.cvat.ai --auth <username> auto-annotate <task_id> --function-file path/to/aa.py
```

for more info about cvat-cli please visit
https://docs.cvat.ai/docs/api_sdk/cli/
