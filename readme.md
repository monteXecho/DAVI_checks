### How to install

- Package installation

```
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

```

```
pip install -r requirements.txt -c constraints.txt
```

- Run FastAPI server

```
uvicorn main:app --port 8801 --host 0.0.0.0
```
