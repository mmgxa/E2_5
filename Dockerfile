FROM python:3.8-slim-buster

WORKDIR /app

COPY ["req.txt", "app.py", "cifar10_classes.txt", "./"]

RUN pip install --no-cache-dir -r req.txt

EXPOSE 80

ENTRYPOINT ["streamlit", "run", "./app.py", "--server.port",  "80"]
