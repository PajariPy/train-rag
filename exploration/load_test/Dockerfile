FROM python:3.13-slim
RUN pip install --no-cache-dir fastapi==0.162.0 "uvicorn[standard]==0.54.0" bm25s==0.3.16 \
    PyStemmer==3.1.0 numpy==2.5.4 scipy==1.18.1
WORKDIR /app
COPY server.py /app/server.py
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"]
