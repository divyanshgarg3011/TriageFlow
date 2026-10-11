FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# Pass ANTHROPIC_API_KEY with `docker run -e ANTHROPIC_API_KEY=...`; without it the app runs in demo mode.
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
