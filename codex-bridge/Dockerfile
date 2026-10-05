FROM python:3.12-slim
WORKDIR /app
COPY bridge.py responses_adapter.py anthropic_adapter.py LICENSE ./
ENV PYTHONDONTWRITEBYTECODE=1
USER 10001:10001
EXPOSE 7864
CMD ["python", "-u", "bridge.py"]
