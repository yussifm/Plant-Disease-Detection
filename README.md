pip install click Flask gunicorn itsdangerous Jinja2 MarkupSafe numpy pandas Pillow python-dateutil pytz six typing-extensions Werkzeug -f https://download.pytorch.org/whl/torch_stable.html torch torchvision


pip install waitress
waitress-serve --port=8000 app:app


http://localhost:8000/

flask --app app run --port=8000 --reload