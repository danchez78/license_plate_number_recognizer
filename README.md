# License Plate Number Recognizer

Сервис, предназначенный для распознавания регистрационных номеров автомобилей по изображениям.

## Содержание
- [Начало работы](#начало-работы)
- [Использование](#использование)


## Начало работы


Для запуска сервиса нам нужно выполнить следующую команду:

```
pip install -r requirements.txt
```

Далее нужно изменить переменную окружения `DJANGO_SECRET_KEY` в файле `docker.env`. Для того чтобы его получить вам необходимо ввести следующую команду в терминал:

```
python3 src/manage.py shell -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Подставляем его к `DJANGO_SECRET_KEY`

![secret_key.png](images/secret_key.png)

Наконец, запустим наш сервис в docker'е следующей командой:

```
docker-compose up --build; docker-compose logs -f
```

![docker_logs.png](images/docker_logs.png)

## Использование

Допустим есть следующее изображение с автомобилем:

![е060кх177.jpg](images/е060кх177.jpg)

Для распознавания регистрационного номера автомобиля можно воспользоваться библиотекой `requests` в python:

![requests.png](images/requests.png)

Также можно воспользоваться ПО Postman

![postman.png](images/postman.png)



