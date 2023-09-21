### Как установить и запустить
1 - Активировать и заупстить python виртуальное окружен в корне проекта (venv)
2 - Изменить 'base_dataset' на строке 00 в файле main.py
```
posts = pd.read_excel('./base_dataset.xlsx')
```

## Получить отфильтрованный датасет
Запустить FastAPI
```
uvicorn main:app --load
```
И перейти по адресу http://127.0.0.1/get-filtered