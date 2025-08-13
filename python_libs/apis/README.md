# 🚀 Երևանյան Շաուրմա APIs

Այս պանակում կգտնեք FastAPI և Flask-ի օրինակները՝ «Երևանյան Շաուրմա» ռեստորանի թեմայով:

## 📁 Ֆայլերը

- `fastapi_shawarma.py` - FastAPI կիրառություն
- `flask_shawarma.py` - Flask կիրառություն  
- `requirements.txt` - Անհրաժեշտ գրադարանները
- `README.md` - Այս ֆայլ

## 🛠️ Տեղադրում

```bash
# Անհրաժեշտ գրադարանների տեղադրում
pip install -r requirements.txt
```

## ▶️ Գործարկում

### FastAPI
```bash
# Ճանապարհ 1 - uvicorn հրամանով
uvicorn fastapi_shawarma:app --reload --port 8000

# Ճանապարհ 2 - Python ֆայլ կանչել
python fastapi_shawarma.py
```

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API**: http://localhost:8000

### Flask
```bash
# Python ֆայլ կանչել
python flask_shawarma.py
```

- **API**: http://localhost:5000

## 🧪 Թեստավորում

### FastAPI թեստ
```bash
# Ցանկի ստացում
curl http://localhost:8000/menu

# Նոր պատվեր
curl -X POST http://localhost:8000/orders \
  -H "Content-Type: application/json" \
  -d '{"customer_name": "Արամ", "items": ["հավով", "տավարով"]}'
```

### Flask թեստ
```bash
# Ցանկի ստացում
curl http://localhost:5000/menu

# Նոր պատվեր
curl -X POST http://localhost:5000/orders \
  -H "Content-Type: application/json" \
  -d '{"customer_name": "Նարե", "items": ["բանջարեղենով"]}'
```

## 🔍 Տարբերությունները

### FastAPI ✨
- Ավտոմատ API documentation
- Type hints և validation
- Async/await աջակցություն
- Pydantic մոդելներ
- Ժամանակակից և արագ

### Flask 🌶️
- Պարզ և ճկուն
- Ձեռքով validation
- Լայն community
- Microframework
- Փորձառության մեծ բազա

Երկու framework-ը էլ հիանալի են, բայց տարբեր նպատակների համար!
