# 🔮 Sistema de Previsão de Resultados do Brasileirão (Projeto TrabalhoVital)

Este projeto utiliza aprendizado de máquina (XGBoost) para prever o resultado de partidas do Campeonato Brasileiro com base em dados históricos. Ele expõe uma API usando FastAPI e conta com um frontend (Next.js) que permite ao usuário inserir os times e ver a previsão.

---

## 📦 Estrutura do Projeto

- `BRA.csv`: dataset original com os jogos.
- `features_brasileirao.csv`: dataset com features extraídas.
- `model_training.py`: código para treinar e avaliar o modelo.
- `api.py`: API FastAPI que faz a previsão.
- `extract_features.py`, `data_preprocessing.py`: módulos que tratam e extraem features.
- `.pkl`: arquivos serializados (modelo, encoder, imputer etc).
- `main.py`: pipeline principal para gerar as features, salvar datasets e treinar modelo.
- `predict_single.py`: script para prever um único jogo direto no terminal.

---

## ⚙️ Como Executar Localmente

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/nome-do-repo.git
cd nome-do-repo
```

### 2 Crie e ative o Ambiente Virtual

```
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
````

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
pip freeze > requirements.txt
```

### 4. Gere os dados e treine o modelo

```bash 
python main.py
```
Este script:

Carrega o dataset BRA.csv

Gera o arquivo features_brasileirao.csv

Treina o modelo com XGBoost

Salva os arquivos .pkl necessários

### 5. Inicie a API

```bash
uvicorn api:app --reload
```

Acesse a api em 
```bash
http://127.0.0.1:8000/docs
``` 
