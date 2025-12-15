# Ueasys

> Un sistema avanzato di personaggi fantasy viventi basato su RAG (Retrieval-Augmented Generation)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![LightRAG](https://img.shields.io/badge/LightRAG-latest-orange.svg)](https://github.com/HKUDS/LightRAG)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview

Fantasy World RAG è un sistema che permette di interagire con personaggi fantasy autentici che:

- 🎭 **Personalità Autentiche**: Basate su fonti canoniche (libri, leggende, miti)
- 🧠 **Memoria Persistente**: Ricordano le interazioni precedenti
- 🤖 **Comportamento Agentico**: Agiscono autonomamente nel loro mondo
- 📈 **Evoluzione Dinamica**: Crescono nel tempo mantenendo coerenza
- 📱 **Multi-Piattaforma**: Accessibili da web e mobile

---

## ⚡ Quick Start (5 minuti)

### 🖱️ Metodo Facile (Windows)

1. **Scarica e installa i prerequisiti**: Python 3.11+, Node.js 18+, Docker Desktop
2. **Doppio click su** `scripts/SETUP.bat` per l'installazione guidata
3. **Configura le API key** nel file `.env` (vedi sotto)
4. **Doppio click su** `scripts/AVVIA.bat` per avviare tutto!
5. **Apri** http://localhost:5173 nel browser 🎉

### 🔑 API Key Necessarie

| Servizio | Per cosa | Dove ottenerla |
|----------|----------|----------------|
| **Grok (xAI)** | Chat personaggi | [console.x.ai](https://console.x.ai) |
| **DeepSeek** | Analisi documenti | [platform.deepseek.com](https://platform.deepseek.com) |

> 📖 **Guida completa**: Leggi [GUIDA_INSTALLAZIONE.md](GUIDA_INSTALLAZIONE.md) per istruzioni dettagliate passo-passo!

---

## 🚀 Quick Start (Avanzato)

### Prerequisiti

- Python 3.11+
- Docker & Docker Compose
- Poetry (package manager)
- Node.js 18+ (per frontend)

### Installazione

```bash
# 1. Clona il repository
git clone https://github.com/yourusername/fantasy-world-rag.git
cd fantasy-world-rag

# 2. Copia le variabili d'ambiente
cp .env.example .env
# Modifica .env con le tue API keys

# 3. Avvia i servizi con Docker
docker-compose up -d

# 4. Installa le dipendenze Python
poetry install

# 5. Setup del database
poetry run alembic upgrade head

# 6. (Opzionale) Carica i personaggi di esempio
poetry run python scripts/seed_data.py

# 7. Avvia il server backend
poetry run uvicorn src.main:app --reload --port 8000

# 8. Avvia il frontend (nuovo terminale)
cd frontend
npm install
npm run dev
```

### 🎮 Accesso

- **Frontend**: http://localhost:5173
- **API Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Accesso

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:5173

## 🏗️ Architettura

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                          │
│                    (React Web / React Native)                   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                         REST API / WebSocket
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                          API Layer                              │
│              FastAPI + WebSocket Connection Manager             │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                       Services Layer                            │
│    CharacterService │ ConversationService │ WorldService        │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                         Core Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Character  │  │   Memory    │  │      Decision           │ │
│  │   Engine    │  │   System    │  │       Engine            │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                         RAG Layer                               │
│        LightRAG Integration │ Knowledge Indexer │ Retriever     │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                       Data Layer                                │
│      PostgreSQL │ Redis │ Qdrant (Vector DB) │ Neo4j            │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Struttura Progetto

```
fantasy-world-rag/
├── src/                    # Codice sorgente Python
│   ├── api/               # FastAPI routes e WebSocket
│   ├── core/              # Character engine, memory, decision
│   ├── rag/               # LightRAG integration
│   ├── agents/            # Sistema agentico
│   ├── llm/               # LLM providers e templates
│   ├── db/                # Models e repositories
│   ├── services/          # Business logic
│   └── utils/             # Utilities
├── frontend/              # Frontend applications
│   ├── web/              # React web app
│   └── mobile/           # React Native app (Phase 2)
├── data/                  # Character data e world definitions
├── tests/                 # Test suite
├── scripts/               # Utility scripts
└── docs/                  # Documentation
```

## 🎭 Personaggi Disponibili (MVP)

| Personaggio | Origine | Status |
|-------------|---------|--------|
| 🧙‍♂️ Gandalf | Lord of the Rings | ✅ Attivo |
| 🧝‍♀️ Galadriel | Lord of the Rings | ✅ Attivo |
| 🐉 Smaug | The Hobbit | ✅ Attivo |

## 📚 API Reference

### REST Endpoints

```
GET  /api/v1/characters          # Lista personaggi
GET  /api/v1/characters/{id}     # Dettaglio personaggio
POST /api/v1/chat                # Invia messaggio
GET  /api/v1/conversations       # Lista conversazioni
GET  /api/v1/world/events        # Eventi del mondo
```

### WebSocket

```javascript
// Connessione
ws://localhost:8000/ws/chat/{character_id}

// Eventi
- message: Nuovo messaggio
- emotion_update: Cambio stato emotivo
- world_event: Evento nel mondo
```

Vedi [docs/api/rest-api.md](docs/api/rest-api.md) per la documentazione completa.

## 🧪 Testing

```bash
# Esegui tutti i test
poetry run pytest

# Con coverage
poetry run pytest --cov=src --cov-report=html

# Solo unit tests
poetry run pytest tests/unit/

# Solo integration tests
poetry run pytest tests/integration/
```

## 🐳 Docker

```bash
# Avvia tutto
docker-compose up -d

# Solo servizi di supporto (DB, Redis, Qdrant)
docker-compose up -d postgres redis qdrant

# Vedi i log
docker-compose logs -f

# Ferma tutto
docker-compose down
```

## 🔧 Configurazione

Le configurazioni sono gestite tramite variabili d'ambiente. Vedi `.env.example` per tutte le opzioni disponibili.

### Variabili Chiave

| Variabile | Descrizione | Default |
|-----------|-------------|---------|
| `OPENAI_API_KEY` | API key OpenAI | - |
| `DATABASE_URL` | URL PostgreSQL | `postgresql://...` |
| `REDIS_URL` | URL Redis | `redis://localhost:6379` |
| `QDRANT_URL` | URL Qdrant | `http://localhost:6333` |

## 🗺️ Roadmap

### Phase 1 - MVP ✅
- [x] Core character engine
- [x] RAG integration
- [x] Basic API
- [x] Web frontend
- [x] 3-5 personaggi

### Phase 2 - Enhancement 🚧
- [ ] Memoria avanzata
- [ ] Sistema emotivo completo
- [ ] Mobile app
- [ ] 10-15 personaggi

### Phase 3 - Agentic System 📋
- [ ] Comportamento autonomo
- [ ] World simulation
- [ ] Multi-character scenes
- [ ] Advanced UI

## 🤝 Contributing

Vedi [CONTRIBUTING.md](CONTRIBUTING.md) per le linee guida.

## 📄 License

MIT License - vedi [LICENSE](LICENSE) per dettagli.

## 🙏 Acknowledgments

- [LightRAG](https://github.com/HKUDS/LightRAG) - RAG framework
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [LangChain](https://langchain.com/) - LLM orchestration
