# ContextFix - Context-Aware Grammar and Typing Error Recovery System

NLP course project (Unit V: variable-length language models + Bayesian parameter estimation).

## Folder layout
```
ContextFix/
├── app.py                 <- the Streamlit web app (run this)
├── engine.py              <- the NLP engine: language model, spelling, context, grammar
├── ui.py                  <- CSS + HTML helpers (the look of the site)
├── requirements.txt       <- packages to install
├── data/
│   └── seed_corpus.txt    <- extra training sentences (you can add your own!)
└── .streamlit/
    └── config.toml        <- theme colours
```

## Setup (VS Code terminal, inside the ContextFix folder)
```
python -m venv venv
venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
streamlit run app.py
```
The first start downloads two NLTK datasets (Brown corpus + word list) and builds the model (~20 s).
After that, restarts are quick because Streamlit caches the model.

## Optional: test the engine without the website
```
python test_engine.py
```
