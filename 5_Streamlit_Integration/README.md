# Segment 5 - Streamlit Integration

Segment 5 is a presentation-only, light-theme Streamlit workspace. It imports `ZendsResponseEngine` from Segment 4 and displays the returned analysis, source chunks, and recommended response without reimplementing AI logic.

## Run locally

From the project root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run 5_Streamlit_Integration/app.py
```

The first submitted query may load the existing Segment 2/3 resources and the pretrained Segment 4 FLAN-T5 Small adapter. Streamlit caches the response engine for the lifetime of the app process.

## Architecture

`Streamlit query -> Segment 4 ZendsResponseEngine.respond() -> Segment 2 analysis + Segment 3 knowledge + Segment 4 grounded response -> Streamlit presentation`

The interface displays source and page references, but not embeddings, retrieval distances, vector-store objects, or internal prompts. If Segment 4 abstains, the app makes that insufficiency visible rather than inventing a company fact.

## Deployment preparation

- Entry point: `5_Streamlit_Integration/app.py`
- Install dependencies from the root `requirements.txt`.
- Run from the project root so the existing relative model and Chroma paths resolve correctly.
- No secrets or machine-specific absolute paths are used.
- Deploy by creating a service that installs `requirements.txt` and runs `streamlit run 5_Streamlit_Integration/app.py --server.address 0.0.0.0 --server.port $PORT`.

## Known limitations

- The app intentionally inherits the locked Segment 2 classifications and Segment 3 retrieval behavior.
- A known broad privacy query safely abstains because the required privacy evidence is not in Segment 3's top-four result set.
- The application is an agent-support recommendation workspace, not a deployed customer-facing service or human-approval system.
