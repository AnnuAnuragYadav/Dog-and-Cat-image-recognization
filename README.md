# Cat vs Dog Streamlit app

## Files

- `app.py` — the Streamlit interface and prediction code
- `requirements.txt` — Python packages the app needs
- `cat_dog_model.keras` — add the trained model downloaded from Colab

## Run locally on Windows

1. Install 64-bit Python 3.11, then open this folder in VS Code.
2. In VS Code, open **Terminal → New Terminal** and run:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Put `cat_dog_model.keras` in this folder beside `app.py`.
4. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

5. Open the local URL printed in the terminal (usually `http://localhost:8501`).

The app resizes uploaded images to 128×128 RGB. Since the saved model already
contains `Rescaling(1./255)`, the app passes pixel values in the 0–255 range and
does not scale them a second time. A sigmoid score of 0.5 or more is reported as
dog; a score below 0.5 is reported as cat.
