# ALPR OCR Model Selector — installation

This update adds a dropdown for the three OCR model generations to the existing ALPR web app. It also remembers results for the **same uploaded image**, so you can compare the text recognized by each generation.

The **plate detector does not change**. The existing `POST /api/recognize` API still works without new parameters; it defaults to Generation 3.

## 1. Install the source files

Extract `ALPR_Model_Selector_Update.zip` into `C:\ALPR` (not into `C:\ALPR\alpr_webui`). Allow it to overwrite **only**:

- `C:\ALPR\alpr_webui\main.py`
- `C:\ALPR\alpr_webui\static\index.html`

Back up the two existing source files first if you have local changes.

Example in PowerShell, if the ZIP is in Downloads:

```powershell
cd C:\ALPR
Copy-Item .\alpr_webui\main.py .\alpr_webui\main.py.bak
Copy-Item .\alpr_webui\static\index.html .\alpr_webui\static\index.html.bak
Expand-Archive -Path "$env:USERPROFILE\Downloads\ALPR_Model_Selector_Update.zip" -DestinationPath 'C:\ALPR' -Force
```

## 2. Install OCR generations

Use the weights from your completed training runs:

```powershell
$models = 'C:\ALPR\alpr_webui\models'
New-Item -ItemType Directory -Force $models | Out-Null

# Generation 1: Spanish
Copy-Item 'C:\ALPR\runs\character_reader\weights\best.pt' "$models\character_reader_gen1.pt" -Force

# Generation 2: Spanish + Danish
Copy-Item 'C:\ALPR\runs\character_reader_dk-2\weights\best.pt' "$models\character_reader_gen2.pt" -Force

# Generation 3: Spanish + Danish + German
# The existing character_reader.pt is kept unchanged by this update.
# If needed, copy best.pt from your latest character_reader_dk_de* run here.
Get-ChildItem $models\*.pt | Select-Object Name, Length
```

Required names:

```text
alpr_webui/models/
    plate_detector.pt         (existing plate detector, unchanged)
    character_reader_gen1.pt  (Generation 1)
    character_reader_gen2.pt  (Generation 2)
    character_reader.pt       (Generation 3, original API default)
```

If your training runs have a different numbered suffix, adjust the source path. The API automatically disables generations whose weights are missing.

## 3. Run the app

```powershell
cd C:\ALPR\alpr_webui
& C:\ALPR\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>.

1. Upload one vehicle photo.
2. Select Generation 1 and click **Recognize plates**.
3. Select Generation 2 and repeat.
4. Select Generation 3 and repeat.
5. Compare the recognized texts in the table. The image preview always shows the currently selected generation's result.

**You do not need to restart FastAPI when changing generations in the dropdown.** Models are lazily loaded and cached. Restart only after replacing the actual weight files on disk.

## 4. API usage

`GET /api/models` returns the available generations.

`POST /api/recognize` accepts a file and an optional form field `ocr_model` (`gen1`, `gen2`, `gen3`).

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/recognize -F 'ocr_model=gen2' -F 'file=@C:\ALPR\test_car.jpg'
```

The JSON response now also includes `ocr_model` and `ocr_model_name`. Existing fields remain unchanged.

## 5. Commit to GitHub (optional)

Your `.gitignore` excludes `.pt` weights. Add these explicitly with Git LFS **only if you want to distribute the three generations**:

```powershell
cd C:\ALPR
git lfs install
git lfs track 'alpr_webui/models/*.pt'
git add .gitattributes alpr_webui/main.py alpr_webui/static/index.html
git add -f alpr_webui/models/character_reader_gen1.pt alpr_webui/models/character_reader_gen2.pt
git lfs ls-files
git commit -m 'Add OCR generation selector and comparison UI'
git push origin main
```

Only `character_reader.pt` and `plate_detector.pt` were previously used by the web app. Both remain at their original filenames; this update **does not contain any trained `.pt` weights**.

> Note: this is an inference comparison, not a formal accuracy evaluation. To evaluate generations objectively, test them on labeled, unseen images and record full-plate exact-match rates for each country.
