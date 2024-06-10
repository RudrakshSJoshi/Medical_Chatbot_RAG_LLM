

1. First download folder `chroma` from this [link](https://drive.google.com/file/d/11Jp8miWxfwUDTQq2WVujsKyGHmZi_0RV/view?usp=sharing).
2. Then clone the repo:
```bash
git clone <repo_url>
```
3. And then add `chroma` folder after extracting it in the root directory only.

4. Create a virtual environment using:
```bash
python -m venv < env_name >
```
5. Activate it:
- On Windows:
```bash
 < env_name >\Scripts\activate
```
- On macOS/Linux:
```bash
source < env_name >/bin/activate
```

6. Install required packages:
```bash
pip install -r requirements.txt
```

7. Setup your environment file by creating a `.env` file and adding your API keys as mentioned in `.env.example`.

8. Then run the `main.py` file.

**Additional Steps:**

9. In case you want to configure local-based or OpenAI-based LLMs (such as `llama2`, `mistal7b`, `gpt-4`, etc.) and/or local embeddings function (using `OllamaEmbeddings`, `OpenAIEmbeddings`, etc.), you may do so by configuring model and LLM names written in `main.py`, `db_update.py`, and `testing.ipynb`.

10. You can also use the `testing.ipynb` file to run a temporary storage RAG-based LLM model.

**Additional config:**
- You can set up your database and model by adding your PDF files in the `data` folder (remove the current PDF files from the `data` folder), for that you do not need to download the `chroma` folder, and run `db_update.py`.
- In case the number of documents exceeds 5000, add the files to your database in batches.

To add more to the current model, just add whatever files you want once you remove the current PDF files in the `data` folder into the `data` folder and run `db_update.py`.
