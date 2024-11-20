# no-lose-janken-ai

## 前提条件

- **OS**: Windows 11  
- **CUDAバージョン**: 12.4  

---

## 使用方法

1. **仮想環境の作成**  
   以下のコマンドで仮想環境を作成します。(順番は以下のコードの通りにしてください。)  

   ```bash
   python -m venv venv  
   venv\Scripts\activate.bat  
   pip install -r requirements.txt  
   pip install -r requirements_dev.txt  
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
3. **バックエンドの確認**  
   以下のコマンドでバックエンド上の予測モデルを確認できます。
   ```bash
   python -m backend.main
   ```
3. **アプリの起動**  
   以下のコマンドでアプリを実行します。
   ```bash
   streamlit run app.py
   ```
