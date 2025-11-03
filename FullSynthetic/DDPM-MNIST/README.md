# FullSynthetic DDPM-MNIST MADness Experiment

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- ~10GB free disk space for generated data
- Git for cloning the repository

### Step 1: Clone the Repository
```bash
git clone https://github.com/PatrickBats/MAD-Detection.git
cd MAD-Detection
```

### Step 2: Activate Python Environment
The repository includes a virtual environment. Activate it using:

**Windows (PowerShell):**
```bash
.\activate.ps1
```

**Note:** If you need to create a new virtual environment instead:
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
Install all necessary packages including PyTorch with CUDA support:
```bash
pip install -r requirements.txt
```

### Step 4: Navigate to right Dir

### If in Windows
```bash
cd FullSynthetic/DDPM-MNIST
```


### Step 5: Run the Main Experiment
```bash
python main.py
```
### During Execution
The script will run 20 generations (Generation 0 through 19), where each generation:

1. **Training Phase** 
   - Loads 60,000 synthetic images from the previous generation
   - Trains a new DDPM model from scratch for 40 epochs

2. **Generation Phase** 
   - Generates 60,000 new synthetic samples for the next generation
   - This is the slowest part due to 500 denoising steps per image

3. **Evaluation Phase** 
   - Generates 1,000 samples for quality metrics
   - Computes FID, Precision, Recall, Density, Coverage
   - Displays a plot showing metric trends (close window to continue)
   - Creates t-SNE visualization comparing real vs synthetic data

### Output Files
Each generation creates:
- `data/diffusion_outputs10/model_{generation}_w0.pth` - Trained model checkpoint
- `data/diffusion_outputs10/gen_data_with_w{generation}_w0` - Generated synthetic dataset
- `data/diffusion_outputs10/All-genration={generation}w=0.png` - t-SNE visualization





