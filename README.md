# Chess-AI

A machine learning model trained on data from my own online chess games on [Chess.com](https://www.chess.com/home). The resulting AI engine should play similar moves to me.
Game data was downloaded in PGN format from https://www.openingtree.com/. To train your own model using this framework, download your PGN games from the link and place them in the /games folder.

I used a CNN architecture with 3 convolution layers. The input tensor is of shape 19x8x8, where the 8x8 are the board representation and the 19 channels correspond to:

- [0-11] - Piece types (e.g. channel 0 is the board occupancy with white pawns, where 1 = white pawn at this square, and 0 = no white pawn)
- [12-15] - Castling rights (white/black, kingside/queenside)
- [16] - En-passant captures
- [17] - Turn indicator (1 = white's turn, 0 = black's turn)
- [18] - Check indicator (1 means the current side to move is in check)

I have also integrated the resulting model into my C++ [Chess](https://github.com/jordi-lete/Chess) application. At the time of writing, this is contained within a seperate `PvAI` branch of that repository.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/jordi-lete/Chess-AI.git
cd Chess-AI
```

### 3. Create a virtual environment (Recommended)

I am using Python 3.13.3.

```bash
python -m venv venv
```

On Windows:
```bash
venv\Scripts\activate.ps1 #Powershell
venv\Scripts\activate.bat #Bash
```

On Linux:
```bash
source venv/bin/activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Download your chess games

Go to: https://www.openingtree.com/. Place the resulting PGN files into the /games folder.

### 5. Train the model

Run the `train.ipynb` Jupyter Notebook, editing the filepath for the games to your PGN files.

### 6. Test the model

Run the `predict.ipynb` Jupyter Notebook, editing the model name to the name you saved your model to in the previous step.

