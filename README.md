# "Pytorch gang quote generator"

This project is an implementation of a text generator using a neural network. 
The program trains the model on a given set of texts and allows you to generate new texts 
based on the initial phrases entered by the user.

Don't pay attention to the name, it was just originally conceived to create a generator of "cool" teenage quotes 
(a hint at Russian memes with a wolf and a Statham) to tease a teenage nephew

This project is an implementation of a text generator using a neural network. 
The program trains the model on a given set of texts and allows you to generate new texts 
based on the initial phrases entered by the user.

## Installation

1. Install Python 3.x on your computer if it is not already installed.
2. Clone the repository using the following command:
   ```
   git clone https://github.com/fidarit/pytorch-gang-quote-generator.git
   ```
3. Navigate to the project directory:
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare a text dataset on which the model will be trained. Place the text file in the `data` directory with the name `dataset_main.txt`.
2. Train the model by running the `train.py` script:
   ```
   python train.py --model model.pth --dataset data/dataset_main.txt
   ```
3. Generate text by running the `test.py` script:
   ```
   python test.py --model model.pth
   ```
   After running the script, you will be able to enter initial phrases and see the generated text based on the model.

## Parameters

The project provides the following parameters:

- `--model` - path to the model file (default: `model.pth`).
- `--max-epochs` - the maximum number of training epochs (default: 256).
- `--batch-size` - the size of the training data packet (default: 1024).
- `--sequence-length` - the length of the training data sequence (default: 128).
- `--dataset` - path to the file with the set of texts (by default: `data/dataset_main.txt`).
- `--cpu` - flag to use CPU instead of GPU (default: False).

You can configure these parameters by passing them as command line arguments when running scripts.

## Disclaimer

The author of the project is not responsible for the content of the generated text.
The text generator works on the basis of a trained model and can generate texts
that are not correct or meet the user's requirements.
